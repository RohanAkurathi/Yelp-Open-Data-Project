"""Feature engineering for the ranking stage.

We build two numeric tables — one per user, one per business — from a *history*
window of interactions (everything strictly before the period we're predicting).
Behavioural aggregates are always computed from that window so the ranker never
sees the future. A handful of slowly-changing profile fields (account age, elite
status, check-in totals) are pulled from the static dumps and treated as metadata.

The cross-features that compare a specific user to a specific candidate
(star gap, price gap, same-city, category match, ALS score) are added later in
candidates.py, where both sides are in scope.

Together the three groups come to ~40 features; build_feature_store() prints the
exact count.
"""

import json

import numpy as np
import pandas as pd

from src import config as C

USER_FILE = C.DATA_DIR / "yelp_academic_dataset_user.json"
CHECKIN_FILE = C.DATA_DIR / "yelp_academic_dataset_checkin.json"
TIP_FILE = C.DATA_DIR / "yelp_academic_dataset_tip.json"

_COMPLIMENT_COLS = [
    "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
    "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
    "compliment_funny", "compliment_writer", "compliment_photos",
]


# ── static-dump loaders (kept to the cohort to stay in memory) ───────────────────

def load_cohort_users(user_ids: set[str]) -> pd.DataFrame:
    """Stream the 3 GB user file, keep only the cohort's rows."""
    rows = []
    with open(USER_FILE, encoding="utf-8") as fh:
        for line in fh:
            u = json.loads(line)
            if u["user_id"] in user_ids:
                rows.append(u)
    df = pd.DataFrame(rows)
    df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")
    df["is_elite"] = df["elite"].astype(str).str.strip().replace("nan", "").ne("")
    df["total_compliments"] = df[_COMPLIMENT_COLS].sum(axis=1)
    return df.set_index("user_id")


def load_checkins() -> pd.DataFrame:
    """Per-business check-in count and most-recent check-in date."""
    rows = []
    with open(CHECKIN_FILE, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            stamps = c["date"].split(", ") if c.get("date") else []
            rows.append((c["business_id"], len(stamps), stamps[-1] if stamps else None))
    df = pd.DataFrame(rows, columns=["business_id", "checkin_count", "last_checkin"])
    df["last_checkin"] = pd.to_datetime(df["last_checkin"], errors="coerce")
    return df.set_index("business_id")


def load_tip_counts() -> pd.Series:
    counts: dict[str, int] = {}
    with open(TIP_FILE, encoding="utf-8") as fh:
        for line in fh:
            bid = json.loads(line)["business_id"]
            counts[bid] = counts.get(bid, 0) + 1
    return pd.Series(counts, name="tip_count")


# ── attribute parsing ────────────────────────────────────────────────────────────

def _attr_bool(attrs, key) -> int:
    if not isinstance(attrs, dict):
        return 0
    return 1 if str(attrs.get(key)).strip().lower() == "true" else 0


def _price_range(attrs) -> int:
    if not isinstance(attrs, dict):
        return -1
    raw = attrs.get("RestaurantsPriceRange2")
    try:
        return int(float(str(raw).strip("'\" ")))
    except (TypeError, ValueError):
        return -1


def _primary_category(categories) -> str:
    if not categories:
        return "Unknown"
    parts = [c.strip() for c in categories.split(",")]
    # the first category that isn't the generic umbrella label
    for c in parts:
        if c not in ("Restaurants", "Food"):
            return c
    return parts[0]


# ── user features ────────────────────────────────────────────────────────────────

def build_user_features(
    history: pd.DataFrame, biz: pd.DataFrame, users_static: pd.DataFrame
) -> pd.DataFrame:
    """One row per user, computed from their interactions in `history`."""
    window_end = history["date"].max()
    pop = biz.set_index("business_id")[["review_count", "stars", "price_range", "city", "primary_category"]]
    pop = pop.rename(columns={
        "review_count": "biz_rc", "stars": "biz_gs",
        "price_range": "biz_pr", "city": "biz_city", "primary_category": "biz_cat",
    })
    h = history.join(pop, on="business_id")

    g = h.groupby("user_id")
    f = pd.DataFrame({
        "u_n_reviews":      g.size(),
        "u_avg_stars":      g["stars"].mean(),          # stars the user hands out
        "u_std_stars":      g["stars"].std().fillna(0.0),
        "u_pct_positive":   g["stars"].apply(lambda s: (s >= C.POSITIVE_THRESHOLD).mean()),
        "u_avg_review_len": g["review_len"].mean(),
        "u_n_distinct_biz": g["business_id"].nunique(),
        "u_avg_biz_pop":    g["biz_rc"].mean(),         # popularity of places they pick
        "u_avg_biz_stars":  g["biz_gs"].mean(),         # global rating of places they pick
        "u_avg_price":      g["biz_pr"].apply(lambda s: s[s > 0].mean()),
        "u_active_days":    (g["date"].max() - g["date"].min()).dt.days,
    })
    f["u_reviews_per_month"] = f["u_n_reviews"] / (f["u_active_days"] / 30.0 + 1.0)
    # most-frequent city and primary category they engage with
    f["u_modal_city"] = g["biz_city"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    f["u_top_cat"] = g["biz_cat"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    f["u_avg_price"] = f["u_avg_price"].fillna(f["u_avg_price"].median())

    # static profile signals (slowly-changing metadata)
    st = users_static.reindex(f.index)
    f["u_account_age_days"] = (window_end - st["yelping_since"]).dt.days.clip(lower=0)
    f["u_is_elite"] = st["is_elite"].astype(float).fillna(0.0)
    f["u_fans"] = pd.to_numeric(st["fans"], errors="coerce").fillna(0.0)
    f["u_total_compliments"] = st["total_compliments"].fillna(0.0)
    return f


# ── business features ────────────────────────────────────────────────────────────

def build_business_features(
    history: pd.DataFrame, biz: pd.DataFrame, checkins: pd.DataFrame, tips: pd.Series
) -> pd.DataFrame:
    """One row per business, computed from interactions in `history` + metadata."""
    window_end = history["date"].max()
    g = history.groupby("business_id")
    f = pd.DataFrame({
        "b_n_reviews":      g.size(),
        "b_avg_stars":      g["stars"].mean(),
        "b_std_stars":      g["stars"].std().fillna(0.0),
        "b_pct_positive":   g["stars"].apply(lambda s: (s >= C.POSITIVE_THRESHOLD).mean()),
        "b_avg_review_len": g["review_len"].mean(),
        "b_useful_per_rev": g["useful"].mean(),
        "b_age_days":       (window_end - g["date"].min()).dt.days,
    })
    f["b_reviews_per_month"] = f["b_n_reviews"] / (f["b_age_days"] / 30.0 + 1.0)

    meta = biz.set_index("business_id")
    f["b_global_stars"] = meta["stars"]
    f["b_global_review_count"] = meta["review_count"]
    f["b_is_open"] = pd.to_numeric(meta["is_open"], errors="coerce").fillna(0)
    f["b_price_range"] = meta["price_range"]
    f["b_n_categories"] = meta["categories"].fillna("").apply(lambda s: len([c for c in s.split(",") if c.strip()]))
    f["b_city"] = meta["city"]
    f["b_primary_category"] = meta["primary_category"]
    for col, key in [
        ("b_takeout", "RestaurantsTakeOut"),
        ("b_delivery", "RestaurantsDelivery"),
        ("b_groups", "RestaurantsGoodForGroups"),
        ("b_reservations", "RestaurantsReservations"),
        ("b_cards", "BusinessAcceptsCreditCards"),
    ]:
        f[col] = meta["attributes"].apply(lambda a: _attr_bool(a, key))

    ck = checkins.reindex(f.index)
    f["b_checkin_count"] = ck["checkin_count"].fillna(0.0)
    f["b_checkin_per_month"] = f["b_checkin_count"] / (f["b_age_days"] / 30.0 + 1.0)
    f["b_tip_count"] = tips.reindex(f.index).fillna(0.0)

    # popularity percentile within the cohort (handy as a calibrated prior)
    f["b_pop_percentile"] = f["b_n_reviews"].rank(pct=True)
    return f


# ── orchestration ────────────────────────────────────────────────────────────────

def enrich_businesses(biz: pd.DataFrame) -> pd.DataFrame:
    """Add parsed columns the feature builders rely on."""
    biz = biz.copy()
    biz["price_range"] = biz["attributes"].apply(_price_range)
    biz["primary_category"] = biz["categories"].apply(_primary_category)
    return biz


# numeric columns handed to the model (everything except string/helper columns)
USER_NUMERIC = [
    "u_n_reviews", "u_avg_stars", "u_std_stars", "u_pct_positive", "u_avg_review_len",
    "u_n_distinct_biz", "u_avg_biz_pop", "u_avg_price", "u_active_days",
    "u_reviews_per_month", "u_account_age_days", "u_is_elite", "u_fans",
    "u_total_compliments",
]
BUSINESS_NUMERIC = [
    "b_n_reviews", "b_avg_stars", "b_std_stars", "b_pct_positive", "b_avg_review_len",
    "b_useful_per_rev", "b_age_days", "b_reviews_per_month", "b_global_stars",
    "b_global_review_count", "b_is_open", "b_price_range", "b_n_categories",
    "b_takeout", "b_delivery", "b_groups", "b_reservations", "b_cards",
    "b_checkin_count", "b_checkin_per_month", "b_tip_count", "b_pop_percentile",
]
# We pass both the ALS score and the candidate rank as cross-features: the
# collaborative signal is the backbone of a two-stage ranker. LambdaMART then
# optimises NDCG directly (which plain ALS does not) and uses the business-quality,
# user-behaviour, temporal and matching features to rerank within the candidate set.
CROSS_NUMERIC = ["als_score", "cf_rank", "star_gap", "price_gap", "same_city", "same_top_cat"]

FEATURE_COLUMNS = USER_NUMERIC + BUSINESS_NUMERIC + CROSS_NUMERIC
