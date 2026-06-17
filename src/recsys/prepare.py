"""Build the interaction table the recommender trains on.

Pipeline:
    1. Pick the dining businesses in our target metros.
    2. Stream the (5 GB) review file and keep only reviews for those businesses.
    3. Apply iterative k-core filtering to drop cold users and items.
    4. Split chronologically into train / valid / test (a single global cutoff
       date per split, so we never train on the future).

Outputs (parquet, under artifacts/):
    interactions.parquet   user_id, business_id, stars, date, review_len, useful, split
    businesses.parquet      the business rows for the surviving cohort
"""

import json
import time

import pandas as pd

from src import config as C

REVIEW_FILE = C.DATA_DIR / "yelp_academic_dataset_review.json"
BUSINESS_FILE = C.DATA_DIR / "yelp_academic_dataset_business.json"


def _is_dining(categories: str | None) -> bool:
    if not categories:
        return False
    return any(cat in categories for cat in C.DINING_CATEGORIES)


def select_businesses() -> pd.DataFrame:
    """Load businesses, keep dining spots in the target metros."""
    rows = []
    with open(BUSINESS_FILE, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            if _is_dining(b.get("categories")):
                rows.append(b)
    biz = pd.DataFrame(rows)

    cities = C.CITIES
    if cities is None:
        cities = biz["city"].value_counts().head(C.N_CITIES_IF_AUTO).index.tolist()
        print(f"  auto-selected cities: {cities}")
    biz = biz[biz["city"].isin(cities)].copy()
    biz["stars"] = pd.to_numeric(biz["stars"], errors="coerce")
    biz["review_count"] = pd.to_numeric(biz["review_count"], errors="coerce")
    print(f"  {len(biz):,} dining businesses in {len(cities)} metros")
    return biz


def stream_reviews(keep_ids: set[str]) -> pd.DataFrame:
    """Scan the review file once, keeping only reviews for `keep_ids`."""
    records = []
    t0 = time.time()
    with open(REVIEW_FILE, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i and i % 1_000_000 == 0:
                print(f"    scanned {i:,} reviews … kept {len(records):,}")
            r = json.loads(line)
            if r["business_id"] in keep_ids:
                records.append(
                    (
                        r["user_id"],
                        r["business_id"],
                        float(r["stars"]),
                        r["date"],
                        len(r["text"]),
                        int(r["useful"]),
                    )
                )
    df = pd.DataFrame(
        records,
        columns=["user_id", "business_id", "stars", "date", "review_len", "useful"],
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"  kept {len(df):,} reviews in {time.time() - t0:.0f}s")
    return df


def kcore_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively drop users < MIN_USER_REVIEWS and items < MIN_BUSINESS_REVIEWS."""
    before = len(df)
    while True:
        u = df["user_id"].value_counts()
        df = df[df["user_id"].isin(u[u >= C.MIN_USER_REVIEWS].index)]
        b = df["business_id"].value_counts()
        df = df[df["business_id"].isin(b[b >= C.MIN_BUSINESS_REVIEWS].index)]
        # stable once a full pass changes nothing
        if (df["user_id"].value_counts() >= C.MIN_USER_REVIEWS).all() and (
            df["business_id"].value_counts() >= C.MIN_BUSINESS_REVIEWS
        ).all():
            break
    print(
        f"  k-core: {before:,} -> {len(df):,} reviews "
        f"({df['user_id'].nunique():,} users, {df['business_id'].nunique():,} businesses)"
    )
    return df


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each interaction train/valid/test by global date quantiles."""
    df = df.sort_values("date").reset_index(drop=True)
    t_cut = df["date"].quantile(C.TRAIN_FRAC)
    v_cut = df["date"].quantile(C.TRAIN_FRAC + C.VALID_FRAC)
    split = pd.Series("test", index=df.index)
    split[df["date"] <= t_cut] = "train"
    split[(df["date"] > t_cut) & (df["date"] <= v_cut)] = "valid"
    df["split"] = split.values
    print(
        f"  split @ {t_cut.date()} / {v_cut.date()}  ->  "
        + ", ".join(f"{k}={v:,}" for k, v in df['split'].value_counts().items())
    )
    return df


def run() -> pd.DataFrame:
    print("[prepare] selecting businesses …")
    biz = select_businesses()
    keep_ids = set(biz["business_id"])

    print("[prepare] streaming reviews …")
    df = stream_reviews(keep_ids)

    print("[prepare] k-core filtering …")
    df = kcore_filter(df)

    print("[prepare] temporal split …")
    df = temporal_split(df)

    # keep only businesses that survived k-core
    biz = biz[biz["business_id"].isin(df["business_id"].unique())].reset_index(drop=True)

    df.to_parquet(C.ARTIFACTS / "interactions.parquet", index=False)
    biz.to_parquet(C.ARTIFACTS / "businesses.parquet", index=False)
    print(f"[prepare] wrote interactions.parquet and businesses.parquet to {C.ARTIFACTS}")
    return df


if __name__ == "__main__":
    run()
