"""Per-user taste profiles.

A user's history is distilled into a structured summary (favourite cuisines,
price point, cities, go-to spots, how generously they rate) and a *taste vector*
— the centroid of the embeddings of the places they liked. The summary is turned
into a short natural-language profile: by default with a local template, or, if
ANTHROPIC_API_KEY is set, by Claude. The taste vector also powers a content-based
"more places you'd like" recommendation that complements the collaborative model.
"""

import functools
import os
from collections import Counter

import numpy as np
import pandas as pd

from src import config as C
from src.search import index as faiss_index

LLM_MODEL = "claude-haiku-4-5-20251001"
GENERIC_CATS = {"Restaurants", "Food", "Nightlife", "Bars", "Event Planning & Services"}


@functools.lru_cache(maxsize=1)
def _store():
    inter = pd.read_parquet(C.ARTIFACTS / "interactions.parquet")
    biz = pd.read_parquet(C.ARTIFACTS / "businesses.parquet").set_index("business_id")
    corpus = pd.read_parquet(C.ARTIFACTS / "search_corpus.parquet")
    emb = np.load(C.ARTIFACTS / "corpus_emb.npy")

    # business-level embedding = mean of its review embeddings, normalised
    by_biz = corpus.groupby("business_id").indices
    ids, vecs = [], []
    for bid, pos in by_biz.items():
        v = emb[pos].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            ids.append(bid); vecs.append(v / n)
    biz_emb = np.vstack(vecs).astype("float32")
    biz_index = faiss_index.build_index(biz_emb)
    return inter, biz, biz_emb, np.array(ids), biz_index


def _liked(user_id: str) -> pd.DataFrame:
    inter, biz, *_ = _store()
    mine = inter[inter["user_id"] == user_id]
    liked = mine[mine["stars"] >= C.POSITIVE_THRESHOLD]
    return (liked if len(liked) else mine), mine


def profile_stats(user_id: str) -> dict:
    inter, biz, _, _, _ = _store()
    liked, mine = _liked(user_id)
    info = biz.reindex(liked["business_id"].values).dropna(subset=["name"])

    cats = Counter()
    for c in info["categories"].dropna():
        cats.update(t.strip() for t in c.split(",") if t.strip() not in GENERIC_CATS)

    fav = (
        liked.sort_values("stars", ascending=False)
        .drop_duplicates("business_id")
        .head(5)["business_id"].map(biz["name"])
        .dropna().tolist()
    )
    price = info["attributes"].apply(_price).dropna()
    return {
        "user_id": user_id,
        "n_reviews": int(len(mine)),
        "avg_stars": round(float(mine["stars"].mean()), 2),
        "top_categories": [c for c, _ in cats.most_common(5)],
        "price_pref": "$" * int(price.mode().iloc[0]) if len(price) else "n/a",
        "top_cities": info["city"].value_counts().head(3).index.tolist(),
        "favorites": fav,
    }


def _price(attrs):
    if isinstance(attrs, dict):
        try:
            return int(float(str(attrs.get("RestaurantsPriceRange2")).strip("'\" ")))
        except (TypeError, ValueError):
            return None
    return None


def taste_vector(user_id: str):
    _, _, biz_emb, biz_ids, _ = _store()
    liked, _ = _liked(user_id)
    pos = np.where(np.isin(biz_ids, liked["business_id"].unique()))[0]
    if len(pos) == 0:
        return None
    v = biz_emb[pos].mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n).astype("float32") if n > 0 else None


def recommend_by_taste(user_id: str, k: int = 10) -> pd.DataFrame:
    """Content-based picks: businesses whose review-embedding matches the taste vector."""
    inter, biz, _, biz_ids, biz_index = _store()
    v = taste_vector(user_id)
    if v is None:
        return pd.DataFrame()
    seen = set(inter.loc[inter["user_id"] == user_id, "business_id"])
    scores, idx = faiss_index.search(biz_index, v, k + len(seen) + 1)
    rows = []
    for s, j in zip(scores[0], idx[0]):
        bid = biz_ids[j]
        if bid in seen:
            continue
        b = biz.loc[bid]
        rows.append({"name": b["name"], "city": b["city"], "stars": b["stars"],
                     "categories": b["categories"], "match": float(s)})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


# ── narrative rendering ──────────────────────────────────────────────────────────

def render_local(stats: dict) -> str:
    cats = ", ".join(stats["top_categories"][:3]) or "a wide range of places"
    cities = " and ".join(stats["top_cities"][:2]) or "your area"
    favs = "; ".join(stats["favorites"][:3])
    tone = ("a tough grader" if stats["avg_stars"] < 3.5
            else "an enthusiastic regular" if stats["avg_stars"] >= 4.2 else "a balanced reviewer")
    profile = (
        f"You're {tone} who has reviewed {stats['n_reviews']} places, averaging "
        f"{stats['avg_stars']}★. You gravitate toward {cats}, usually around "
        f"{stats['price_pref']} in {cities}."
    )
    if favs:
        profile += f" Your go-to spots include {favs}."
    return profile


def render_llm(stats: dict) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    prompt = (
        "Write a warm, 2-3 sentence dining taste profile (second person, no preamble) "
        f"from these stats:\n{stats}"
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def build_profile(user_id: str, use_llm: str = "auto") -> dict:
    stats = profile_stats(user_id)
    text = render_llm(stats) if use_llm in ("auto", "llm") else None
    source = "claude" if text else "local"
    if text is None:
        text = render_local(stats)
    return {"stats": stats, "profile": text, "source": source}
