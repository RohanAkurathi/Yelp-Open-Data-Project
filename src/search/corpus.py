"""Build the review text corpus that the semantic index is built from.

The interactions table dropped review text to stay small, so we stream the
review file once more and keep a sample of reviews for our cohort businesses
(the dining spots in the target metros). Reviews are tied to real businesses so
search results map back to places a user could actually visit.
"""

import json

import pandas as pd

from src import config as C

REVIEW_FILE = C.DATA_DIR / "yelp_academic_dataset_review.json"
CORPUS_PATH = C.ARTIFACTS / "search_corpus.parquet"

# only index reviews with enough substance to embed meaningfully
MIN_CHARS = 40
MAX_CHARS = 1500   # truncate very long reviews so encoding stays fast
PER_BIZ_CAP = 40   # cap per business so the corpus spans many places, not a few


def build_corpus(n: int = C.EMBED_SAMPLE_REVIEWS) -> pd.DataFrame:
    biz = pd.read_parquet(C.ARTIFACTS / "businesses.parquet")
    keep = set(biz["business_id"])

    # collect cohort reviews with a per-business cap, then sample for spread
    rows, seen = [], {}
    with open(REVIEW_FILE, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            bid = r["business_id"]
            if bid not in keep or seen.get(bid, 0) >= PER_BIZ_CAP:
                continue
            text = r["text"].strip().replace("\n", " ")
            if len(text) < MIN_CHARS:
                continue
            seen[bid] = seen.get(bid, 0) + 1
            rows.append((r["review_id"], bid, float(r["stars"]), text[:MAX_CHARS]))

    df = pd.DataFrame(rows, columns=["review_id", "business_id", "stars", "text"])
    if len(df) > n:
        df = df.sample(n, random_state=C.SEED).reset_index(drop=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"[corpus] {len(df):,} reviews across {df['business_id'].nunique():,} businesses "
          f"-> {CORPUS_PATH}")
    return df


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(CORPUS_PATH)


if __name__ == "__main__":
    build_corpus()
