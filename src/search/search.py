"""Query-time semantic search over the review index.

Loads the fine-tuned encoder and the FAISS index once, then answers free-text
queries either as matching reviews or, more usefully, as matching *businesses*
(reviews are pooled back to the place they describe). A plain-English query like
"cozy spot with great oat-milk lattes" is embedded and matched against review
meaning, not keywords.
"""

import functools

import numpy as np
import pandas as pd

from src import config as C
from src.search import corpus as cp
from src.search import embed as E
from src.search import index as faiss_index


class SemanticSearch:
    def __init__(self):
        self.model = E.load_model(E.FINETUNED_DIR)
        self.index = faiss_index.load_index()
        self.corpus = cp.load_corpus()
        biz = pd.read_parquet(C.ARTIFACTS / "businesses.parquet")
        # rename to avoid colliding with the corpus's review-level "stars" on join
        self.biz = (
            biz.set_index("business_id")[["name", "city", "state", "stars", "categories"]]
            .rename(columns={"stars": "biz_stars"})
        )

    def _embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True).astype("float32")

    def _decorate(self, positions, scores) -> pd.DataFrame:
        rows = self.corpus.iloc[positions].copy()
        rows["score"] = scores
        rows = rows.join(self.biz, on="business_id")
        return rows

    def search_reviews(self, query: str, k: int = 10) -> pd.DataFrame:
        """Top-k individual reviews most semantically similar to the query."""
        scores, idx = faiss_index.search(self.index, self._embed_query(query), k)
        out = self._decorate(idx[0], scores[0]).rename(columns={"biz_stars": "stars"})
        return out[["name", "city", "stars", "score", "text"]].reset_index(drop=True)

    def search_businesses(self, query: str, k: int = 10, pool: int = 200) -> pd.DataFrame:
        """Top-k businesses, pooling review-level hits up to the business."""
        scores, idx = faiss_index.search(self.index, self._embed_query(query), pool)
        hits = self._decorate(idx[0], scores[0])
        agg = (
            hits.groupby("business_id")
            .agg(name=("name", "first"), city=("city", "first"),
                 stars=("biz_stars", "first"), categories=("categories", "first"),
                 match=("score", "max"), n_hits=("score", "size"),
                 snippet=("text", "first"))
            .sort_values("match", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )
        agg["snippet"] = agg["snippet"].str.slice(0, 160) + "…"
        return agg


@functools.lru_cache(maxsize=1)
def get_engine() -> SemanticSearch:
    """Cached singleton so the model/index load only once per process."""
    return SemanticSearch()
