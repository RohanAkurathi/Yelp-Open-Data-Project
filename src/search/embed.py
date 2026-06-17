"""Embedding model: encoding, in-domain fine-tuning, and retrieval evaluation.

We start from a general MiniLM sentence encoder and adapt it to Yelp reviews with
a contrastive objective: two reviews of the same business should land close
together, everything else should be pushed apart (MultipleNegativesRankingLoss
with in-batch negatives). Fine-tuning pairs are drawn from one set of businesses
and retrieval is scored on a *disjoint* set, so the lift reflects generalisation,
not memorisation.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src import config as C

torch.set_num_threads(1)   # play nicely alongside FAISS/OpenBLAS in one process

FINETUNED_DIR = C.ARTIFACTS / "finetuned-encoder"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path: str = C.BASE_EMBED_MODEL) -> SentenceTransformer:
    model = SentenceTransformer(str(path), device=get_device())
    model.max_seq_length = C.MAX_SEQ_LEN
    return model


def encode(model: SentenceTransformer, texts, batch: int = C.EMBED_BATCH) -> np.ndarray:
    return model.encode(
        list(texts), batch_size=batch, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")


def build_pairs(corpus: pd.DataFrame, business_ids, n_pairs: int) -> list[InputExample]:
    """Sample (anchor, positive) pairs of reviews that share a business."""
    rng = np.random.default_rng(C.SEED)
    sub = corpus[corpus["business_id"].isin(set(business_ids))]
    groups = [g["text"].tolist() for _, g in sub.groupby("business_id") if len(g) >= 2]
    examples = []
    while len(examples) < n_pairs and groups:
        texts = groups[rng.integers(len(groups))]
        i, j = rng.choice(len(texts), 2, replace=False)
        examples.append(InputExample(texts=[texts[i], texts[j]]))
    return examples


def finetune(examples: list[InputExample], out_dir=FINETUNED_DIR,
             base: str = C.BASE_EMBED_MODEL, epochs: int = C.FINETUNE_EPOCHS,
             batch: int = C.FINETUNE_BATCH) -> SentenceTransformer:
    model = SentenceTransformer(str(base), device=get_device())
    model.max_seq_length = C.MAX_SEQ_LEN
    loader = DataLoader(examples, shuffle=True, batch_size=batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader)),
        show_progress_bar=True,
    )
    model.save(str(out_dir))   # persist so the search engine can embed queries later
    return model


def eval_retrieval(corpus: pd.DataFrame, embeddings: np.ndarray,
                   query_pos: np.ndarray, k: int = 10, chunk: int = 256) -> dict:
    """Same-business retrieval quality for a set of query reviews.

    For each query we search the whole corpus, drop the query itself, and check
    whether other reviews of the same business surface in the top-k.
    Hit@k = fraction with at least one same-business hit; MRR = mean reciprocal
    rank of the first such hit. Pure NumPy (chunked dot-products) so we don't run
    FAISS in the same process as the torch encoder.
    """
    emb = np.ascontiguousarray(embeddings, dtype="float32")
    biz = corpus["business_id"].to_numpy()
    hits, rr = [], []
    for start in range(0, len(query_pos), chunk):
        qpos = query_pos[start:start + chunk]
        sims = emb[qpos] @ emb.T                      # (chunk, N) cosine sims
        cand = np.argpartition(-sims, k + 1, axis=1)[:, :k + 1]
        for r, q in enumerate(qpos):
            order = cand[r][np.argsort(-sims[r, cand[r]])]
            retrieved = [j for j in order if j != q][:k]
            same = [biz[j] == biz[q] for j in retrieved]
            hits.append(1.0 if any(same) else 0.0)
            rr.append(1.0 / (same.index(True) + 1) if any(same) else 0.0)
    return {f"Hit@{k}": float(np.mean(hits)), f"MRR@{k}": float(np.mean(rr)), "n_queries": len(query_pos)}
