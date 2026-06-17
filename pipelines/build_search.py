"""Fine-tune the encoder, measure the retrieval lift, and build the FAISS index.

    1. Split corpus businesses into disjoint train / eval sets.
    2. Score same-business retrieval with the off-the-shelf MiniLM encoder.
    3. Fine-tune on same-business pairs drawn only from the train businesses.
    4. Re-score on the held-out eval businesses → the generalisation lift.
    5. Encode the whole corpus with the fine-tuned model and persist the index.

Outputs (artifacts/ + outputs/reports/): finetuned-encoder/, faiss.index,
corpus_emb.npy, search_metrics.md.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# keep the native thread pools from fighting each other (a cause of segfaults
# when torch, OpenBLAS and FAISS share a process)
torch.set_num_threads(1)

from src import config as C
from src.search import corpus as cp
from src.search import embed as E

RNG = np.random.default_rng(C.SEED)
N_EVAL_QUERIES = 1500


def main():
    print(f"[search] device = {E.get_device()}")
    corpus = cp.load_corpus()
    print(f"[search] corpus: {len(corpus):,} reviews / {corpus['business_id'].nunique():,} businesses")

    # disjoint business split (eval businesses must have >=2 reviews for positives)
    counts = corpus["business_id"].value_counts()
    eligible = counts[counts >= 2].index.to_numpy()
    eval_biz = set(RNG.choice(eligible, size=int(0.15 * len(eligible)), replace=False))
    train_biz = [b for b in eligible if b not in eval_biz]

    query_pos = np.where(corpus["business_id"].isin(eval_biz).to_numpy())[0]
    if len(query_pos) > N_EVAL_QUERIES:
        query_pos = RNG.choice(query_pos, N_EVAL_QUERIES, replace=False)
    print(f"[search] {len(train_biz):,} train / {len(eval_biz):,} eval businesses; "
          f"{len(query_pos):,} eval queries")

    # ── baseline ───────────────────────────────────────────────────────────────
    print("[search] encoding corpus with the base MiniLM encoder …", flush=True)
    base = E.load_model()
    emb_base = E.encode(base, corpus["text"])
    base_metrics = E.eval_retrieval(corpus, emb_base, query_pos, k=C.RANK_EVAL_K)
    print(f"[search] base       {base_metrics}", flush=True)
    del base, emb_base
    gc.collect()

    # ── fine-tune ────────────────────────────────────────────────────────────────
    print(f"[search] building {C.FINETUNE_PAIRS:,} same-business pairs and fine-tuning …", flush=True)
    pairs = E.build_pairs(corpus, train_biz, C.FINETUNE_PAIRS)
    ft = E.finetune(pairs)
    print("[search] encoding corpus with the fine-tuned encoder …", flush=True)
    emb_ft = E.encode(ft, corpus["text"])
    ft_metrics = E.eval_retrieval(corpus, emb_ft, query_pos, k=C.RANK_EVAL_K)
    print(f"[search] fine-tuned {ft_metrics}", flush=True)
    del ft                          # release the torch model before touching FAISS
    gc.collect()

    # ── persist the production index (fine-tuned embeddings) ─────────────────────
    np.save(C.ARTIFACTS / "corpus_emb.npy", emb_ft)
    import faiss                    # imported only now, after the encoder is gone
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(emb_ft.shape[1])
    index.add(np.ascontiguousarray(emb_ft, dtype="float32"))
    faiss.write_index(index, str(C.ARTIFACTS / "faiss.index"))
    print(f"[search] saved FAISS index ({emb_ft.shape[0]:,} x {emb_ft.shape[1]} vectors)", flush=True)

    # ── report ───────────────────────────────────────────────────────────────────
    k = C.RANK_EVAL_K
    lift_hit = (ft_metrics[f"Hit@{k}"] / base_metrics[f"Hit@{k}"] - 1) * 100
    lift_mrr = (ft_metrics[f"MRR@{k}"] / base_metrics[f"MRR@{k}"] - 1) * 100
    md = (
        "# Semantic search — encoder fine-tuning\n\n"
        f"Same-business retrieval on **{len(query_pos):,} held-out queries** from "
        f"**{len(eval_biz):,} businesses not seen during fine-tuning**.\n\n"
        f"| encoder | Hit@{k} | MRR@{k} |\n| --- | --- | --- |\n"
        f"| MiniLM (base) | {base_metrics[f'Hit@{k}']:.4f} | {base_metrics[f'MRR@{k}']:.4f} |\n"
        f"| fine-tuned | {ft_metrics[f'Hit@{k}']:.4f} | {ft_metrics[f'MRR@{k}']:.4f} |\n\n"
        f"Fine-tuning lifts Hit@{k} by **{lift_hit:.1f}%** and MRR@{k} by **{lift_mrr:.1f}%** "
        f"on unseen businesses.\n"
    )
    (C.REPORTS / "search_metrics.md").write_text(md)
    print(f"\n[result] fine-tuning lift: Hit@{k} {lift_hit:+.1f}%, MRR@{k} {lift_mrr:+.1f}%")
    print(f"[done] wrote {C.REPORTS / 'search_metrics.md'}")


if __name__ == "__main__":
    main()
