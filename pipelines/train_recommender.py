"""End-to-end training + evaluation of the two-stage recommender.

Protocol (strict temporal, no leakage):

    ranker training   ALS & features fit on TRAIN, labels from the VALID period
    final evaluation  ALS & features refit on TRAIN+VALID, labels from TEST

We compare three systems on the held-out TEST period:
    popularity   most-reviewed businesses (non-personalised floor)
    als          stage 1 only — candidates ordered by ALS score
    two-stage    stage 1 candidates reranked by LightGBM   ← the system

Outputs (artifacts/ and outputs/reports/):
    ranker.txt, feature_importance.csv, recsys_metrics.csv/.md, test_frame.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.recsys import candidates as cand
from src.recsys import evaluate as ev
from src.recsys import features as F
from src.recsys import ranker as rk

RNG = np.random.default_rng(C.SEED)


def _sample_users(pool, k):
    pool = list(pool)
    if len(pool) <= k:
        return pool
    return list(RNG.choice(pool, size=k, replace=False))


def _feature_store(history, biz, users_static, checkins, tips):
    uf = F.build_user_features(history, biz, users_static)
    bf = F.build_business_features(history, biz, checkins, tips)
    return uf, bf


def main():
    print("[load] interactions, businesses, static dumps …")
    inter = pd.read_parquet(C.ARTIFACTS / "interactions.parquet")
    biz = F.enrich_businesses(pd.read_parquet(C.ARTIFACTS / "businesses.parquet"))
    users_static = F.load_cohort_users(set(inter["user_id"].unique()))
    checkins = F.load_checkins()
    tips = F.load_tip_counts()

    train = inter[inter["split"] == "train"]
    valid = inter[inter["split"] == "valid"]
    test = inter[inter["split"] == "test"]
    trainvalid = inter[inter["split"].isin(["train", "valid"])]
    print(f"       train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    # ── ranker training: ALS+features on TRAIN, labels from VALID ──────────────
    print("[stage1] fitting ALS on TRAIN …")
    als_tr = cand.ALSRecommender().fit(train)
    uf_tr, bf_tr = _feature_store(train, biz, users_static, checkins, tips)

    valid_pos = cand.relevance_dict(valid, als_tr.uid)              # users with future positives
    targets = _sample_users(valid_pos.keys(), C.RANK_TRAIN_USERS)
    cut = int(len(targets) * 0.85)
    print(f"[stage2] building ranking frames for {len(targets):,} users …")
    fit_frame = cand.build_ranking_frame(targets[:cut], als_tr, uf_tr, bf_tr, valid, inject_positives=True)
    es_frame = cand.build_ranking_frame(targets[cut:], als_tr, uf_tr, bf_tr, valid, inject_positives=True)
    print(f"         fit rows={len(fit_frame):,}  early-stop rows={len(es_frame):,}")

    print("[stage2] training LightGBM LambdaMART …")
    model = rk.train_ranker(fit_frame, es_frame)
    model.booster_.save_model(str(C.ARTIFACTS / "ranker.txt"))

    imp = rk.feature_importance(model)
    imp.to_csv(C.ARTIFACTS / "feature_importance.csv", index=False)
    print("         top features:", ", ".join(imp["feature"].head(6)))

    # ── final evaluation: ALS+features on TRAIN+VALID, labels from TEST ─────────
    print("[eval] refitting ALS on TRAIN+VALID for the held-out test …")
    als_te = cand.ALSRecommender().fit(trainvalid)
    uf_te, bf_te = _feature_store(trainvalid, biz, users_static, checkins, tips)

    test_pos = cand.relevance_dict(test, als_te.uid)
    eval_users = _sample_users(test_pos.keys(), C.RANK_EVAL_USERS)
    eval_frame = cand.build_ranking_frame(eval_users, als_te, uf_te, bf_te, test, inject_positives=False)
    eval_frame.to_parquet(C.ARTIFACTS / "test_frame.parquet", index=False)
    relevance = cand.relevance_dict(test, eval_users)
    print(f"       evaluating on {len(relevance):,} users, {len(eval_frame):,} candidate rows")

    results = {}
    for name, method in [("popularity", "popularity"), ("als", "als"), ("two-stage (LightGBM)", "lightgbm")]:
        ranks = rk.rank_candidates(eval_frame, method, model=model)
        results[name] = ev.evaluate_rankings(ranks, relevance, C.RANK_EVAL_K)
        m = results[name]
        print(f"       {name:24s} NDCG@{C.RANK_EVAL_K}={m[f'NDCG@{C.RANK_EVAL_K}']:.4f}  "
              f"Recall@{C.RANK_EVAL_K}={m[f'Recall@{C.RANK_EVAL_K}']:.4f}  "
              f"MAP@{C.RANK_EVAL_K}={m[f'MAP@{C.RANK_EVAL_K}']:.4f}")

    table = ev.comparison_table(results)
    table.to_csv(C.REPORTS / "recsys_metrics.csv")
    (C.REPORTS / "recsys_metrics.md").write_text(
        "# Two-stage recommender — held-out test results\n\n"
        f"Evaluated on {len(relevance):,} users from the held-out TEST period "
        f"(reviews after the train/valid cutoff).\n\n"
        + ev.df_to_markdown(table) + "\n"
    )

    k = C.RANK_EVAL_K
    base = results["als"][f"NDCG@{k}"]
    full = results["two-stage (LightGBM)"][f"NDCG@{k}"]
    print(f"\n[result] two-stage lifts NDCG@{k} by {(full/base-1)*100:.1f}% over ALS-only "
          f"({base:.4f} -> {full:.4f})")
    print(f"[done] wrote metrics to {C.REPORTS/'recsys_metrics.md'}")


if __name__ == "__main__":
    main()
