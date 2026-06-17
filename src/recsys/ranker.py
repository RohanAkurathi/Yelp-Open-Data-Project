"""Stage 2 — LightGBM LambdaMART reranker.

Takes the candidate frame from stage 1 and learns to reorder each user's
candidates so the ones they'll actually like land at the top. Training optimises
LambdaMART's NDCG surrogate with the user as the query group. We also expose the
simpler scoring rules (popularity, ALS score) so the ranker can be compared
against the baselines it's meant to beat.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config as C
from src.recsys.features import FEATURE_COLUMNS


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    """Per-user row counts, in the frame's (user-sorted) order."""
    return frame.groupby("user_id", sort=False).size().tolist()


def train_ranker(train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[C.RANK_EVAL_K],
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=C.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train_frame[FEATURE_COLUMNS], train_frame["label"],
        group=_group_sizes(train_frame),
        eval_set=[(valid_frame[FEATURE_COLUMNS], valid_frame["label"])],
        eval_group=[_group_sizes(valid_frame)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def rank_candidates(frame: pd.DataFrame, method: str, model=None) -> dict:
    """Return {user_id: [business_id ranked best→worst]} for a scoring method."""
    f = frame.copy()
    if method == "popularity":
        f["score"] = f["b_global_review_count"]
    elif method == "als":
        f["score"] = f["als_score"]
    elif method == "lightgbm":
        f["score"] = model.predict(f[FEATURE_COLUMNS])
    else:
        raise ValueError(method)

    f = f.sort_values(["user_id", "score"], ascending=[True, False])
    return {u: sub["business_id"].tolist() for u, sub in f.groupby("user_id", sort=False)}


def feature_importance(model: lgb.LGBMRanker) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    })
    imp["gain_pct"] = imp["gain"] / imp["gain"].sum() * 100
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)
