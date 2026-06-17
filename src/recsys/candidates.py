"""Stage 1 — ALS candidate generation, and assembly of the ranking frame.

The ALS model factorises the user×business implicit-feedback matrix (confidence
scaled by how much each user liked the place) and, for any user, proposes the
N businesses they're most likely to engage with next. Those candidates — plus,
at training time, the user's known positives — become the rows the LightGBM
ranker scores. This module also attaches the user/business feature tables and
computes the user↔candidate cross-features.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src import config as C
from src.recsys import features as F


class ALSRecommender:
    """Thin wrapper around implicit's ALS with string-id bookkeeping."""

    def __init__(self, factors=C.ALS_FACTORS, regularization=C.ALS_REGULARIZATION,
                 iterations=C.ALS_ITERATIONS, alpha=C.ALS_ALPHA, seed=C.SEED):
        self.alpha = alpha
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization,
            iterations=iterations, random_state=seed,
        )

    def fit(self, history: pd.DataFrame) -> "ALSRecommender":
        self.users = history["user_id"].unique()
        self.items = history["business_id"].unique()
        self.uid = {u: i for i, u in enumerate(self.users)}
        self.bid = {b: i for i, b in enumerate(self.items)}

        rows = history["user_id"].map(self.uid).to_numpy()
        cols = history["business_id"].map(self.bid).to_numpy()
        gain = history["stars"].map(C.STAR_TO_GAIN).fillna(0).to_numpy()
        conf = (1.0 + self.alpha * gain).astype(np.float32)
        self.user_items = sp.csr_matrix(
            (conf, (rows, cols)), shape=(len(self.users), len(self.items))
        )
        self.model.fit(self.user_items, show_progress=False)
        return self

    def recommend(self, user_ids, n=C.N_CANDIDATES):
        """Return {user_id: [(business_id, als_score), …]} for known users."""
        known = [u for u in user_ids if u in self.uid]
        idx = np.array([self.uid[u] for u in known])
        ids, scores = self.model.recommend(
            idx, self.user_items[idx], N=n, filter_already_liked_items=True,
        )
        out = {}
        for u, row_ids, row_scores in zip(known, ids, scores):
            out[u] = list(zip((self.items[j] for j in row_ids), row_scores))
        return out

    def score_pairs(self, user_ids, business_ids) -> np.ndarray:
        """ALS score (factor dot-product) for aligned user/business id arrays."""
        uf = self.model.user_factors
        itf = self.model.item_factors
        uf = uf.to_numpy() if hasattr(uf, "to_numpy") else np.asarray(uf)
        itf = itf.to_numpy() if hasattr(itf, "to_numpy") else np.asarray(itf)
        ui = np.array([self.uid[u] for u in user_ids])
        bi = np.array([self.bid[b] for b in business_ids])
        return np.einsum("ij,ij->i", uf[ui], itf[bi]).astype(np.float32)


def label_gains(label_df: pd.DataFrame) -> pd.DataFrame:
    """Best graded relevance per (user, business) in a label window."""
    g = label_df.assign(gain=label_df["stars"].map(C.STAR_TO_GAIN).fillna(0))
    return g.groupby(["user_id", "business_id"], as_index=False)["gain"].max()


def relevance_dict(label_df: pd.DataFrame, users) -> dict:
    """{user_id: {business_id: gain>0}} for evaluation."""
    g = label_gains(label_df)
    g = g[(g["gain"] > 0) & (g["user_id"].isin(set(users)))]
    return {u: dict(zip(sub["business_id"], sub["gain"]))
            for u, sub in g.groupby("user_id")}


def build_ranking_frame(target_users, als, user_feats, biz_feats, label_df,
                        n_candidates=C.N_CANDIDATES, inject_positives=False):
    """Assemble the per-(user, candidate) frame with features and labels."""
    recs = als.recommend(target_users, n=n_candidates)

    users, biz, als_score, cf_rank = [], [], [], []
    for u, cand in recs.items():
        for rank, (b, s) in enumerate(cand):
            users.append(u); biz.append(b); als_score.append(s); cf_rank.append(rank)
    frame = pd.DataFrame({
        "user_id": users, "business_id": biz,
        "als_score": als_score, "cf_rank": cf_rank,
    })

    if inject_positives:
        pos = label_gains(label_df)
        pos = pos[(pos["gain"] > 0) & pos["user_id"].isin(als.uid) & pos["business_id"].isin(als.bid)]
        pos = pos[pos["user_id"].isin(set(target_users))][["user_id", "business_id"]]
        if len(pos):
            pos = pos.copy()
            pos["als_score"] = als.score_pairs(pos["user_id"].tolist(), pos["business_id"].tolist())
            pos["cf_rank"] = n_candidates  # sentinel: not in the top-N list
            frame = pd.concat([frame, pos], ignore_index=True)
            frame = frame.drop_duplicates(["user_id", "business_id"], keep="first")

    # labels
    gains = label_gains(label_df).rename(columns={"gain": "label"})
    frame = frame.merge(gains, on=["user_id", "business_id"], how="left")
    frame["label"] = frame["label"].fillna(0).astype(int)

    # feature joins
    frame = frame.join(user_feats, on="user_id").join(biz_feats, on="business_id")

    # cross-features
    frame["star_gap"] = frame["u_avg_stars"] - frame["b_avg_stars"]
    price = frame["b_price_range"].where(frame["b_price_range"] > 0, np.nan)
    frame["price_gap"] = (frame["u_avg_price"] - price).abs().fillna(0.0)
    frame["same_city"] = (frame["b_city"] == frame["u_modal_city"]).astype(int)
    frame["same_top_cat"] = (frame["b_primary_category"] == frame["u_top_cat"]).astype(int)

    frame = frame.dropna(subset=["u_n_reviews", "b_n_reviews"])  # drop unseen ids
    return frame.sort_values("user_id").reset_index(drop=True)
