"""Ranking metrics for top-K recommendation.

Each metric takes a ranked list of business_ids for one user plus that user's
held-out relevant items, then we average over users. Relevance is graded by star
rating (see config.STAR_TO_GAIN): a 5-star visit is worth more than a 3-star one.
NDCG uses those grades; Recall and MAP treat any positive grade as relevant.
"""

import numpy as np
import pandas as pd


def dcg(gains: list[float]) -> float:
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(((2 ** gains - 1) * discounts).sum())


def ndcg_at_k(ranked_ids: list, relevant_gains: dict, k: int) -> float:
    top = ranked_ids[:k]
    gains = [relevant_gains.get(i, 0.0) for i in top]
    ideal = sorted(relevant_gains.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list, relevant_gains: dict, k: int) -> float:
    rel = {i for i, g in relevant_gains.items() if g > 0}
    if not rel:
        return 0.0
    hits = sum(1 for i in ranked_ids[:k] if i in rel)
    return hits / len(rel)


def average_precision_at_k(ranked_ids: list, relevant_gains: dict, k: int) -> float:
    rel = {i for i, g in relevant_gains.items() if g > 0}
    if not rel:
        return 0.0
    hits, score = 0, 0.0
    for rank, i in enumerate(ranked_ids[:k], start=1):
        if i in rel:
            hits += 1
            score += hits / rank
    return score / min(len(rel), k)


def evaluate_rankings(rankings: dict, relevance: dict, k: int) -> dict:
    """Average NDCG/Recall/MAP @k over the users present in both dicts.

    rankings:  {user_id: [business_id ranked best→worst]}
    relevance: {user_id: {business_id: gain}}
    """
    ndcg, recall, ap, n = [], [], [], 0
    for user, ranked in rankings.items():
        rel = relevance.get(user)
        if not rel:
            continue
        ndcg.append(ndcg_at_k(ranked, rel, k))
        recall.append(recall_at_k(ranked, rel, k))
        ap.append(average_precision_at_k(ranked, rel, k))
        n += 1
    return {
        f"NDCG@{k}": float(np.mean(ndcg)) if ndcg else 0.0,
        f"Recall@{k}": float(np.mean(recall)) if recall else 0.0,
        f"MAP@{k}": float(np.mean(ap)) if ap else 0.0,
        "n_users": n,
    }


def df_to_markdown(df: pd.DataFrame, index_label: str = "method") -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = [index_label] + list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tidy table of {method_name: metric_dict}, with lift over the first row."""
    df = pd.DataFrame(results).T
    metric_cols = [c for c in df.columns if c != "n_users"]
    baseline = df.iloc[0]
    for c in metric_cols:
        df[f"{c} lift%"] = (df[c] / baseline[c] - 1) * 100
    ordered = []
    for c in metric_cols:
        ordered += [c, f"{c} lift%"]
    return df[ordered + ["n_users"]].round(4)
