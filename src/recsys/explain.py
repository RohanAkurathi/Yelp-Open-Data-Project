"""Interpret the ranker with SHAP and plot the head-to-head results.

SHAP attributes each prediction back to the features, so we can say *why* the
ranker promotes a candidate rather than just how important a feature is on
average. Run after train_recommender.py — it reads the saved model and the
held-out candidate frame.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.recsys.features import FEATURE_COLUMNS

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def shap_analysis(n_sample: int = 4000) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(C.ARTIFACTS / "ranker.txt"))
    frame = pd.read_parquet(C.ARTIFACTS / "test_frame.parquet")
    X = frame[FEATURE_COLUMNS]
    if len(X) > n_sample:
        X = X.sample(n_sample, random_state=C.SEED)

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)

    # beeswarm — direction and magnitude of each feature's effect
    plt.figure()
    shap.summary_plot(values, X, max_display=15, show=False, plot_size=(9, 7))
    plt.title("What drives the ranker (SHAP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(C.PLOTS / "shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    # mean |SHAP| bar — overall ranking signal strength
    mean_abs = np.abs(values).mean(axis=0)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2a6f97")
    plt.xlabel("mean |SHAP value|")
    plt.title("Top ranking signals", fontsize=12)
    plt.tight_layout()
    plt.savefig(C.PLOTS / "shap_importance.png", bbox_inches="tight")
    plt.close()

    imp.to_csv(C.REPORTS / "shap_importance.csv", index=False)
    print(f"[shap] saved beeswarm + importance plots; top: {', '.join(imp['feature'].head(6))}")
    return imp


def plot_metrics_comparison() -> None:
    df = pd.read_csv(C.REPORTS / "recsys_metrics.csv", index_col=0)
    metrics = [f"NDCG@{C.RANK_EVAL_K}", f"Recall@{C.RANK_EVAL_K}", f"MAP@{C.RANK_EVAL_K}"]
    methods = df.index.tolist()
    colors = ["#bdbdbd", "#74a9cf", "#045a8d"]

    x = np.arange(len(metrics))
    w = 0.25
    plt.figure(figsize=(9, 5.5))
    for i, m in enumerate(methods):
        vals = [df.loc[m, met] for met in metrics]
        bars = plt.bar(x + (i - 1) * w, vals, w, label=m, color=colors[i % len(colors)])
        plt.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    plt.xticks(x, metrics)
    plt.ylabel("score")
    plt.title(f"Held-out ranking quality (n={int(df['n_users'].iloc[0]):,} users)", fontsize=12)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(C.PLOTS / "recsys_metrics.png", bbox_inches="tight")
    plt.close()
    print("[plot] saved recsys_metrics.png")


if __name__ == "__main__":
    plot_metrics_comparison()
    shap_analysis()
