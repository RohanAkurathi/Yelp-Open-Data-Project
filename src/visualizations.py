"""Reusable plotting helpers for the Yelp Open Dataset project."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUTS = Path(__file__).parent.parent / "outputs" / "plots"
OUTPUTS.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Business ──────────────────────────────────────────────────────────────────

def plot_rating_distribution(rating_series: pd.Series, title: str = "Star Rating Distribution") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("RdYlGn", len(rating_series))
    bars = ax.bar(rating_series.index.astype(str), rating_series.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set(title=title, xlabel="Stars", ylabel="Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return _save(fig, "rating_distribution")


def plot_top_categories(cat_series: pd.Series, n: int = 20) -> Path:
    data = cat_series.head(n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Blues_r", len(data))
    bars = ax.barh(data.index, data.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set(title=f"Top {n} Business Categories", xlabel="Number of Businesses")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return _save(fig, "top_categories")


def plot_city_bubble(city_df: pd.DataFrame) -> Path:
    """Bubble chart: cities sized by review volume, colored by avg stars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(
        range(len(city_df)),
        city_df["avg_stars"],
        s=city_df["total_reviews"] / city_df["total_reviews"].max() * 3000 + 50,
        c=city_df["avg_stars"],
        cmap="RdYlGn",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xticks(range(len(city_df)))
    ax.set_xticklabels(city_df.index, rotation=45, ha="right", fontsize=9)
    ax.set(title="Top Cities — Avg Rating (bubble = total reviews)", ylabel="Avg Star Rating")
    ax.set_ylim(3, 5)
    plt.colorbar(sc, ax=ax, label="Avg Stars")
    return _save(fig, "city_bubble")


def plot_open_closed(series: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#4CAF50", "#F44336"]
    wedges, texts, autotexts = ax.pie(
        series.values,
        labels=series.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax.set_title("Open vs Closed Businesses", fontsize=14)
    return _save(fig, "open_closed")


def plot_price_range(price_series: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#81C784", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(price_series.index, price_series.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set(title="Restaurants by Price Range", xlabel="Price Range", ylabel="Count")
    return _save(fig, "price_range")


# ── Reviews ───────────────────────────────────────────────────────────────────

def plot_reviews_over_time(ts: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(ts.index, ts.values, alpha=0.3, color="#1976D2")
    ax.plot(ts.index, ts.values, color="#1976D2", linewidth=1.5)
    ax.set(title="Monthly Review Volume Over Time", xlabel="Date", ylabel="Reviews")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return _save(fig, "reviews_over_time")


def plot_review_length(length_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette("RdYlGn", 5)
    axes[0].bar(length_df.index.astype(str), length_df["avg_chars"], color=colors)
    axes[0].set(title="Avg Review Length by Stars", xlabel="Stars", ylabel="Avg Characters")
    axes[1].bar(length_df.index.astype(str), length_df["n_reviews"], color=colors)
    axes[1].set(title="Review Count by Stars", xlabel="Stars", ylabel="Count")
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return _save(fig, "review_length")


def plot_voting_heatmap(voting_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        voting_df.T,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Avg Votes"},
    )
    ax.set(title="Avg Useful / Funny / Cool Votes by Star Rating", xlabel="Stars")
    return _save(fig, "voting_heatmap")


# ── Users ──────────────────────────────────────────────────────────────────────

def plot_user_activity(activity_series: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Blues", len(activity_series))
    bars = ax.bar(activity_series.index.astype(str), activity_series.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set(title="Users by Review Count Tier", xlabel="Reviews Written", ylabel="Number of Users")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    return _save(fig, "user_activity")


def plot_elite_comparison(elite_df: pd.DataFrame) -> Path:
    metrics = ["avg_reviews", "avg_stars", "avg_fans"]
    labels = ["Avg Reviews", "Avg Stars", "Avg Fans"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        colors = ["#1976D2", "#FF9800"]
        bars = ax.bar(elite_df.index, elite_df[metric], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(title=label, ylabel=label)
    fig.suptitle("Elite vs Regular Users", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "elite_comparison")


def plot_yelping_cohorts(cohort_series: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(cohort_series))
    ax.bar(cohort_series.index.astype(str), cohort_series.values, color=colors, edgecolor="white")
    ax.set(title="New Yelpers per Year", xlabel="Year Joined", ylabel="Users")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=45)
    return _save(fig, "yelping_cohorts")


# ── Word cloud ────────────────────────────────────────────────────────────────

def plot_word_clouds(word_data: dict[int, list[tuple[str, int]]]) -> Path:
    """word_data: {stars: [(word, freq), ...]}"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed — skipping word cloud")
        return None

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    cmap_map = {1: "Reds", 2: "OrRd", 3: "YlOrBr", 4: "YlGn", 5: "Greens"}
    for ax, (stars, words) in zip(axes, sorted(word_data.items())):
        freq = dict(words)
        wc = WordCloud(
            width=400,
            height=300,
            background_color="white",
            colormap=cmap_map.get(stars, "viridis"),
            max_words=60,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{'⭐' * stars}", fontsize=14)
    fig.suptitle("Most Common Words by Star Rating", fontsize=14)
    fig.tight_layout()
    return _save(fig, "word_clouds")


# ── Geographic ────────────────────────────────────────────────────────────────

def make_folium_map(business_df: pd.DataFrame, sample: int = 5000) -> Path:
    """Create an interactive Folium map of businesses."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("folium not installed — skipping map")
        return None

    sample_df = business_df.dropna(subset=["latitude", "longitude"]).sample(
        min(sample, len(business_df)), random_state=42
    )
    center = [sample_df["latitude"].mean(), sample_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
    heat_data = list(zip(sample_df["latitude"], sample_df["longitude"]))
    HeatMap(heat_data, radius=6, blur=10, max_zoom=13).add_to(m)

    out = Path(__file__).parent.parent / "outputs" / "business_heatmap.html"
    m.save(str(out))
    return out
