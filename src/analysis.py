"""Core analysis functions for the Yelp Open Dataset."""

import re
from collections import Counter

import numpy as np
import pandas as pd


# ── Business analysis ────────────────────────────────────────────────────────

def top_categories(business_df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Return the n most common business categories."""
    cats = (
        business_df["categories"]
        .dropna()
        .str.split(", ")
        .explode()
        .str.strip()
    )
    return cats.value_counts().head(n)


def rating_distribution(df: pd.DataFrame, col: str = "stars") -> pd.Series:
    """Frequency count of star ratings (rounded to nearest 0.5)."""
    return df[col].dropna().round(1).value_counts().sort_index()


def city_summary(business_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top n cities by business count with avg rating."""
    return (
        business_df.groupby("city")
        .agg(
            business_count=("business_id", "count"),
            avg_stars=("stars", "mean"),
            total_reviews=("review_count", "sum"),
        )
        .sort_values("business_count", ascending=False)
        .head(n)
        .round(2)
    )


def open_vs_closed(business_df: pd.DataFrame) -> pd.Series:
    return business_df["is_open"].map({1: "Open", 0: "Closed"}).value_counts()


def price_range_dist(business_df: pd.DataFrame) -> pd.Series:
    """Distribution of RestaurantsPriceRange2 attribute."""
    prices = business_df["attributes"].dropna().apply(
        lambda a: a.get("RestaurantsPriceRange2") if isinstance(a, dict) else None
    )
    label = {
        "1": "$",
        "2": "$$",
        "3": "$$$",
        "4": "$$$$",
        "1.0": "$",
        "2.0": "$$",
        "3.0": "$$$",
        "4.0": "$$$$",
    }
    return (
        prices.dropna()
        .astype(str)
        .str.strip("'\" ")
        .map(label)
        .dropna()
        .value_counts()
        .reindex(["$", "$$", "$$$", "$$$$"])
        .fillna(0)
        .astype(int)
    )


# ── Review analysis ───────────────────────────────────────────────────────────

def reviews_over_time(review_df: pd.DataFrame, freq: str = "ME") -> pd.Series:
    """Monthly review counts."""
    return review_df.set_index("date").resample(freq).size()


def review_length_stats(review_df: pd.DataFrame) -> pd.DataFrame:
    """Average review text length (chars) grouped by star rating."""
    tmp = review_df.copy()
    tmp["length"] = tmp["text"].str.len()
    return (
        tmp.groupby("stars")["length"]
        .agg(["mean", "median", "count"])
        .round(1)
        .rename(columns={"mean": "avg_chars", "median": "median_chars", "count": "n_reviews"})
    )


def voting_by_stars(review_df: pd.DataFrame) -> pd.DataFrame:
    """Mean useful/funny/cool votes per star rating."""
    return (
        review_df.groupby("stars")[["useful", "funny", "cool"]]
        .mean()
        .round(3)
    )


def top_reviewed_businesses(review_df: pd.DataFrame, business_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    counts = review_df["business_id"].value_counts().head(n).reset_index()
    counts.columns = ["business_id", "review_count"]
    return counts.merge(business_df[["business_id", "name", "city", "stars"]], on="business_id", how="left")


# ── User analysis ─────────────────────────────────────────────────────────────

def user_activity_bins(user_df: pd.DataFrame) -> pd.Series:
    """Bin users by review count into activity tiers."""
    bins = [0, 1, 10, 50, 100, 500, np.inf]
    labels = ["1", "2–10", "11–50", "51–100", "101–500", "500+"]
    return pd.cut(user_df["review_count"], bins=bins, labels=labels).value_counts().sort_index()


def elite_vs_regular(user_df: pd.DataFrame) -> pd.DataFrame:
    return (
        user_df.groupby("is_elite")
        .agg(
            count=("user_id", "count"),
            avg_reviews=("review_count", "mean"),
            avg_stars=("average_stars", "mean"),
            avg_fans=("fans", "mean"),
        )
        .round(2)
        .rename(index={True: "Elite", False: "Regular"})
    )


def yelping_cohorts(user_df: pd.DataFrame) -> pd.Series:
    """Users per year they joined Yelp."""
    return user_df["yelping_since"].dt.year.value_counts().sort_index()


# ── Sentiment / text analysis ─────────────────────────────────────────────────

_CLEAN = re.compile(r"[^a-z\s]")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "was", "i", "we", "they", "this", "that",
    "be", "are", "have", "has", "had", "my", "our", "not", "so", "as",
    "very", "just", "get", "got", "do", "did", "one", "you", "me", "he",
    "she", "from", "all", "there", "been", "will", "would", "what",
    "if", "up", "out", "no", "by", "also", "when", "than", "more", "like",
    "their", "them", "which", "s", "t", "re", "m",
}


def word_freq_by_stars(review_df: pd.DataFrame, stars: int, n: int = 50) -> list[tuple[str, int]]:
    """Top n words in reviews with a given star rating."""
    texts = review_df.loc[review_df["stars"] == stars, "text"].dropna()
    words = []
    for t in texts:
        words.extend(
            w for w in _CLEAN.sub("", t.lower()).split()
            if w not in _STOPWORDS and len(w) > 2
        )
    return Counter(words).most_common(n)


def polarity_from_stars(stars: float) -> str:
    """Coarse sentiment label from star rating."""
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"
