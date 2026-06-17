"""Efficient loaders for the Yelp Open Dataset (line-delimited JSON)."""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_jsonl(filename: str, n: int | None = None, show_progress: bool = True) -> pd.DataFrame:
    """Read a line-delimited JSON file into a DataFrame.

    Args:
        filename: File name inside DATA_DIR.
        n: Maximum rows to load (None = all rows).
        show_progress: Show a tqdm progress bar.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Make sure you've extracted the dataset into data/."
        )
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        it = tqdm(fh, desc=filename, unit=" lines") if show_progress else fh
        for i, line in enumerate(it):
            if n is not None and i >= n:
                break
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_business(n: int | None = None) -> pd.DataFrame:
    """Load the business dataset (~150 k rows).

    Key columns: business_id, name, city, state, stars, review_count,
                 is_open, categories, latitude, longitude, attributes, hours.
    """
    df = _load_jsonl("yelp_academic_dataset_business.json", n=n)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    return df


def load_reviews(n: int | None = 200_000) -> pd.DataFrame:
    """Load the review dataset (7 M+ rows — default sample: 200 k).

    Key columns: review_id, user_id, business_id, stars, date,
                 text, useful, funny, cool.
    """
    df = _load_jsonl("yelp_academic_dataset_review.json", n=n)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("useful", "funny", "cool"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_users(n: int | None = 100_000) -> pd.DataFrame:
    """Load the user dataset (2 M+ rows — default sample: 100 k).

    Key columns: user_id, name, review_count, yelping_since,
                 useful, funny, cool, elite, fans, average_stars.
    """
    df = _load_jsonl("yelp_academic_dataset_user.json", n=n)
    df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")
    df["average_stars"] = pd.to_numeric(df["average_stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df["fans"] = pd.to_numeric(df["fans"], errors="coerce")
    df["is_elite"] = df["elite"].astype(str).str.strip().ne("").ne("nan")
    return df


def load_tips(n: int | None = None) -> pd.DataFrame:
    """Load the tip dataset (~1 M rows).

    Key columns: user_id, business_id, text, date, compliment_count.
    """
    df = _load_jsonl("yelp_academic_dataset_tip.json", n=n)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["compliment_count"] = pd.to_numeric(df["compliment_count"], errors="coerce")
    return df


def load_checkins(n: int | None = None) -> pd.DataFrame:
    """Load the check-in dataset (~130 k rows).

    Key columns: business_id, date (comma-separated timestamps).
    """
    return _load_jsonl("yelp_academic_dataset_checkin.json", n=n)
