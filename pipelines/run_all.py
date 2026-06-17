"""Run the whole project end to end.

    python pipelines/run_all.py

Steps (each is also runnable on its own):
    1. data prep            src/recsys/prepare.py
    2. train recommender    pipelines/train_recommender.py
    3. interpret + plots    src/recsys/explain.py
    4. build search corpus  src/search/corpus.py
    5. fine-tune + index    pipelines/build_search.py

Expects the raw Yelp JSON under data/. Set OPENBLAS_NUM_THREADS=1 for ALS.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("data prep", [sys.executable, "-m", "src.recsys.prepare"]),
    ("train recommender", [sys.executable, "pipelines/train_recommender.py"]),
    ("interpret + plots", [sys.executable, "-m", "src.recsys.explain"]),
    ("build search corpus", [sys.executable, "-m", "src.search.corpus"]),
    ("fine-tune + index", [sys.executable, "pipelines/build_search.py"]),
]


def main():
    for name, cmd in STEPS:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\nAll steps complete. Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
