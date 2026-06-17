"""
Run all Yelp Open Data analyses and save plots to outputs/plots/.

Usage:
    python run_analysis.py            # full run (loads defaults)
    python run_analysis.py --quick    # smaller samples for speed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import analysis as az
from src import visualizations as vz
from src.data_loader import load_business, load_reviews, load_tips, load_users


def main(quick: bool = False) -> None:
    n_reviews = 50_000 if quick else 200_000
    n_users = 30_000 if quick else 100_000

    print("=" * 60)
    print("Yelp Open Data — Analysis Pipeline")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading business data…")
    biz = load_business()
    print(f"      {len(biz):,} businesses loaded")

    print(f"\n[2/4] Loading {n_reviews:,} reviews…")
    rev = load_reviews(n=n_reviews)
    print(f"      {len(rev):,} reviews loaded")

    print(f"\n[3/4] Loading {n_users:,} users…")
    usr = load_users(n=n_users)
    print(f"      {len(usr):,} users loaded")

    print("\n[4/4] Loading tips…")
    tips = load_tips()
    print(f"      {len(tips):,} tips loaded")

    # ── Business analysis ──────────────────────────────────────
    print("\n── Business Analysis ──")
    cats = az.top_categories(biz)
    p = vz.plot_top_categories(cats)
    print(f"  [✓] Top categories → {p}")

    rating_dist = az.rating_distribution(biz)
    p = vz.plot_rating_distribution(rating_dist, "Business Star Rating Distribution")
    print(f"  [✓] Rating distribution → {p}")

    cities = az.city_summary(biz)
    p = vz.plot_city_bubble(cities)
    print(f"  [✓] City bubble chart → {p}")

    oc = az.open_vs_closed(biz)
    p = vz.plot_open_closed(oc)
    print(f"  [✓] Open vs closed → {p}")

    price = az.price_range_dist(biz)
    p = vz.plot_price_range(price)
    print(f"  [✓] Price range → {p}")

    # ── Review analysis ────────────────────────────────────────
    print("\n── Review Analysis ──")
    ts = az.reviews_over_time(rev)
    p = vz.plot_reviews_over_time(ts)
    print(f"  [✓] Reviews over time → {p}")

    length_df = az.review_length_stats(rev)
    p = vz.plot_review_length(length_df)
    print(f"  [✓] Review length by stars → {p}")

    voting = az.voting_by_stars(rev)
    p = vz.plot_voting_heatmap(voting)
    print(f"  [✓] Voting heatmap → {p}")

    # ── User analysis ──────────────────────────────────────────
    print("\n── User Analysis ──")
    activity = az.user_activity_bins(usr)
    p = vz.plot_user_activity(activity)
    print(f"  [✓] User activity tiers → {p}")

    elite_df = az.elite_vs_regular(usr)
    p = vz.plot_elite_comparison(elite_df)
    print(f"  [✓] Elite vs regular → {p}")

    cohorts = az.yelping_cohorts(usr)
    p = vz.plot_yelping_cohorts(cohorts)
    print(f"  [✓] Yelping cohorts → {p}")

    # ── Word cloud (requires wordcloud package) ────────────────
    print("\n── Sentiment / Text Analysis ──")
    star_words = {s: az.word_freq_by_stars(rev, s, n=80) for s in [1, 2, 3, 4, 5]}
    p = vz.plot_word_clouds(star_words)
    if p:
        print(f"  [✓] Word clouds → {p}")

    # ── Geographic map ─────────────────────────────────────────
    print("\n── Geographic Analysis ──")
    map_path = vz.make_folium_map(biz)
    if map_path:
        print(f"  [✓] Interactive heatmap → {map_path}")

    print("\n" + "=" * 60)
    print("All plots saved to outputs/plots/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Yelp data analysis pipeline")
    parser.add_argument("--quick", action="store_true", help="Use smaller samples for speed")
    args = parser.parse_args()
    main(quick=args.quick)
