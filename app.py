"""
Yelp Open Data — Streamlit Dashboard

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import analysis as az
from src.data_loader import load_business, load_reviews, load_users

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Yelp Open Data Explorer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading business data…")
def get_business():
    return load_business(show_progress=False)


@st.cache_data(show_spinner="Loading reviews…")
def get_reviews(n):
    return load_reviews(n=n, show_progress=False)


@st.cache_data(show_spinner="Loading users…")
def get_users(n):
    return load_users(n=n, show_progress=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg", width=120)
    st.title("Yelp Open Data")
    st.markdown("---")
    page = st.radio(
        "Section",
        ["Overview", "Businesses", "Reviews", "Users", "Geography"],
        index=0,
    )
    st.markdown("---")
    n_reviews = st.select_slider(
        "Reviews to load",
        options=[10_000, 50_000, 100_000, 200_000],
        value=50_000,
        format_func=lambda x: f"{x:,}",
    )
    n_users = st.select_slider(
        "Users to load",
        options=[10_000, 30_000, 50_000, 100_000],
        value=30_000,
        format_func=lambda x: f"{x:,}",
    )


# ── Load data ─────────────────────────────────────────────────────────────────

biz = get_business()
rev = get_reviews(n_reviews)
usr = get_users(n_users)


# ── Overview ──────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("⭐ Yelp Open Data Explorer")
    st.markdown("Explore 150k+ businesses, 7M+ reviews, and 2M+ users from the Yelp Academic Dataset.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Businesses", f"{len(biz):,}")
    c2.metric("Reviews (sample)", f"{len(rev):,}")
    c3.metric("Users (sample)", f"{len(usr):,}")
    c4.metric("Cities", f"{biz['city'].nunique():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Business Rating Distribution")
        rd = az.rating_distribution(biz)
        fig = px.bar(
            x=rd.index.astype(str),
            y=rd.values,
            labels={"x": "Stars", "y": "Businesses"},
            color=rd.values,
            color_continuous_scale="RdYlGn",
            text_auto=True,
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Open vs Closed")
        oc = az.open_vs_closed(biz)
        fig = px.pie(
            values=oc.values,
            names=oc.index,
            color=oc.index,
            color_discrete_map={"Open": "#4CAF50", "Closed": "#F44336"},
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reviews Over Time")
    ts = az.reviews_over_time(rev)
    fig = px.area(
        x=ts.index,
        y=ts.values,
        labels={"x": "Date", "y": "Reviews"},
        color_discrete_sequence=["#1976D2"],
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Businesses ────────────────────────────────────────────────────────────────

elif page == "Businesses":
    st.title("Business Analysis")
    tab1, tab2, tab3 = st.tabs(["Categories", "Cities", "Price Range"])

    with tab1:
        n_cats = st.slider("Top N categories", 10, 40, 20)
        cats = az.top_categories(biz, n=n_cats)
        fig = px.bar(
            x=cats.values,
            y=cats.index,
            orientation="h",
            labels={"x": "Businesses", "y": "Category"},
            color=cats.values,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        n_cities = st.slider("Top N cities", 10, 30, 20)
        cities = az.city_summary(biz, n=n_cities)
        fig = px.scatter(
            cities.reset_index(),
            x="city",
            y="avg_stars",
            size="total_reviews",
            color="avg_stars",
            color_continuous_scale="RdYlGn",
            hover_data=["business_count", "total_reviews"],
            labels={"city": "City", "avg_stars": "Avg Stars"},
            title="Top Cities — Avg Rating (bubble = total reviews)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cities.reset_index(), use_container_width=True)

    with tab3:
        price = az.price_range_dist(biz)
        fig = px.bar(
            x=price.index,
            y=price.values,
            labels={"x": "Price Range", "y": "Businesses"},
            color=price.index,
            color_discrete_map={"$": "#81C784", "$$": "#4CAF50", "$$$": "#FF9800", "$$$$": "#F44336"},
            text_auto=True,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Reviews ───────────────────────────────────────────────────────────────────

elif page == "Reviews":
    st.title("Review Analysis")
    tab1, tab2, tab3 = st.tabs(["Volume Trends", "Length & Stars", "Voting Patterns"])

    with tab1:
        freq = st.radio("Frequency", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
        freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        ts = az.reviews_over_time(rev, freq=freq_map[freq])
        fig = px.area(x=ts.index, y=ts.values, labels={"x": "Date", "y": "Reviews"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        length_df = az.review_length_stats(rev)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                length_df.reset_index(),
                x="stars",
                y="avg_chars",
                color="avg_chars",
                color_continuous_scale="RdYlGn",
                labels={"stars": "Stars", "avg_chars": "Avg Characters"},
                title="Avg Review Length by Stars",
                text_auto=".0f",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                length_df.reset_index(),
                x="stars",
                y="n_reviews",
                color="n_reviews",
                color_continuous_scale="Blues",
                labels={"stars": "Stars", "n_reviews": "Reviews"},
                title="Review Count by Stars",
                text_auto=True,
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        voting = az.voting_by_stars(rev)
        fig = px.imshow(
            voting.T,
            text_auto=".3f",
            color_continuous_scale="YlOrRd",
            labels={"x": "Stars", "color": "Avg Votes"},
            title="Avg Useful / Funny / Cool Votes by Star Rating",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Users ─────────────────────────────────────────────────────────────────────

elif page == "Users":
    st.title("User Analysis")
    tab1, tab2, tab3 = st.tabs(["Activity Tiers", "Elite vs Regular", "Cohorts"])

    with tab1:
        activity = az.user_activity_bins(usr)
        fig = px.bar(
            x=activity.index.astype(str),
            y=activity.values,
            labels={"x": "Reviews Written", "y": "Users"},
            color=activity.values,
            color_continuous_scale="Blues",
            text_auto=True,
            title="Users by Review Count Tier",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        elite_df = az.elite_vs_regular(usr)
        for metric, label in [("avg_reviews", "Avg Reviews Written"), ("avg_stars", "Avg Star Rating"), ("avg_fans", "Avg Fans")]:
            fig = px.bar(
                elite_df.reset_index().rename(columns={"index": "user_type"}),
                x="is_elite",
                y=metric,
                color="is_elite",
                color_discrete_map={"Elite": "#1976D2", "Regular": "#FF9800"},
                labels={"is_elite": "User Type", metric: label},
                title=label,
                text_auto=".1f",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cohorts = az.yelping_cohorts(usr)
        fig = px.bar(
            x=cohorts.index.astype(str),
            y=cohorts.values,
            labels={"x": "Year Joined", "y": "Users"},
            color=cohorts.values,
            color_continuous_scale="Viridis",
            text_auto=True,
            title="New Yelpers per Year",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ── Geography ─────────────────────────────────────────────────────────────────

elif page == "Geography":
    st.title("Geographic Analysis")

    state_df = biz.groupby("state").agg(
        count=("business_id", "count"),
        avg_stars=("stars", "mean"),
    ).reset_index().sort_values("count", ascending=False).head(30)

    fig = px.bar(
        state_df,
        x="state",
        y="count",
        color="avg_stars",
        color_continuous_scale="RdYlGn",
        labels={"state": "State", "count": "Businesses", "avg_stars": "Avg Stars"},
        title="Top States by Business Count",
        text_auto=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Scatter Map")
    sample_biz = biz.dropna(subset=["latitude", "longitude"]).sample(
        min(8000, len(biz)), random_state=42
    )
    fig = px.scatter_mapbox(
        sample_biz,
        lat="latitude",
        lon="longitude",
        color="stars",
        hover_name="name",
        hover_data=["city", "state", "review_count"],
        color_continuous_scale="RdYlGn",
        zoom=3,
        height=550,
        mapbox_style="carto-positron",
        title="Business Locations (sample of 8,000)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("For a full heatmap, run `python run_analysis.py` — it saves an interactive HTML map to outputs/business_heatmap.html")
