
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from mplsoccer.pitch import VerticalPitch

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | StatsBomb Style",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- File Paths ---
DATA_DIR = "data"

# --- Metric Mapping ---
metric_files = {
    "Expected Goals (xG)": "xG",
    "Expected Assists (xA)": "assists",
    "Expected Threat (xT)": "xT",
    "Expected Disruption (xDisruption)": "xDisruption",
    "Goal Probability Added (GPA)": "gpa",
    "Minutes Played": "minutes",
    "Corners": "corners"
}
competitions = ["WSL"]  # extend with more leagues later

# --- Helpers ---
def inject_statsbomb_css():
    st.markdown("""<style>body {background:#0C1A2A;color:#fff;}</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_competition_data(league: str, metric: str):
    file_path = os.path.join(DATA_DIR, f"{league}_{metric}.csv")
    return load_csv(file_path)

def create_statsbomb_shot_map(df, title="SHOT MAP"):
    if df.empty:
        return None
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color="#152642", line_color="#2D4A76")
    fig, ax = pitch.draw(figsize=(10, 7))
    for _, row in df.iterrows():
        ax.scatter(row["y"], row["x"], c="#FF6B6B" if not row.get("isGoal", False) else "#00FF88",
                   s=max(row.get("xG", 0) * 800, 50), alpha=0.7, edgecolors="white")
    ax.set_title(title, color="#00FF88", fontsize=16)
    return fig

# --- Page Components ---
def create_navigation_header():
    st.markdown("""
    <div style="background:#152642;padding:1rem;border-bottom:2px solid #00FF88;">
        <h2 style="margin:0;color:#00FF88;font-family:monospace;">WOSO ANALYTICS</h2>
    </div>
    """, unsafe_allow_html=True)

def display_landing_page():
    st.markdown("<h1 style='text-align:center;color:#00FF88;'>WOSO ANALYTICS</h1>", unsafe_allow_html=True)
    if st.button("ENTER ANALYTICS SUITE", use_container_width=True, type="primary"):
        st.session_state.app_mode = "MainApp"
        st.rerun()

def display_data_scouting_page():
    st.subheader("Player Performance Analysis")

    # Selectors
    col1, col2 = st.columns(2)
    with col1:
        league = st.selectbox("Competition", competitions)
    with col2:
        metric_display = st.selectbox("Metric", list(metric_files.keys()))
        metric = metric_files[metric_display]

    df = load_competition_data(league, metric)
    if df.empty:
        st.warning("No data available for this selection.")
        return

    st.dataframe(df.head(20), use_container_width=True)

def display_match_analysis_page():
    st.subheader("Match Analysis Centre")
    matches = load_csv(os.path.join(DATA_DIR, "WSL.csv"))
    if matches.empty:
        st.warning("No match data found.")
        return

    match = st.selectbox("Select Match", matches["match"].unique())
    events = load_csv(os.path.join(DATA_DIR, "WSL_events.csv"))  # if exists
    if not events.empty:
        shots = events[events["type"] == "Shot"]
        fig = create_statsbomb_shot_map(shots, f"{match} Shot Map")
        if fig:
            st.pyplot(fig)

def display_player_profiling_page():
    st.subheader("Player Profiles")
    # Merge different metric CSVs for richer profiles
    data = {}
    for metric in metric_files.values():
        df = load_competition_data("WSL", metric)
        if not df.empty:
            data[metric] = df.set_index("player")

    if not data:
        st.warning("No player data available.")
        return

    combined = pd.concat(data.values(), axis=1, join="outer").reset_index().fillna(0)
    player = st.selectbox("Select Player", combined["player"].unique())
    row = combined[combined["player"] == player].iloc[0]

    st.metric("Minutes", row.get("minutes", 0))
    st.metric("xG", row.get("xG", 0))
    st.metric("Assists", row.get("assists", 0))
    st.metric("GPA", row.get("gpa", 0))

def display_corners_page():
    st.subheader("Set Piece Analysis")
    df = load_competition_data("WSL", "corners")
    if df.empty:
        st.warning("No corner data available.")
        return

    st.write("Corner Kicks Overview")
    st.dataframe(df.head(20), use_container_width=True)

# --- Main ---
def main():
    inject_statsbomb_css()

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Landing"

    if st.session_state.app_mode == "Landing":
        display_landing_page()
    else:
        create_navigation_header()
        tabs = st.tabs(["📊 Performance", "🎯 Matches", "👤 Profiles", "⛳ Set Pieces"])
        with tabs[0]:
            display_data_scouting_page()
        with tabs[1]:
            display_match_analysis_page()
        with tabs[2]:
            display_player_profiling_page()
        with tabs[3]:
            display_corners_page()

        st.markdown("---")
        st.markdown("<p style='text-align:center;color:#2D4A76;'>© 2024 Women's Football Analytics</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

