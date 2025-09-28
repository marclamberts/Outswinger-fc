import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from mplsoccer.pitch import VerticalPitch

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | Modern Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- File Paths ---
DATA_DIR = "data"

# --- Metric Mapping ---
metric_files = {
    "Expected Goals (xG)": "xG",           # From WSL.csv
    "Expected Assists (xA)": "assists",
    "Expected Threat (xT)": "xT",
    "Expected Disruption (xDisruption)": "xDisruption",
    "Goal Probability Added (GPA)": "gpa",
    "Minutes Played": "minutes",
    "Corners": "corners"
}
competitions = ["WSL"]

# --- Modern CSS ---
def inject_modern_css():
    st.markdown("""
    <style>
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; background: #0B111D; color: #E0E0E0; }
    h1,h2,h3,h4,h5 { font-family: 'Roboto Mono', monospace; color: #00FFD5; }

    [data-testid="stSidebar"] { background: #101826; border-right: 1px solid #1C2A48; }
    [data-testid="stSidebar"] h1,h2,h3 { color: #00FFD5; }

    .stButton>button { background: linear-gradient(90deg,#00FFD5,#00A6FF); color:#0B111D; font-weight:600; border:none; border-radius:6px; }
    .stButton>button:hover { opacity:0.85; transform: translateY(-1px); }

    .stDataFrame { border:1px solid #1C2A48; border-radius:6px; }
    .stDataFrame th { background:#101826; color:#00FFD5; font-weight:600; }
    .stDataFrame td { background:#0B111D; color:#E0E0E0; }

    [data-testid="metric-container"] { background:#101826; border-radius:8px; padding:1rem; color:#E0E0E0; }
    [data-testid="metric-container"] div { color:#00FFD5; font-weight:700; font-size:1.4rem; }

    .stTabs [data-baseweb="tab-list"] { gap:3px; }
    .stTabs [data-baseweb="tab"] { background:#101826; border-radius:6px 6px 0 0; color:#A0A0A0; font-weight:500; }
    .stTabs [aria-selected="true"] { background:#00FFD5 !important; color:#0B111D !important; }

    .stSelectbox div[data-baseweb="select"]>div { background:#101826; border:1px solid #1C2A48; color:#E0E0E0; border-radius:6px; }
    input, select { background:#101826 !important; color:#E0E0E0 !important; border:1px solid #1C2A48; border-radius:6px; padding:0.4rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loaders ---
@st.cache_data(ttl=3600)
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_competition_data(league: str, metric: str):
    if metric == "xG":  # xG comes from master league CSV
        return load_csv(os.path.join(DATA_DIR, f"{league}.csv"))
    else:
        return load_csv(os.path.join(DATA_DIR, f"{league}_{metric}.csv"))

# --- Shot Map ---
def create_modern_shot_map(df, title="SHOT MAP"):
    if df.empty:
        return None
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color="#101826", line_color="#1C2A48")
    fig, ax = pitch.draw(figsize=(10, 7))
    for _, row in df.iterrows():
        ax.scatter(
            row["y"], row["x"], 
            c="#FF6B6B" if not row.get("isGoal", False) else "#00FFD5",
            s=max(row.get("xG", 0) * 800, 50), alpha=0.8, edgecolors="white"
        )
    ax.set_title(title, color="#00FFD5", fontsize=16)
    return fig

# --- Page Components ---
def create_navigation_header():
    st.markdown("""
    <div style="background:#101826;padding:1rem;border-bottom:2px solid #00FFD5;">
        <h2 style="margin:0;color:#00FFD5;font-family:monospace;">WOSO ANALYTICS</h2>
    </div>
    """, unsafe_allow_html=True)

def display_landing_page():
    st.markdown("<h1 style='text-align:center;color:#00FFD5;'>WOSO ANALYTICS</h1>", unsafe_allow_html=True)
    if st.button("ENTER ANALYTICS SUITE", use_container_width=True, type="primary"):
        st.session_state.app_mode = "MainApp"
        st.rerun()

# --- Scouting Page ---
def display_data_scouting_page():
    st.subheader("Player Performance Analysis")
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

    if metric == "xG":
        columns_to_show = ["player", "team", "xG", "minutes"]
        df_display = df[columns_to_show]
    else:
        df_display = df

    st.dataframe(df_display.head(20), use_container_width=True)

# --- Match Analysis ---
def display_match_analysis_page():
    st.subheader("Match Analysis Centre")
    matches = load_csv(os.path.join(DATA_DIR, "WSL.csv"))
    if matches.empty:
        st.warning("No match data found.")
        return
    match = st.selectbox("Select Match", matches["match"].unique())
    events = load_csv(os.path.join(DATA_DIR, "WSL_events.csv"))
    if not events.empty:
        shots = events[events["type"] == "Shot"]
        fig = create_modern_shot_map(shots, f"{match} Shot Map")
        if fig:
            st.pyplot(fig)

# --- Player Profiles ---
def display_player_profiling_page():
    st.subheader("Player Profiles")

    # Load xG from master CSV
    xg_df = load_competition_data("WSL", "xG").set_index("player")

    # Load other metrics
    data_frames = {"xG": xg_df}
    for metric in ["assists", "gpa", "minutes", "corners", "xDisruption"]:
        df = load_competition_data("WSL", metric)
        if not df.empty:
            data_frames[metric] = df.set_index("player")

    combined = pd.concat(data_frames.values(), axis=1, join="outer").reset_index().fillna(0)
    player = st.selectbox("Select Player", combined["player"].unique())
    row = combined[combined["player"] == player].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Minutes", row.get("minutes", 0))
    col2.metric("xG", row.get("xG", 0))
    col3.metric("Assists", row.get("assists", 0))
    col4.metric("GPA", row.get("gpa", 0))

# --- Corners Page ---
def display_corners_page():
    st.subheader("Set Piece Analysis")
    df = load_competition_data("WSL", "corners")
    if df.empty:
        st.warning("No corner data available.")
        return
    st.dataframe(df.head(20), use_container_width=True)

# --- Main App ---
def main():
    inject_modern_css()
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
