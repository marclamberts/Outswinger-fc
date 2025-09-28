import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mplsoccer.pitch import VerticalPitch
from matplotlib.patches import Circle

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | Modern Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Neon/Modern Styling ---
def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');

            html, body, [class*="st-"] {
                font-family: 'Inter', sans-serif;
                background-color: #0f1923;
                color: #fff;
            }

            h1, h2, h3, h4 { font-family: 'Roboto Mono', monospace; color: #00FFA3; }
            .stButton>button { background-color:#00FFA3; color:#0f1923; font-weight:600; border-radius:6px; }
            .stButton>button:hover { background-color:#00CC7F; }
            .stSelectbox div[data-baseweb="select"] > div { background-color:#152642; color:#fff; border-radius:6px; }
            .stDataFrame { background-color:#152642; border-radius:6px; }
            .stDataFrame .data-grid-header { background-color:#1E3A5C; color:#00FFA3; font-weight:600; }
            .section-header { border-bottom:2px solid #00FFA3; padding-bottom:0.5rem; margin-bottom:1.5rem; font-family:'Roboto Mono'; }
        </style>
    """, unsafe_allow_html=True)

# --- Load CSV Data ---
@st.cache_data(ttl=3600)
def load_csv_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data(ttl=3600)
def load_excel_data(file_path):
    return pd.read_excel(file_path)

# --- Load all metric CSVs ---
def load_all_metrics(base_path="data"):
    metrics = {}
    files = glob.glob(os.path.join(base_path, "*.csv"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        metrics[name] = df
    return metrics

# --- Load Match xG Data ---
@st.cache_data(ttl=3600)
def load_match_xg_data(base_path="data/matchxg"):
    league_data = {}
    leagues = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for league in leagues:
        league_path = os.path.join(base_path, league)
        files = glob.glob(os.path.join(league_path, "*.csv"))
        league_data[league] = {}
        for f in files:
            df = pd.read_csv(f)
            match_name = os.path.splitext(os.path.basename(f))[0]
            league_data[league][match_name] = df
    return league_data

# --- Shot Map Plotting ---
def plot_shot_map(df, player_name=None, title_sub="Shot map"):
    if df.empty:
        st.warning("No shot data available.")
        return
    
    if player_name:
        df = df[df['PlayerId'] == player_name]
        if df.empty:
            st.warning(f"No data for {player_name}.")
            return

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12,8))

    colors = {"missed": "#003f5c", "goal": "#bc5090", "on_target": "#58508d"}
    max_size = 300  # cap marker size

    for _, row in df.iterrows():
        color = colors["goal"] if row.get("isGoal", False) else colors["missed"]
        size = min(row.get("xG",0)*500, max_size)
        ax.scatter(row["y"], row["x"], color=color, s=size, alpha=0.7, zorder=3)

    # Summary stats
    df_plot = df
    total_shots = df_plot.shape[0]
    total_goals = df_plot['isGoal'].sum()
    non_penalty_goals = df_plot[(df_plot['Type_of_play'] != 'Penalty') & (df_plot['isGoal'] == True)].shape[0]
    total_xG = df_plot['xG'].sum()
    total_xG_minus_penalties = total_xG - df_plot[df_plot['Type_of_play']=="Penalty"]['xG'].sum()
    xG_per_shot = total_xG / total_shots if total_shots>0 else 0

    circle_positions = [(0.15,-0.15),(0.35,-0.15),(0.55,-0.15),(0.15,-0.3),(0.35,-0.3),(0.55,-0.3)]
    circle_texts = ["Shots","Goals","NP Goals","xG/Shot","Total xG","Total NpxG"]
    values = [total_shots,total_goals,non_penalty_goals,round(xG_per_shot,2),round(total_xG,2),round(total_xG_minus_penalties,2)]
    circle_colors = [colors["missed"],colors["goal"],colors["goal"],colors["on_target"],colors["on_target"],colors["on_target"]]

    for pos, text, value, color in zip(circle_positions, circle_texts, values, circle_colors):
        circle = Circle(pos,0.04, transform=ax.transAxes,color=color,zorder=5,clip_on=False)
        ax.add_artist(circle)
        ax.text(pos[0], pos[1]+0.06, text, transform=ax.transAxes, color='black', fontsize=12, ha='center', va='center', zorder=6)
        ax.text(pos[0], pos[1], value, transform=ax.transAxes, color='white', fontsize=12, weight='bold', ha='center', va='center', zorder=6)

    title_text = player_name if player_name else "Team/Match"
    ax.text(52,105,title_text, fontsize=20, weight='bold', color='black', ha='center', va='top')
    st.pyplot(fig)

# --- Page Displays ---
def display_landing_page():
    st.markdown("<h1 style='text-align:center;'>WOSO ANALYTICS</h1>", unsafe_allow_html=True)
    if st.button("ENTER DASHBOARD"):
        st.session_state.app_mode = "MainApp"
        st.experimental_rerun()

def display_performance_page(metrics):
    st.subheader("Player Performance Scouting")
    metric = st.selectbox("Select Metric", list(metrics.keys()))
    df_metric = metrics[metric]
    st.dataframe(df_metric)

def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    # Load match xG data
    league_data = load_match_xg_data("data/matchxg")

    st.markdown("**Metric:** Expected Goals (xG)")
    
    # League filter
    league_selected = st.selectbox("Select League", list(league_data.keys()))
    matches_in_league = league_data[league_selected]

    # Extract team names for each match
    match_teams = {}
    for match_name, df in matches_in_league.items():
        teams = df['Team'].unique()
        if len(teams)>=2:
            match_teams[match_name] = tuple(teams[:2])
        else:
            match_teams[match_name] = ("Team 1","Team 2")

    # Select match by Team vs Team
    match_display_names = [f"{t1} vs {t2}" for t1,t2 in match_teams.values()]
    match_idx = st.selectbox("Select Match", range(len(match_display_names)), format_func=lambda x: match_display_names[x])
    match_name = list(matches_in_league.keys())[match_idx]
    df_match = matches_in_league[match_name]

    # Player selection
    player_list = ["All"] + df_match['PlayerId'].unique().tolist()
    player_selected = st.selectbox("Select Player", player_list)
    player_name = None if player_selected=="All" else player_selected

    plot_shot_map(df_match, player_name, title_sub=f"{match_display_names[match_idx]} | {league_selected}")

def display_profiles_page(metrics):
    st.subheader("Player Profiles")
    if 'WSL' not in metrics:
        st.warning("WSL metrics not loaded.")
        return
    player_ids = list(metrics['WSL'].PlayerId.unique())
    player_selected = st.selectbox("Select Player", player_ids)
    df_player = metrics['WSL'][metrics['WSL'].PlayerId==player_selected]
    st.dataframe(df_player)

def display_set_pieces_page():
    st.subheader("Set Pieces / Corners Analysis")
    st.info("Visualizations coming soon.")

# --- Main App ---
def main():
    inject_custom_css()
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"

    metrics = load_all_metrics("data")

    if st.session_state.app_mode=="Landing":
        display_landing_page()
    else:
        tabs = st.tabs(["📊 Performance","🎯 Matches","👤 Profiles","⛳ Set Pieces"])
        with tabs[0]:
            display_performance_page(metrics)
        with tabs[1]:
            display_matches
