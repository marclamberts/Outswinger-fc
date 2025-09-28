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
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; background-color: #0f1923; color: #fff; }
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
    if not os.path.exists(base_path):
        return league_data
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
    total_shots = df.shape[0]
    total_goals = df['isGoal'].sum()
    non_penalty_goals = df[(df['Type_of_play'] != 'Penalty') & (df['isGoal'] == True)].shape[0]
    total_xG = df['xG'].sum()
    total_xG_minus_penalties = total_xG - df[df['Type_of_play']=="Penalty"]['xG'].sum()
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

def display_performance_page(metrics):
    st.subheader("Player Performance Scouting")
    if not metrics:
        st.warning("No metric CSVs loaded.")
        return
    metric = st.selectbox("Select Metric", list(metrics.keys()))
    df_metric = metrics[metric]
    st.dataframe(df_metric)

def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    league_data = load_match_xg_data("data/matchxg")
    if not league_data:
        st.warning("No match data found.")
        return

    st.markdown("**Metric:** Expected Goals (xG)")
    league_selected = st.selectbox("Select League", list(league_data.keys()))
    matches_in_league = league_data[league_selected]

    # Parse team names from filenames
    match_display_names = []
    match_files = list(matches_in_league.keys())
    team_names_map = {}  # store team names for each match
    for match_file in match_files:
        parts = match_file.split("_", 1)
        name_part = parts[1] if len(parts) > 1 else parts[0]
        if " - " in name_part:
            t1, t2 = name_part.split(" - ", 1)
            display_name = f"{t1.strip()} vs {t2.strip()}"
            team_names_map[display_name] = (t1.strip(), t2.strip())
        else:
            display_name = name_part
            team_names_map[display_name] = (None, None)
        match_display_names.append(display_name)

    match_selected_idx = st.selectbox("Select Match", range(len(match_display_names)), 
                                      format_func=lambda x: match_display_names[x])
    match_display_name = match_display_names[match_selected_idx]
    match_name = match_files[match_selected_idx]
    df_match = matches_in_league[match_name]

    team1, team2 = team_names_map[match_display_name]

    # Create two buttons for team selection
    col1, col2 = st.columns(2)
    team_filter = None
    with col1:
        if st.button(team1):
            team_filter = team1
    with col2:
        if st.button(team2):
            team_filter = team2

    # Optional player-level filtering within team
    player_name = None
    if team_filter and 'Team' in df_match.columns:
        df_team = df_match[df_match['Team'] == team_filter]
        if 'PlayerId' in df_team.columns:
            player_list = ["All"] + df_team['PlayerId'].unique().tolist()
            player_selected = st.selectbox("Select Player", player_list)
            player_name = None if player_selected == "All" else player_selected
        else:
            player_name = None
    else:
        df_team = df_match  # no team filter

    # Display shot map
    plot_shot_map(df_team, player_name, title_sub=f"{team_filter if team_filter else 'Full Match'} | {match_display_name} | {league_selected}")


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
            display_matches_page()
        with tabs[2]:
            display_profiles_page(metrics)
        with tabs[3]:
            display_set_pieces_page()

if __name__=="__main__":
    main()
