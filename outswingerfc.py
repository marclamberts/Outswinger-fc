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
            .stButton>button { background-color:#00FFA3; color:#0f1923; font-weight:600; border-radius:6px; padding:0.7rem 1.2rem; }
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

def load_all_metrics(base_path="data"):
    metrics = {}
    files = glob.glob(os.path.join(base_path, "*.csv"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        metrics[name] = df
    return metrics

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
def plot_shot_map(df, title_sub="Shot map"):
    if df.empty:
        st.warning("No shot data available.")
        return

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12,8))

    colors = {"missed": "#003f5c", "goal": "#bc5090", "on_target": "#58508d"}
    max_size = 300

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

    ax.text(52,105,title_sub, fontsize=20, weight='bold', color='black', ha='center', va='top')
    st.pyplot(fig)

# --- Page Displays ---
def display_performance_page(metrics):
    st.subheader("Advanced Metrics / Player Scouting")
    if not metrics:
        st.warning("No metric CSVs loaded.")
        return
    metric = st.selectbox("Select Metric", list(metrics.keys()))
    df_metric = metrics[metric]
    st.dataframe(df_metric)

def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    # Load league/match data
    league_data = load_match_xg_data("data/matchxg")
    if not league_data:
        st.warning("No match data found.")
        return

    # --- Sidebar filters only when Matches tab is active ---
    with st.sidebar:
        st.markdown("### Match Filters")

        # League selection
        league_selected = st.selectbox("Select League", list(league_data.keys()))
        matches_in_league = league_data[league_selected]

        # Parse team names from filenames
        match_display_names = []
        team_names_map = {}
        match_files = list(matches_in_league.keys())
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

        # Match selection
        match_selected_idx = st.selectbox(
            "Select Match",
            range(len(match_display_names)),
            format_func=lambda x: match_display_names[x]
        )

        match_display_name = match_display_names[match_selected_idx]
        match_name = match_files[match_selected_idx]
        df_match = matches_in_league[match_name]

        team1, team2 = team_names_map[match_display_name]

        # Team selection (choose team1, team2, or Full Match)
        team_options = ["Full Match"]
        if team1: team_options.append(team1)
        if team2: team_options.append(team2)
        team_filter = st.selectbox("Select Team", team_options)

    # Filter shots by team
    if team_filter != "Full Match" and 'Team' in df_match.columns:
        df_team = df_match[df_match['Team'] == team_filter].copy()
    else:
        df_team = df_match.copy()

    # Determine opponent team name for title
    if team_filter == team1:
        opponent = team2
    elif team_filter == team2:
        opponent = team1
    else:
        opponent = None

    opponent_text = f"vs {opponent}" if opponent else ""
    title_main = team_filter if team_filter != "Full Match" else "Full Match"
    title_sub = f"{opponent_text} | {league_selected}" if opponent else league_selected

    # --- Plot Shot Map ---
    if df_team.empty:
        st.warning("No shot data for selected team.")
        return

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12,8))

    colors = {"missed": "#003f5c", "goal": "#bc5090", "on_target": "#58508d"}
    max_size = 300

    for _, row in df_team.iterrows():
        color = colors["goal"] if row.get("isGoal", False) else colors["missed"]
        size = min(row.get("xG",0)*500, max_size)
        ax.scatter(row["y"], row["x"], color=color, s=size, alpha=0.7, zorder=3)

    # Summary stats
    total_shots = df_team.shape[0]
    total_goals = df_team['isGoal'].sum()
    non_penalty_goals = df_team[(df_team['Type_of_play'] != 'Penalty') & (df_team['isGoal'] == True)].shape[0]
    total_xG = df_team['xG'].sum()
    total_xG_minus_penalties = total_xG - df_team[df_team['Type_of_play']=="Penalty"]['xG'].sum()
    xG_per_shot = total_xG / total_shots if total_shots > 0 else 0

    circle_positions = [(0.15,-0.15),(0.35,-0.15),(0.55,-0.15),(0.15,-0.3),(0.35,-0.3),(0.55,-0.3)]
    circle_texts = ["Shots","Goals","NP Goals","xG/Shot","Total xG","Total NpxG"]
    values = [total_shots,total_goals,non_penalty_goals,round(xG_per_shot,2),round(total_xG,2),round(total_xG_minus_penalties,2)]
    circle_colors = [colors["missed"],colors["goal"],colors["goal"],colors["on_target"],colors["on_target"],colors["on_target"]]

    for pos, text, value, color in zip(circle_positions, circle_texts, values, circle_colors):
        circle = Circle(pos,0.04, transform=ax.transAxes,color=color,zorder=5,clip_on=False)
        ax.add_artist(circle)
        ax.text(pos[0], pos[1]+0.06, text, transform=ax.transAxes, color='black', fontsize=12, ha='center', va='center', zorder=6)
        ax.text(pos[0], pos[1], value, transform=ax.transAxes, color='white', fontsize=12, weight='bold', ha='center', va='center', zorder=6)

    # Multi-line title
    ax.text(52, 105, title_main, fontsize=22, weight='bold', color='black', ha='center', va='top')
    ax.text(52, 101, title_sub, fontsize=14, style='italic', color='black', ha='center', va='top')

    st.pyplot(fig)



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

# --- Landing / Main Menu ---
def display_landing_page():
    st.markdown("<h1 style='text-align:center;'>WOSO ANALYTICS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#00FFA3;'>Select a Section</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    metrics = load_all_metrics("data")  # load early for Performance & Profiles

    if col1.button("Advanced Metrics"):
        st.session_state.app_mode = "Performance"
    if col2.button("Matches"):
        st.session_state.app_mode = "Matches"
    if col3.button("Player Shot Maps"):
        st.session_state.app_mode = "Profiles"
    if col4.button("Corners"):
        st.session_state.app_mode = "SetPieces"

# --- Main App ---
def main():
    inject_custom_css()
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"

    metrics = load_all_metrics("data")

    if st.session_state.app_mode=="Landing":
        display_landing_page()
    elif st.session_state.app_mode=="Performance":
        display_performance_page(metrics)
    elif st.session_state.app_mode=="Matches":
        display_matches_page()
    elif st.session_state.app_mode=="Profiles":
        display_profiles_page(metrics)
    elif st.session_state.app_mode=="SetPieces":
        display_set_pieces_page()

if __name__=="__main__":
    main()
