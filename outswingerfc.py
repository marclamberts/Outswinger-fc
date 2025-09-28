import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from mplsoccer.pitch import VerticalPitch
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | StatsBomb Style",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- StatsBomb Inspired Styling ---
def inject_statsbomb_css():
    st.markdown("""
        <style>
        html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0C1A2A; color: #FFFFFF; }
        h1, h2, h3, h4 { color: #00FF88; font-family: 'Roboto Mono', monospace; }
        .stSidebar { background-color: #152642; border-right: 1px solid #1E3A5C; }
        .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: #00FF88; }
        .stSidebar .stMarkdown { color: #B0B7C3; }
        </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_match_xg_data(base_folder="data/matchxg"):
    """Load all CSVs from a folder structure: base_folder/league/*.csv"""
    leagues = {}
    if not os.path.exists(base_folder):
        return leagues
    for league_folder in os.listdir(base_folder):
        league_path = os.path.join(base_folder, league_folder)
        if os.path.isdir(league_path):
            files = glob.glob(os.path.join(league_path, "*.csv"))
            league_dict = {}
            for f in files:
                df = pd.read_csv(f)
                filename = os.path.basename(f)
                league_dict[filename] = df
            leagues[league_folder] = league_dict
    return leagues

def plot_shot_map(df, player_name=None, title_sub="Shot Map"):
    """Plot StatsBomb-style vertical shot map"""
    if df.empty:
        st.warning("No shots to display.")
        return

    if player_name:
        df = df[df['PlayerId'] == player_name]

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#152642', line_color='#00FF88', half=True)
    fig, ax = pitch.draw(figsize=(12, 8))
    
    for _, shot in df.iterrows():
        x, y, xg, is_goal = shot['x'], shot['y'], shot['xG'], shot['isGoal']
        color = '#00FF88' if is_goal else '#FF6B6B'
        size = max(xg * 500, 50)
        ax.scatter(y, x, c=color, s=size, alpha=0.7, edgecolors='white', linewidth=1)

    ax.set_title(title_sub, color="#00FF88", fontsize=18)
    st.pyplot(fig)

# --- Page Display Functions ---
def display_landing_page():
    st.markdown("<h1 style='text-align:center; color:#00FF88;'>WOSO ANALYTICS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#B0B7C3;'>StatsBomb-Inspired Football Intelligence Platform</p>", unsafe_allow_html=True)
    if st.button("ENTER ANALYTICS SUITE"):
        st.session_state.app_mode = "MainApp"

def display_data_scouting_page():
    st.subheader("Player Performance / Data Scouting")
    st.markdown("**League & Metric filters coming soon**")
    st.write("Leaderboard placeholder.")

def display_player_profiling_page():
    st.subheader("Player Profiles")
    st.write("Profile card placeholder.")

def display_corners_page():
    st.subheader("Set Piece Analysis")
    st.write("Corners visualization placeholder.")

def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    league_data = load_match_xg_data("data/matchxg")
    if not league_data:
        st.warning("No match data found.")
        return

    # Sidebar filters
    st.sidebar.markdown("### Match Filters")
    league_selected = st.sidebar.selectbox("Select League", list(league_data.keys()))
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

    match_selected_idx = st.sidebar.selectbox(
        "Select Match",
        range(len(match_display_names)),
        format_func=lambda x: match_display_names[x]
    )
    match_display_name = match_display_names[match_selected_idx]
    match_name = match_files[match_selected_idx]
    df_match = matches_in_league[match_name]

    team1, team2 = team_names_map[match_display_name]

    # Team filter
    team_options = ["Full Match"]
    if team1: team_options.append(team1)
    if team2: team_options.append(team2)
    team_filter = st.sidebar.selectbox("Select Team", team_options)

    # Filter shots by team
    if team_filter != "Full Match" and 'Team' in df_match.columns:
        df_team = df_match[df_match['Team'] == team_filter]
    else:
        df_team = df_match.copy()

    # Optional player filter
    player_name = None
    if 'PlayerId' in df_team.columns:
        player_list = ["All"] + df_team['PlayerId'].unique().tolist()
        player_selected = st.sidebar.selectbox("Select Player", player_list)
        player_name = None if player_selected == "All" else player_selected

    # Plot shot map
    plot_shot_map(
        df_team,
        player_name,
        title_sub=f"{team_filter} | {match_display_name} | {league_selected}"
    )

# --- Main App Logic ---
def main():
    inject_statsbomb_css()

    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"
    if 'page_view' not in st.session_state:
        st.session_state.page_view = "Performance"

    if st.session_state.app_mode == "Landing":
        display_landing_page()
    else:
        # Tabs
        tabs = st.tabs(["📊 Performance", "🎯 Matches", "👤 Profiles", "⛳ Set Pieces"])
        with tabs[0]:
            display_data_scouting_page()
        with tabs[1]:
            display_matches_page()
        with tabs[2]:
            display_player_profiling_page()
        with tabs[3]:
            display_corners_page()

if __name__ == "__main__":
    main()
