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

def plot_shot_map_with_stats(df_team, title_main, title_sub):
    if df_team is None or df_team.empty:
        st.warning("No shot data available for this selection.")
        return

    # --- Summary stats ---
    total_shots = int(df_team.shape[0])
    total_goals = int(df_team['isGoal'].sum()) if 'isGoal' in df_team.columns else 0
    non_penalty_goals = (
        int(df_team[(df_team['Type_of_play'] != 'Penalty') & (df_team['isGoal'] == True)].shape[0])
        if 'Type_of_play' in df_team.columns and 'isGoal' in df_team.columns
        else total_goals
    )
    total_xG = float(df_team['xG'].sum()) if 'xG' in df_team.columns else 0.0
    total_xG_from_pen = (
        float(df_team[df_team.get('Type_of_play','') == 'Penalty']['xG'].sum())
        if ('xG' in df_team.columns and 'Type_of_play' in df_team.columns)
        else 0.0
    )
    total_npxG = total_xG - total_xG_from_pen
    xG_per_shot = (total_xG / total_shots) if total_shots > 0 else 0.0

    # --- Pitch ---
    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12, 8))

    colors = {"missed": "#003f5c", "goal": "#bc5090"}
    max_size = 300

    # Plot shots
    for _, row in df_team.iterrows():
        try:
            x, y = row['x'], row['y']
            isgoal = bool(row.get('isGoal', False))
            xg = float(row.get('xG', 0.0))
        except Exception:
            continue
        color = colors["goal"] if isgoal else colors["missed"]
        size = min(max(xg * 500, 30), max_size)
        ax.scatter(y, x, color=color, s=size, alpha=0.8, edgecolors='white', linewidth=0.8, zorder=3)

    # Titles
    ax.text(0.5, 1.12, title_main, transform=ax.transAxes, fontsize=22, fontweight='bold',
            ha='center', va='top', color='black')
    ax.text(0.5, 1.06, title_sub, transform=ax.transAxes, fontsize=12, fontstyle='italic',
            ha='center', va='top', color='black')

    # Stat circles
    circle_positions = [(0.15,-0.18),(0.35,-0.18),(0.55,-0.18),(0.75,-0.18),(0.87,-0.18),(0.95,-0.18)]
    circle_texts = ["Shots","Goals","NP Goals","xG/Shot","Total xG","Total NpxG"]
    values = [total_shots,total_goals,non_penalty_goals,f"{xG_per_shot:.2f}",f"{total_xG:.2f}",f"{total_npxG:.2f}"]
    circle_colors = ["#152642","#00FF88","#00FF88","#58508d","#58508d","#58508d"]

    for (cx,cy),label,val,col in zip(circle_positions,circle_texts,values,circle_colors):
        circ = Circle((cx, cy), 0.05, transform=ax.transAxes, color=col, zorder=5, clip_on=False)
        ax.add_artist(circ)
        ax.text(cx, cy+0.06, label, transform=ax.transAxes, color='white', fontsize=10, ha='center', va='center')
        ax.text(cx, cy, str(val), transform=ax.transAxes, color='black', fontsize=11, fontweight='bold', ha='center', va='center')

    st.pyplot(fig)


def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    league_data = load_match_xg_data("data/matchxg")
    if not league_data:
        st.warning("No match data found.")
        return

    with st.sidebar:
        st.markdown("### Match Filters")
        total_option = st.selectbox("Total", ["No", "Yes"])
        league_selected = st.selectbox("League", list(league_data.keys()))

        df_team, title_main, title_sub = None, "", ""

        if total_option == "Yes":
            # --- aggregate all matches for selected league ---
            matches_in_league = league_data[league_selected]
            all_matches = pd.concat(matches_in_league.values(), ignore_index=True)

            if 'Team' in all_matches.columns and 'TeamId' in all_matches.columns:
                team_map = all_matches[['Team','TeamId']].drop_duplicates().set_index('Team')['TeamId'].to_dict()
                team_choice = st.selectbox("Team", sorted(team_map.keys()))
                df_team = all_matches[all_matches['TeamId'] == team_map[team_choice]]
                title_main = team_choice
                title_sub = f"All Matches | {league_selected}"
            elif 'TeamId' in all_matches.columns:
                team_choice = st.selectbox("Team", sorted(all_matches['TeamId'].unique()))
                df_team = all_matches[all_matches['TeamId'] == team_choice]
                title_main = str(team_choice)
                title_sub = f"All Matches | {league_selected}"

        else:
            matches_in_league = league_data[league_selected]
            match_files = list(matches_in_league.keys())

            match_names = [mf.split("_",1)[1] if "_" in mf else mf for mf in match_files]
            match_idx = st.selectbox("Match", range(len(match_names)), format_func=lambda i: match_names[i])
            match_name = match_files[match_idx]
            df_match = matches_in_league[match_name]

            # Team selection
            if 'Team' in df_match.columns and 'TeamId' in df_match.columns:
                team_map = df_match[['Team','TeamId']].drop_duplicates().set_index('Team')['TeamId'].to_dict()
                team_choice = st.selectbox("Team", sorted(team_map.keys()))
                df_team = df_match[df_match['TeamId'] == team_map[team_choice]]
                opponent = [t for t in team_map.keys() if t != team_choice]
                title_main = team_choice
                title_sub = f"vs {opponent[0] if opponent else ''} | {league_selected}"
            elif 'TeamId' in df_match.columns:
                team_choice = st.selectbox("Team", sorted(df_match['TeamId'].unique()))
                df_team = df_match[df_match['TeamId'] == team_choice]
                title_main = str(team_choice)
                title_sub = f"{match_name} | {league_selected}"

    # --- Plot pitch + stats ---
    plot_shot_map_with_stats(df_team, title_main, title_sub)


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
