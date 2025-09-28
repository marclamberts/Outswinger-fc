import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.patches import Circle

# --- App Config ---
st.set_page_config(
    page_title="WoSo Analytics | Modern Scouting",
    page_icon="⚽",
    layout="wide"
)

# --- Custom CSS ---
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
        </style>
    """, unsafe_allow_html=True)

# --- Load CSVs ---
@st.cache_data(ttl=3600)
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
    for league in os.listdir(base_path):
        league_path = os.path.join(base_path, league)
        if not os.path.isdir(league_path): continue
        league_data[league] = {}
        for f in glob.glob(os.path.join(league_path, "*.csv")):
            df = pd.read_csv(f)
            match_name = os.path.splitext(os.path.basename(f))[0]
            league_data[league][match_name] = df
    return league_data

# --- Shotmap ---
def plot_shot_map_with_stats(df_team, title_main, title_sub):
    if df_team is None or df_team.empty:
        st.warning("No shot data available for this selection.")
        return

    total_shots = int(df_team.shape[0])
    total_goals = int(df_team['isGoal'].sum()) if 'isGoal' in df_team.columns else 0
    non_penalty_goals = (
        int(df_team[(df_team['Type_of_play'] != 'Penalty') & (df_team['isGoal'] == True)].shape[0])
        if 'Type_of_play' in df_team.columns and 'isGoal' in df_team.columns
        else total_goals
    )
    total_xG = float(df_team['xG'].sum()) if 'xG' in df_team.columns else 0.0
    total_xG_minus_penalties = total_xG - df_team[df_team.get('Type_of_play','')=="Penalty"]['xG'].sum() if 'Type_of_play' in df_team.columns else total_xG
    xG_per_shot = (total_xG / total_shots) if total_shots > 0 else 0.0

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12, 8))

    colors = {"missed": "#003f5c", "goal": "#bc5090"}
    max_size = 300
    for _, row in df_team.iterrows():
        color = colors["goal"] if row.get("isGoal", False) else colors["missed"]
        size = min(row.get("xG", 0) * 500, max_size)
        ax.scatter(row["y"], row["x"], color=color, s=size, alpha=0.7, zorder=3)

    ax.text(52, 109, title_main, fontsize=20, weight='bold', color='black', ha='center', va='top')
    ax.text(52, 105, title_sub, fontsize=12, color='black', ha='center', va='top')

    # Summary circles
    circle_positions = [(0.15,-0.15),(0.35,-0.15),(0.55,-0.15),(0.15,-0.3),(0.35,-0.3),(0.55,-0.3)]
    circle_texts = ["Shots","Goals","NP Goals","xG/Shot","Total xG","Total NpxG"]
    values = [total_shots,total_goals,non_penalty_goals,round(xG_per_shot,2),round(total_xG,2),round(total_xG_minus_penalties,2)]
    circle_colors = ["#152642","#00FF88","#00FF88","#58508d","#58508d","#58508d"]
    for pos, text, val, col in zip(circle_positions, circle_texts, values, circle_colors):
        circ = Circle(pos, 0.05, transform=ax.transAxes, color=col, zorder=5, clip_on=False)
        ax.add_artist(circ)
        ax.text(pos[0], pos[1]+0.06, text, transform=ax.transAxes, color='white', fontsize=10, ha='center', va='center')
        ax.text(pos[0], pos[1], val, transform=ax.transAxes, color='black', fontsize=11, fontweight='bold', ha='center', va='center')

    st.pyplot(fig)

# --- Pages ---
def display_landing_page():
    st.title("⚽ WoSo Analytics")
    st.markdown("Welcome to the modern scouting & recruitment dashboard.")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    if col1.button("Advanced Metrics"):
        st.session_state.app_mode = "Performance"
    if col2.button("Matches"):
        st.session_state.app_mode = "Matches"
    if col3.button("Player Shot Maps"):
        st.session_state.app_mode = "Profiles"
    if col4.button("Corners"):
        st.session_state.app_mode = "SetPieces"

def display_performance_page(metrics):
    st.subheader("Advanced Metrics / Player Scouting")
    if not metrics:
        st.warning("No metric CSVs loaded.")
        return
    with st.sidebar:
        st.markdown("### Metrics Filters")
        metric = st.selectbox("Select Metric Dataset", list(metrics.keys()))
    df_metric = metrics[metric]
    st.dataframe(df_metric)

def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")
    league_data = load_match_xg_data("data/matchxg")
    if not league_data: st.warning("No match data found."); return

    with st.sidebar:
        st.markdown("### Match Filters")
        total_option = st.selectbox("Total", ["No", "Yes"])
        league_selected = st.selectbox("League", list(league_data.keys()))
        df_team = pd.DataFrame()
        title_main, title_sub = "", ""

        if total_option=="Yes":
            all_matches = pd.concat(league_data[league_selected].values(), ignore_index=True)
            team_selected = st.selectbox("Team", all_matches["TeamId"].unique())
            df_team = all_matches[all_matches["TeamId"]==team_selected]
            title_main = team_selected
            title_sub = f"All Matches | {league_selected}"
        else:
            match_files = list(league_data[league_selected].keys())
            match_names = [mf.split("_",1)[1] if "_" in mf else mf for mf in match_files]
            match_idx = st.selectbox("Match", range(len(match_names)), format_func=lambda i: match_names[i])
            match_name = match_files[match_idx]
            df_match = league_data[league_selected][match_name]
            teams = df_match["TeamId"].unique().tolist()
            team_selected = st.selectbox("Team", teams+["Full Match"])
            if team_selected=="Full Match":
                df_team = df_match.copy()
                title_main = match_names[match_idx]
                title_sub = league_selected
            else:
                df_team = df_match[df_match["TeamId"]==team_selected]
                opponent = [t for t in teams if t!=team_selected]
                title_main = team_selected
                title_sub = f"vs {opponent[0] if opponent else ''} | {league_selected}"

    if not df_team.empty:
        plot_shot_map_with_stats(df_team, title_main, title_sub)

# --- Main ---
def main():
    inject_custom_css()
    if "app_mode" not in st.session_state: st.session_state.app_mode="Landing"
    metrics = load_all_metrics("data")

    if st.session_state.app_mode=="Landing":
        display_landing_page()
    elif st.session_state.app_mode=="Performance":
        display_performance_page(metrics)
    elif st.session_state.app_mode=="Matches":
        display_matches_page()
    elif st.session_state.app_mode=="Profiles":
        st.subheader("Player Shot Maps / Profiles")  # placeholder
    elif st.session_state.app_mode=="SetPieces":
        st.subheader("Set Pieces / Corners")  # placeholder

if __name__=="__main__":
    main()
