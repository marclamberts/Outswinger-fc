```python
import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.patches import Circle

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | Modern Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
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

# --- Load Metrics CSVs ---
@st.cache_data(ttl=3600)
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

# --- Plotting with Stats ---
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
    total_xG_from_pen = (
        float(df_team[df_team['Type_of_play'] == 'Penalty']['xG'].sum())
        if ('xG' in df_team.columns and 'Type_of_play' in df_team.columns)
        else 0.0
    )
    total_xG_minus_penalties = total_xG - total_xG_from_pen
    xG_per_shot = (total_xG / total_shots) if total_shots > 0 else 0.0

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=True)
    fig, ax = pitch.draw(figsize=(12, 8))

    colors = {"missed": "#003f5c", "goal": "#bc5090"}
    max_size = 300

    for _, row in df_team.iterrows():
        color = colors["goal"] if row.get("isGoal", False) else colors["missed"]
        size = min(row.get("xG", 0) * 500, max_size)
        ax.scatter(row["y"], row["x"], color=color, s=size, alpha=0.7, zorder=3)

    # Title and subtitle
    ax.text(52, 109, title_main, fontsize=20, weight='bold', color='black', ha='center', va='top')
    ax.text(52, 105, title_sub, fontsize=12, color='black', ha='center', va='top')

    # Summary stats circles
    circle_positions = [
        (0.15, -0.15), (0.35, -0.15), (0.55, -0.15),
        (0.15, -0.3), (0.35, -0.3), (0.55, -0.3)
    ]
    circle_texts = ["Shots", "Goals", "NP Goals", "xG/Shot", "Total xG", "Total NpxG"]
    values = [total_shots, total_goals, non_penalty_goals,
              round(xG_per_shot, 2), round(total_xG, 2), round(total_xG_minus_penalties, 2)]
    circle_colors = [colors["missed"], colors["goal"], colors["goal"], "#58508d", "#58508d", "#58508d"]

    for pos, text, value, color in zip(circle_positions, circle_texts, values, circle_colors):
        circle = Circle(pos, 0.04, transform=ax.transAxes, color=color, zorder=5, clip_on=False)
        ax.add_artist(circle)
        ax.text(pos[0], pos[1] + 0.06, text, transform=ax.transAxes, color='black', fontsize=12,
                ha='center', va='center', zorder=6)
        ax.text(pos[0], pos[1], value, transform=ax.transAxes, color='white', fontsize=12,
                weight='bold', ha='center', va='center', zorder=6)

    st.pyplot(fig)

# --- Matches Page ---
def display_matches_page():
    st.subheader("Match Analysis / xG Shot Maps")

    league_data = load_match_xg_data("data/matchxg")
    if not league_data:
        st.warning("No match data found.")
        return

    with st.sidebar:
        st.markdown("### Match Filters")
        total_option = st.selectbox("Total", ["No", "Yes"])
        league_selected = st.selectbox("Select League", list(league_data.keys()))

        df_team = pd.DataFrame()
        title_main, title_sub = "", ""

        if total_option == "Yes":
            league_matches = league_data[league_selected]
            combined = pd.concat(league_matches.values(), ignore_index=True)
            teams = combined["TeamId"].unique().tolist()
            team_selected = st.selectbox("Select Team", teams)
            df_team = combined[combined["TeamId"] == team_selected].copy()
            title_main = team_selected
            title_sub = f"All Matches | {league_selected}"
        else:
            matches_in_league = league_data[league_selected]
            match_files = list(matches_in_league.keys())
            match_names = [mf.split("_", 1)[1] if "_" in mf else mf for mf in match_files]

            match_idx = st.selectbox("Select Match", range(len(match_names)), format_func=lambda i: match_names[i])
            match_name = match_files[match_idx]
            df_match = matches_in_league[match_name]

            teams = df_match["TeamId"].unique().tolist()
            team_selected = st.selectbox("Select Team", teams + ["Full Match"])
            if team_selected == "Full Match":
                df_team = df_match.copy()
                title_main = match_names[match_idx]
                title_sub = league_selected
            else:
                df_team = df_match[df_match["TeamId"] == team_selected].copy()
                opponent = [t for t in teams if t != team_selected]
                title_main = team_selected
                title_sub = f"vs {opponent[0] if opponent else ''} | {league_selected}"

    if not df_team.empty:
        plot_shot_map_with_stats(df_team, title_main, title_sub)

# --- Advanced Metrics Page ---
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

# --- Main App ---
def main():
    inject_custom_css()
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"

    metrics = load_all_metrics("data")

    # Tab navigation
    tabs = st.tabs(["🏠 Home", "📊 Matches", "📈 Advanced Metrics"])
    with tabs[0]:
        st.title("⚽ WoSo Analytics")
        st.markdown("Welcome to the modern scouting & recruitment dashboard.")
    with tabs[1]:
        display_matches_page()
    with tabs[2]:
        display_performance_page(metrics)

if __name__ == "__main__":
    main()
```
