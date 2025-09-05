import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch, VerticalPitch
import os
import numpy as np
import json
from functools import reduce

# ==============================================================================
# 1. APP CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Soccer Analytics Hub", layout="wide")

# --- AESTHETIC CHOICES ---
BG_COLOR = '#0d1117'      # Dark background
TEXT_COLOR = '#c9d1d9'    # Light text
PITCH_COLOR = '#0d1117'   # Dark pitch
LINE_COLOR = '#21262d'    # Muted lines for the pitch
TEAM_A_COLOR = '#58a6ff'  # Vibrant blue
TEAM_B_COLOR = '#f0883e'  # Vibrant orange
GOAL_COLOR = '#3fb950'    # Vibrant green

# ==============================================================================
# 2. DATA LOADING & CACHING FUNCTIONS
# ==============================================================================

@st.cache_data
def load_mapping_files():
    """Loads team and event mapping files once and caches them."""
    try:
        # This can be extended for more leagues if needed
        team_mapping_df = pd.read_csv('WSL Matches.csv') 
        home = team_mapping_df[["matchInfo/contestant/0/id", "matchInfo/contestant/0/name"]].rename(columns={"matchInfo/contestant/0/id": "id", "matchInfo/contestant/0/name": "name"})
        away = team_mapping_df[["matchInfo/contestant/1/id", "matchInfo/contestant/1/name"]].rename(columns={"matchInfo/contestant/1/id": "id", "matchInfo/contestant/1/name": "name"})
        team_map_df = pd.concat([home, away]).drop_duplicates('id').dropna()
        team_map_df['id'] = team_map_df['id'].astype(str)
        id_to_name_dict = dict(zip(team_map_df.id, team_map_df.name))
    except FileNotFoundError:
        st.sidebar.error("Mapping file 'WSL Matches.csv' not found.")
        id_to_name_dict = {}

    try:
        event_map_df = pd.read_csv("event_mapping.csv", encoding="ISO-8859-1")
        event_map_df.columns = ["typeId", "Event Type", "Description"]
    except FileNotFoundError:
        st.sidebar.error("Mapping file 'event_mapping.csv' not found.")
        event_map_df = pd.DataFrame(columns=["typeId", "Event Type"])
        
    return id_to_name_dict, event_map_df

@st.cache_data
def get_matches_for_league(league_folder):
    """Scans a league's folder and extracts match metadata for selection."""
    match_files = []
    base_path = os.path.join('data', league_folder)
    if not os.path.exists(base_path):
        return []
    
    for filename in os.listdir(base_path):
        if filename.endswith('.json'):
            file_path = os.path.join(base_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            contestant_ids = []
            if 'matchInfo' in data and 'contestant' in data['matchInfo']:
                 for contestant in data['matchInfo']['contestant']:
                     contestant_ids.append(str(contestant.get('id')))
            
            if len(contestant_ids) == 2:
                 match_files.append({
                     'path': file_path,
                     'ids': frozenset(contestant_ids)
                 })
    return match_files

def load_match_data(json_path, team_map_dict):
    """Loads and processes a single JSON match file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    events = data.get('event', [])
    rows = []
    for e in events:
        row = {'typeId': e.get('typeId'), 'timeMin': e.get('timeMin'), 'contestantId': str(e.get('contestantId')),
               'playerName': e.get('playerName'), 'outcome': 1 if e.get('outcome') == 'Successful' else 0,
               'x': e.get('x'), 'y': e.get('y'), 'id': e.get('id')}
        
        row['xG'] = 0.0
        for q in e.get('qualifier', []):
            if str(q.get('qualifierId')) == '318':
                row['xG'] = float(q.get('value', 0.0))
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame(), "", ""
    
    df['isGoal'] = df['typeId'] == 16
    df['Team'] = df['contestantId'].map(team_map_dict)
    
    teams = df['Team'].dropna().unique()
    team1_name = teams[0] if len(teams) > 0 else "Team A"
    team2_name = teams[1] if len(teams) > 1 else "Team B"
    
    return df, team1_name, team2_name

# ==============================================================================
# 3. VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_shotmap(df, team1_name, team2_name):
    shots = df[df['typeId'].isin([7, 8, 9, 10, 16])].copy()
    team1 = shots[shots['Team'] == team1_name]
    team2 = shots[shots['Team'] == team2_name]
    
    pitch = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=LINE_COLOR, line_zorder=1)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.set_facecolor(BG_COLOR)
    pitch.draw(ax=ax)
    plt.gca().invert_xaxis()

    for _, shot in team1.iterrows():
        color = GOAL_COLOR if shot['isGoal'] else TEAM_A_COLOR
        ax.scatter(shot['x'], 100 - shot['y'], color=color, s=shot['xG'] * 800, alpha=0.8, ec='white', lw=0.5, zorder=3)
    for _, shot in team2.iterrows():
        color = GOAL_COLOR if shot['isGoal'] else TEAM_B_COLOR
        ax.scatter(100 - shot['x'], shot['y'], color=color, s=shot['xG'] * 800, alpha=0.8, ec='white', lw=0.5, zorder=3)

    team1_goals = int(team1['isGoal'].sum())
    team2_goals = int(team2['isGoal'].sum())
    team1_xg = team1['xG'].sum()
    team2_xg = team2['xG'].sum()
    
    title = f"{team1_name} ({team1_goals}) vs. {team2_name} ({team2_goals})"
    ax.text(0.5, 1.02, title, ha='center', va='bottom', fontsize=25, fontweight='bold', color=TEXT_COLOR, transform=ax.transAxes)
    ax.text(80, 88, f"{team1_xg:.2f} xG", color=TEAM_A_COLOR, ha='center', fontsize=20, fontweight='bold')
    ax.text(20, 88, f"{team2_xg:.2f} xG", color=TEAM_B_COLOR, ha='center', fontsize=20, fontweight='bold')
    
    return fig

def plot_flowmap(df, team1_name, team2_name):
    shots = df[df['typeId'].isin([7, 8, 9, 10, 16])].copy().sort_values('timeMin')
    team1 = shots[shots['Team'] == team1_name]
    team2 = shots[shots['Team'] == team2_name]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    t1_cum_xg = np.cumsum(team1['xG'])
    t2_cum_xg = np.cumsum(team2['xG'])
    
    ax.step([0, *team1['timeMin']], [0, *t1_cum_xg], color=TEAM_A_COLOR, lw=4, where='post', label=team1_name)
    ax.step([0, *team2['timeMin']], [0, *t2_cum_xg], color=TEAM_B_COLOR, lw=4, where='post', label=team2_name)
    ax.fill_between([0, *team1['timeMin']], [0, *t1_cum_xg], color=TEAM_A_COLOR, alpha=0.3, step='post')
    ax.fill_between([0, *team2['timeMin']], [0, *t2_cum_xg], color=TEAM_B_COLOR, alpha=0.3, step='post')
    
    ax.grid(True, ls='--', color=LINE_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(LINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=12)
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute', color=TEXT_COLOR, fontsize=14)
    plt.ylabel('Cumulative xG', color=TEXT_COLOR, fontsize=14)
    legend = plt.legend()
    plt.setp(legend.get_texts(), color=TEXT_COLOR)

    return fig
    
def plot_field_tilt(df, team1_name, team2_name):
    final_third_passes = df[(df['typeId'] == 1) & (df['x'] > 66.7)]
    minutes = list(range(0, int(df['timeMin'].max()) + 2))
    t1_tilt_raw = []
    
    for minute in minutes:
        min_df = final_third_passes[final_third_passes['timeMin'] == minute]
        t1_passes = min_df[min_df['Team'] == team1_name].shape[0]
        t2_passes = min_df[min_df['Team'] == team2_name].shape[0]
        total = t1_passes + t2_passes
        t1_tilt_raw.append((t1_passes / total) * 100 if total > 0 else 50)
        
    tilt = pd.Series(t1_tilt_raw).rolling(window=10, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    ax.plot(minutes, tilt, color=TEAM_A_COLOR, lw=3)
    ax.plot(minutes, 100 - tilt, color=TEAM_B_COLOR, lw=3)
    ax.axhline(50, color=LINE_COLOR, linestyle='--', lw=2)
    ax.fill_between(minutes, 50, tilt, where=tilt > 50, color=TEAM_A_COLOR, alpha=0.5, label=f'{team1_name} Dominance')
    ax.fill_between(minutes, 50, tilt, where=tilt < 50, color=TEAM_B_COLOR, alpha=0.5, label=f'{team2_name} Dominance')

    ax.grid(True, ls='--', color=LINE_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(LINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=12)
    plt.ylim(0, 100)
    plt.xlim(0, max(minutes) - 1)
    plt.ylabel('Field Tilt (%)', fontsize=14, color=TEXT_COLOR)
    plt.xlabel('Minute', fontsize=14, color=TEXT_COLOR)
    legend = plt.legend()
    plt.setp(legend.get_texts(), color=TEXT_COLOR)
    
    return fig

def plot_pass_network(df, team_name):
    team_data = df[df['Team'] == team_name].copy()
    team_data['recipient'] = team_data['playerName'].shift(-1)
    passes = team_data[(team_data['typeId'] == 1) & (team_data['outcome'] == 1)]
    
    if passes.empty:
        st.warning(f"No completed passes found for {team_name}.")
        return None

    avg_locs = passes.groupby('playerName').agg({'x': ['mean'], 'y': ['mean']})
    avg_locs.columns = ['x', 'y']
    passes_between = passes.groupby(['playerName', 'recipient']).id.count().reset_index().rename(columns={'id': 'pass_count'})
    passes_between = passes_between.merge(avg_locs, left_on='playerName', right_index=True)
    passes_between = passes_between.merge(avg_locs, left_on='recipient', right_index=True, suffixes=('', '_end'))
    
    pitch = VerticalPitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=LINE_COLOR)
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.set_facecolor(BG_COLOR)
    pitch.draw(ax=ax)
    
    max_lw = 10
    max_pass_count = passes_between['pass_count'].max() if not passes_between.empty else 0
    
    if max_pass_count > 0:
        pitch.lines(passes_between.x, passes_between.y, passes_between.x_end, passes_between.y_end,
                            lw=passes_between.pass_count / max_pass_count * max_lw,
                            color=TEXT_COLOR, zorder=1, ax=ax)
    
    pitch.scatter(avg_locs.x, avg_locs.y, s=800, color=BG_COLOR, edgecolors=TEAM_A_COLOR, linewidth=3, ax=ax, zorder=2)
    for i, row in avg_locs.iterrows():
        pitch.annotate(i.split()[-1], xy=(row.x, row.y), c=TEXT_COLOR, va='center', ha='center', size=12, ax=ax, zorder=3)
        
    return fig

# ==============================================================================
# 4. STREAMLIT APP UI
# ==============================================================================

st.title("Soccer Analytics Hub")

# --- Load Data ---
ID_TO_NAME, EVENT_MAP = load_mapping_files()
# Define leagues and their corresponding folder names
LEAGUES = {
    "WSL": "WSL 2024-2025", 
    "NWSL": "NWSL" # Add other leagues here as you add folders
}

# --- Top Menu: League Selection ---
selected_league_name = st.radio("SELECT LEAGUE", list(LEAGUES.keys()), horizontal=True)

# --- Sidebar: Controls ---
st.sidebar.title("Controls")
selected_visual = st.sidebar.selectbox("Select Visualisation", ["Shot Map", "Flow Map", "Field Tilt", "Pass Network"])

# --- Dynamic Sidebar: Team and Match Selection ---
if selected_league_name:
    league_folder = LEAGUES[selected_league_name]
    all_matches = get_matches_for_league(league_folder)
    
    league_team_ids = reduce(lambda x, y: x.union(y['ids']), all_matches, set())
    league_teams = sorted([ID_TO_NAME[id] for id in league_team_ids if id in ID_TO_NAME])
    
    if not league_teams:
        st.sidebar.warning(f"No teams found for {selected_league_name}. Check your data folder and mapping file.")
    else:
        team_matches = all_matches
        selected_team = None
        if selected_visual == "Pass Network":
            selected_team = st.sidebar.selectbox("Select Team", league_teams)
            if selected_team:
                team_id = [k for k, v in ID_TO_NAME.items() if v == selected_team][0]
                team_matches = [m for m in all_matches if team_id in m['ids']]

        match_labels = {f"{ID_TO_NAME.get(list(m['ids'])[0], '?')} vs. {ID_TO_NAME.get(list(m['ids'])[1], '?')}": m['path'] for m in team_matches}
        
        if not match_labels:
            st.sidebar.warning(f"No matches found for the selected team or league.")
        else:
            selected_match_label = st.sidebar.selectbox("Select Match", list(match_labels.keys()))

            if selected_match_label:
                json_path = match_labels[selected_match_label]
                df, team1, team2 = load_match_data(json_path, ID_TO_NAME)
                
                if not df.empty:
                    st.header(f"{selected_match_label}")
                    st.markdown(f"**Visualisation:** {selected_visual}")
                    
                    if selected_visual == "Shot Map":
                        fig = plot_shotmap(df, team1, team2)
                    elif selected_visual == "Flow Map":
                        fig = plot_flowmap(df, team1, team2)
                    elif selected_visual == "Field Tilt":
                        fig = plot_field_tilt(df, team1, team2)
                    elif selected_visual == "Pass Network":
                        st.header(f"Pass Network for {selected_team}")
                        fig = plot_pass_network(df, selected_team)
                    
                    if 'fig' in locals() and fig is not None:
                        st.pyplot(fig, use_container_width=True)
                else:
                    st.error("Failed to load or process data for the selected match.")