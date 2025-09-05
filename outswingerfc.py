import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import io
import os
import numpy as np
import json

# ==============================================================================
# 1. HELPER FUNCTION FOR LOADING AND PROCESSING JSON DATA
# ==============================================================================

def load_and_process_json(json_path, team_map_dict, event_map_df):
    """
    Loads a JSON event data file, flattens it, and merges it with team/event mappings.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = data.get('event', [])
    rows = []

    for e in events:
        row = {
            'id': e.get('id'),
            'eventId': e.get('eventId'),
            'typeId': e.get('typeId'),
            'periodId': e.get('periodId'),
            'timeMin': e.get('timeMin'),
            'timeSec': e.get('timeSec'),
            'contestantId': str(e.get('contestantId')), # Ensure ID is string for mapping
            'playerId': e.get('playerId'),
            'playerName': e.get('playerName'),
            'outcome': 1 if e.get('outcome') == 'Successful' else 0,
            'x': e.get('x'),
            'y': e.get('y'),
            'timeStamp': e.get('timeStamp'),
            'xG': 0.0,
            'PsxG': 0.0,
            'epv': 0.0,
            'isFromCorner': 0,
            'isSetPiece': 0,
            'isFastBreak': 0,
            'isThrowIn': 0,
        }

        # Extract key metrics and play types from qualifiers
        for q in e.get('qualifier', []):
            qid = str(q.get('qualifierId'))
            val = q.get('value', 1)
            if qid == '318': row['xG'] = float(val)
            if qid == '321': row['PsxG'] = float(val)
            # Add EPV if available, assuming a qualifierId (e.g., '333')
            # if qid == '333': row['epv'] = float(val)
            if qid == '5': row['isFromCorner'] = 1
            if qid == '6': row['isSetPiece'] = 1
            if qid == '23': row['isFastBreak'] = 1
            if qid == '107': row['isThrowIn'] = 1
        
        rows.append(row)

    df = pd.DataFrame(rows)

    # Return empty DataFrame if no data
    if df.empty:
        return pd.DataFrame()

    # Create 'Type_of_play' column for Shot Map
    def get_play_type(row):
        # typeId for Penalty is 9
        if row['typeId'] == 9: return 'Penalty'
        if row['isFromCorner'] == 1: return 'FromCorner'
        if row['isSetPiece'] == 1: return 'SetPiece'
        if row['isFastBreak'] == 1: return 'FastBreak'
        if row['isThrowIn'] == 1: return 'ThrowinSetPiece'
        return 'RegularPlay'

    df['Type_of_play'] = df.apply(get_play_type, axis=1)

    # Create 'isGoal' column (typeId for Goal is 16)
    df['isGoal'] = df['typeId'] == 16
    
    # Merge with event type names
    df = df.merge(event_map_df[['typeId', 'Event Type']], on='typeId', how='left')
    
    # Map contestantId to team name
    df['Team'] = df['contestantId'].map(team_map_dict)

    return df


# ==============================================================================
# 2. INITIAL APP SETUP AND DATA MAPPING LOADING
# ==============================================================================

# Streamlit app layout and settings
st.set_page_config(page_title="Outswinger FC - Data visualisation app", layout="wide")
st.title("Football Data Visualization App")

# --- Load Mapping Files ---
# This is done once to improve performance.

# Load Team Mapping (from WSL Matches.csv or similar)
try:
    team_mapping_df = pd.read_csv('WSL Matches.csv')
    home_teams = team_mapping_df[["matchInfo/contestant/0/id", "matchInfo/contestant/0/name"]].copy()
    home_teams.columns = ["contestantId", "Team"]
    away_teams = team_mapping_df[["matchInfo/contestant/1/id", "matchInfo/contestant/1/name"]].copy()
    away_teams.columns = ["contestantId", "Team"]
    team_map_full = pd.concat([home_teams, away_teams]).drop_duplicates().dropna()
    # Convert contestantId to string to match JSON data
    team_map_full['contestantId'] = team_map_full['contestantId'].astype(str)
    id_to_name_dict = dict(zip(team_map_full["contestantId"], team_map_full["Team"]))
except FileNotFoundError:
    st.error("Team mapping file ('WSL Matches.csv') not found.")
    id_to_name_dict = {}

# Load Event Mapping
try:
    event_map_df = pd.read_csv("event_mapping.csv", encoding="ISO-8859-1")
    event_map_df.columns = ["typeId", "Event Type", "Description"]
except FileNotFoundError:
    st.error("Event mapping file ('event_mapping.csv') not found.")
    event_map_df = pd.DataFrame(columns=["typeId", "Event Type"])

# --- Sidebar Menu ---
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ("Home", "Shot Map", "Flow Map", "Field Tilt", "Pass Network"))

if selected_page == "Home":
    st.header("Welcome!")
    st.write("Select a visualization from the navigation menu on the left.")

# ==============================================================================
# 3. SHOT MAP PAGE
# ==============================================================================
elif selected_page == "Shot Map":
    st.header("Expected Goals (xG) Shot Map")

    json_folder = 'json_data'  # Folder with your JSON match files
    json_files = sorted(
        [f for f in os.listdir(json_folder) if f.endswith('.json')],
        key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
        reverse=True
    )
    selected_file = st.selectbox("Select a Match JSON file", json_files)

    if selected_file:
        file_path = os.path.join(json_folder, selected_file)
        df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
        
        if df.empty or df['Team'].isnull().any():
            st.warning("Could not process the selected file. It might be empty or team IDs could not be mapped.")
        else:
            # Filter for only shot events (typeId 7=Failed attempt, 8=Save, 9=Miss, 10=Post, 16=Goal)
            shots_df = df[df['typeId'].isin([7, 8, 9, 10, 16])].copy()
            team_names = shots_df['Team'].unique()
            team1_name, team2_name = team_names[0], team_names[1]

            team1 = shots_df.loc[shots_df['Team'] == team1_name].reset_index()
            team2 = shots_df.loc[shots_df['Team'] == team2_name].reset_index()

            team1_goals = team1['isGoal'].sum()
            team2_goals = team2['isGoal'].sum()
            team1_xg = team1['xG'].sum()
            team2_xg = team2['xG'].sum()
            
            # --- Plotting ---
            pitch = Pitch(pitch_type='opta', pitch_width=68, pitch_length=105, pad_bottom=1.5, pad_top=5, pitch_color='white', line_color='black', half=False, goal_type='box', goal_alpha=0.8)
            fig, ax = plt.subplots(figsize=(16, 10))
            pitch.draw(ax=ax)
            fig.set_facecolor('white')
            plt.gca().invert_xaxis()
            
            title = f"{team1_name} ({team1_goals}) vs {team2_name} ({team2_goals})"
            plt.text(0.5, 1.05, title, ha='center', va='bottom', fontsize=25, fontweight='bold', transform=ax.transAxes)
            plt.text(80, 90, f"{team1_xg:.2f} xG", color='#ff6361', ha='center', fontsize=30, fontweight='bold')
            plt.text(20, 90, f"{team2_xg:.2f} xG", color='#003f5c', ha='center', fontsize=30, fontweight='bold')

            # Plot Team 1 Shots
            for i, shot in team1.iterrows():
                color = '#ffa600' if shot['isGoal'] else '#ff6361'
                plt.scatter(shot['x'], 100 - shot['y'], color=color, s=shot['xG'] * 800, alpha=0.9, zorder=3 if shot['isGoal'] else 2)

            # Plot Team 2 Shots
            for i, shot in team2.iterrows():
                color = '#ffa600' if shot['isGoal'] else '#003f5c'
                plt.scatter(100 - shot['x'], shot['y'], color=color, s=shot['xG'] * 800, alpha=0.9, zorder=3 if shot['isGoal'] else 2)
            
            st.pyplot(fig)


# ==============================================================================
# 4. FLOW MAP PAGE
# ==============================================================================
elif selected_page == "Flow Map":
    st.header("Expected Goals (xG) Flow Map")

    json_folder = 'json_data'
    json_files = sorted(
        [f for f in os.listdir(json_folder) if f.endswith('.json')],
        key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
        reverse=True
    )
    selected_file = st.selectbox("Select a Match JSON file", json_files)

    if selected_file:
        file_path = os.path.join(json_folder, selected_file)
        df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
        
        if df.empty or df['Team'].isnull().any():
            st.warning("Could not process the selected file.")
        else:
            shots_df = df[df['typeId'].isin([7, 8, 9, 10, 16])].copy()
            team_names = shots_df['Team'].unique()
            hteam, ateam = team_names[0], team_names[1]

            hteam_df = shots_df[shots_df['Team'] == hteam].sort_values('timeMin')
            ateam_df = shots_df[shots_df['Team'] == ateam].sort_values('timeMin')

            h_xG_cumulative = np.cumsum(hteam_df['xG'])
            a_xG_cumulative = np.cumsum(ateam_df['xG'])
            h_min = hteam_df['timeMin']
            a_min = ateam_df['timeMin']

            fig, ax = plt.subplots(figsize=(16, 10))
            fig.set_facecolor('white')
            ax.patch.set_facecolor('white')

            ax.step([0, *h_min], [0, *h_xG_cumulative], color='#ff6361', linewidth=5, where='post', label=hteam)
            ax.step([0, *a_min], [0, *a_xG_cumulative], color='#003f5c', linewidth=5, where='post', label=ateam)

            ax.fill_between([0, *h_min], [0, *h_xG_cumulative], color='#ff6361', alpha=0.3, step='post')
            ax.fill_between([0, *a_min], [0, *a_xG_cumulative], color='#003f5c', alpha=0.3, step='post')
            
            # Goal markers
            h_goals = hteam_df[hteam_df['isGoal']]
            a_goals = ateam_df[ateam_df['isGoal']]
            h_goal_cumulative_xg = np.cumsum(hteam_df['xG'])[h_goals.index.to_series().apply(lambda x: hteam_df.index.get_loc(x))]
            a_goal_cumulative_xg = np.cumsum(ateam_df['xG'])[a_goals.index.to_series().apply(lambda x: ateam_df.index.get_loc(x))]

            ax.scatter(h_goals['timeMin'], h_goal_cumulative_xg, color='#ffa600', marker='*', s=500, zorder=3)
            ax.scatter(a_goals['timeMin'], a_goal_cumulative_xg, color='#ffa600', marker='*', s=500, zorder=3)
            
            plt.xticks([0, 15, 30, 45, 60, 75, 90])
            plt.xlabel('Minute', fontsize=16)
            plt.ylabel('Cumulative xG', fontsize=16)
            plt.legend()
            st.pyplot(fig)

# ==============================================================================
# 5. FIELD TILT PAGE
# ==============================================================================
elif selected_page == "Field Tilt":
    st.header("Field Tilt Analysis")

    json_folder = 'json_data'
    json_files = sorted(
        [f for f in os.listdir(json_folder) if f.endswith('.json')],
        key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
        reverse=True
    )
    selected_match = st.selectbox("Select a Match JSON file", json_files)

    if selected_match:
        file_path = os.path.join(json_folder, selected_match)
        df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
        
        if df.empty or df['Team'].isnull().any():
            st.warning("Could not process the selected file.")
        else:
            team_names = df['Team'].unique()
            hteam_name, ateam_name = team_names[0], team_names[1]

            # Filter for passes (typeId 1) in the final third (x > 66.7)
            df_final_third = df.loc[(df['typeId'] == 1) & (df['x'] > 66.7)]
            
            minutes = list(range(0, 96))
            home_tilt, away_tilt = [], []
            
            for minute in minutes:
                min_df = df_final_third[df_final_third['timeMin'] == minute]
                home_passes = min_df[min_df['Team'] == hteam_name].shape[0]
                away_passes = min_df[min_df['Team'] == ateam_name].shape[0]
                total_passes = home_passes + away_passes
                
                if total_passes == 0:
                    home_tilt.append(50) # Neutral if no passes
                else:
                    home_tilt.append((home_passes / total_passes) * 100)

            # Use a rolling average to smooth the line
            tilt_series = pd.Series(home_tilt)
            smoothed_tilt = tilt_series.rolling(window=10, min_periods=1, center=True).mean()

            fig, ax = plt.subplots(figsize=(22, 12))
            ax.plot(minutes, smoothed_tilt, color='#ff6361', lw=3)
            ax.plot(minutes, 100 - smoothed_tilt, color='#003f5c', lw=3)
            ax.axhline(50, color='grey', linestyle='--', lw=2)

            ax.fill_between(minutes, 50, smoothed_tilt, where=smoothed_tilt > 50, interpolate=True, color='#ff6361', alpha=0.6, label=f'{hteam_name} Dominance')
            ax.fill_between(minutes, 50, smoothed_tilt, where=smoothed_tilt < 50, interpolate=True, color='#003f5c', alpha=0.6, label=f'{ateam_name} Dominance')

            plt.ylim(0, 100)
            plt.xlim(0, 95)
            plt.ylabel('Field Tilt (%)', fontsize=18)
            plt.xlabel('Minute', fontsize=18)
            plt.legend()
            st.pyplot(fig)

# ==============================================================================
# 6. PASS NETWORK PAGE
# ==============================================================================
elif selected_page == "Pass Network":
    st.header("Pass Network Visualization")

    json_folder = 'json_data'
    json_files = sorted(
        [f for f in os.listdir(json_folder) if f.endswith('.json')],
        key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
        reverse=True
    )
    selected_match = st.selectbox("Select a match", json_files, key="pass_network_match")

    if selected_match:
        file_path = os.path.join(json_folder, selected_match)
        df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
        
        if not df.empty and not df['Team'].isnull().any():
            team_names = df['Team'].unique().tolist()
            selected_team = st.selectbox("Select a team", team_names, key="pass_network_team")

            if selected_team:
                team_data = df.loc[df['Team'] == selected_team].copy()
                team_data['recipient'] = team_data['playerName'].shift(-1)
                
                # Filter for successful passes only
                passes = team_data.loc[(team_data['typeId'] == 1) & (team_data['outcome'] == 1)]
                
                if passes.empty:
                    st.warning(f"No completed passes found for {selected_team}.")
                else:
                    avg_locs = passes.groupby('playerName').agg({'x': ['mean'], 'y': ['mean']})
                    avg_locs.columns = ['x', 'y']

                    passes_between = passes.groupby(['playerName', 'recipient']).id.count().reset_index()
                    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
                    
                    passes_between = passes_between.merge(avg_locs, left_on='playerName', right_index=True)
                    passes_between = passes_between.merge(avg_locs, left_on='recipient', right_index=True, suffixes=('', '_end'))
                    
                    pitch = VerticalPitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')
                    fig, ax = plt.subplots(figsize=(16, 11))
                    pitch.draw(ax=ax)

                    # Plot edges
                    for _, row in passes_between.iterrows():
                        if row['pass_count'] > 2: # Filter for significant connections
                            pitch.lines(row.x, row.y, row.x_end, row.y_end,
                                        alpha=1, lw=row.pass_count/2, color='white', ax=ax, zorder=1)

                    # Plot nodes
                    pitch.scatter(avg_locs.x, avg_locs.y, s=600, color='#d3d3d3', edgecolors='black', ax=ax, zorder=2)
                    for i, row in avg_locs.iterrows():
                        pitch.annotate(i.split()[-1], xy=(row.x, row.y), c='black', va='center', ha='center', size=12, ax=ax, zorder=3)
                    
                    st.pyplot(fig)