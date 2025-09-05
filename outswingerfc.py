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
# 1. HELPER FUNCTION FOR LOADING AND PROCESSING JSON DATA (for Field Tilt & Pass Network)
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
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

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

st.set_page_config(page_title="Outswinger FC - Data visualisation app", layout="wide")
st.title("Football Data Visualization App")

# --- Load Mapping Files for JSON Processing ---
try:
    team_mapping_df = pd.read_csv('WSL Matches.csv')
    home_teams = team_mapping_df[["matchInfo/contestant/0/id", "matchInfo/contestant/0/name"]].copy()
    home_teams.columns = ["contestantId", "Team"]
    away_teams = team_mapping_df[["matchInfo/contestant/1/id", "matchInfo/contestant/1/name"]].copy()
    away_teams.columns = ["contestantId", "Team"]
    team_map_full = pd.concat([home_teams, away_teams]).drop_duplicates().dropna()
    team_map_full['contestantId'] = team_map_full['contestantId'].astype(str)
    id_to_name_dict = dict(zip(team_map_full["contestantId"], team_map_full["Team"]))
except FileNotFoundError:
    st.sidebar.error("Mapping file 'WSL Matches.csv' not found. Needed for Field Tilt & Pass Network.")
    id_to_name_dict = {}

try:
    event_map_df = pd.read_csv("event_mapping.csv", encoding="ISO-8859-1")
    event_map_df.columns = ["typeId", "Event Type", "Description"]
except FileNotFoundError:
    st.sidebar.error("Mapping file 'event_mapping.csv' not found. Needed for Field Tilt & Pass Network.")
    event_map_df = pd.DataFrame(columns=["typeId", "Event Type"])

# --- Sidebar Menu ---
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ("Home", "Shot Map", "Flow Map", "Field Tilt", "Pass Network"))

if selected_page == "Home":
    st.header("Welcome!")
    st.write("Select a visualization from the navigation menu on the left.")

# ==============================================================================
# 3. SHOT MAP PAGE (CSV)
# ==============================================================================
elif selected_page == "Shot Map":
    st.header("Expected Goals (xG) Shot Map")

    xg_csv_folder = 'xgCSV'  # Folder with your xG CSV files
    try:
        csv_files = sorted(
            [f for f in os.listdir(xg_csv_folder) if f.endswith('.csv')],
            key=lambda f: os.path.getmtime(os.path.join(xg_csv_folder, f)),
            reverse=True
        )
        if not csv_files:
            st.warning(f"No CSV files found in the '{xg_csv_folder}' folder.")
        else:
            selected_file = st.selectbox("Select a Match CSV file", csv_files)

            if selected_file:
                file_path = os.path.join(xg_csv_folder, selected_file)
                df = pd.read_csv(file_path)
                
                # --- Original CSV processing logic ---
                teams = selected_file.split('_')[-1].split(' - ')
                team1_name = teams[0]
                team2_name = teams[1].split('.')[0]

                team1 = df.loc[df['TeamId'] == team1_name].reset_index()
                team2 = df.loc[df['TeamId'] == team2_name].reset_index()

                team1_goals = int(team1['isGoal'].sum())
                team2_goals = int(team2['isGoal'].sum())
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
                for _, shot in team1.iterrows():
                    color = '#ffa600' if shot['isGoal'] else '#ff6361'
                    plt.scatter(shot['x'], 100 - shot['y'], color=color, s=shot['xG'] * 800, alpha=0.9, zorder=3 if shot['isGoal'] else 2)

                # Plot Team 2 Shots
                for _, shot in team2.iterrows():
                    color = '#ffa600' if shot['isGoal'] else '#003f5c'
                    plt.scatter(100 - shot['x'], shot['y'], color=color, s=shot['xG'] * 800, alpha=0.9, zorder=3 if shot['isGoal'] else 2)
                
                st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"The specified folder for CSV files ('{xg_csv_folder}') was not found.")

# ==============================================================================
# 4. FLOW MAP PAGE (CSV)
# ==============================================================================
elif selected_page == "Flow Map":
    st.header("Expected Goals (xG) Flow Map")

    xg_csv_folder = 'xgCSV' # Folder with your xG CSV files
    try:
        csv_files = sorted(
            [f for f in os.listdir(xg_csv_folder) if f.endswith('.csv')],
            key=lambda f: os.path.getmtime(os.path.join(xg_csv_folder, f)),
            reverse=True
        )
        if not csv_files:
            st.warning(f"No CSV files found in the '{xg_csv_folder}' folder.")
        else:
            selected_file = st.selectbox("Select a Match CSV file", csv_files)

            if selected_file:
                file_path = os.path.join(xg_csv_folder, selected_file)
                df = pd.read_csv(file_path)

                # --- Original CSV processing logic ---
                teams = selected_file.split('_')[-1].split(' - ')
                hteam = teams[0]
                ateam = teams[1].split('.')[0]

                hteam_df = df[df['TeamId'] == hteam].sort_values('timeMin')
                ateam_df = df[df['TeamId'] == ateam].sort_values('timeMin')

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

                h_goals = hteam_df[hteam_df['isGoal'] == True]
                a_goals = ateam_df[ateam_df['isGoal'] == True]

                if not h_goals.empty:
                    h_goal_indices = h_goals.index.to_series().apply(lambda x: hteam_df.index.get_loc(x))
                    h_goal_cumulative_xg = np.cumsum(hteam_df['xG']).iloc[h_goal_indices]
                    ax.scatter(h_goals['timeMin'], h_goal_cumulative_xg, color='#ffa600', marker='*', s=500, zorder=3)
                if not a_goals.empty:
                    a_goal_indices = a_goals.index.to_series().apply(lambda x: ateam_df.index.get_loc(x))
                    a_goal_cumulative_xg = np.cumsum(ateam_df['xG']).iloc[a_goal_indices]
                    ax.scatter(a_goals['timeMin'], a_goal_cumulative_xg, color='#ffa600', marker='*', s=500, zorder=3)
                
                plt.xticks([0, 15, 30, 45, 60, 75, 90])
                plt.xlabel('Minute', fontsize=16)
                plt.ylabel('Cumulative xG', fontsize=16)
                plt.legend()
                st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"The specified folder for CSV files ('{xg_csv_folder}') was not found.")

# ==============================================================================
# 5. FIELD TILT PAGE (JSON)
# ==============================================================================
elif selected_page == "Field Tilt":
    st.header("Field Tilt Analysis")

    json_folder = 'WSL 2024-2025'
    try:
        json_files = sorted(
            [f for f in os.listdir(json_folder) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
            reverse=True
        )
        if not json_files:
            st.warning(f"No JSON files found in the '{json_folder}' folder.")
        else:
            selected_match = st.selectbox("Select a Match JSON file", json_files)
            
            if selected_match:
                file_path = os.path.join(json_folder, selected_match)
                df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
                
                if df.empty or df['Team'].isnull().any():
                    st.warning("Could not process the selected file.")
                else:
                    team_names = df['Team'].dropna().unique()
                    if len(team_names) < 2:
                         st.warning("Could not identify two distinct teams in the data.")
                    else:
                        hteam_name, ateam_name = team_names[0], team_names[1]

                        df_final_third = df.loc[(df['typeId'] == 1) & (df['x'] > 66.7)]
                        
                        minutes = list(range(0, int(df['timeMin'].max()) + 2))
                        home_tilt = []
                        
                        for minute in minutes:
                            min_df = df_final_third[df_final_third['timeMin'] == minute]
                            home_passes = min_df[min_df['Team'] == hteam_name].shape[0]
                            away_passes = min_df[min_df['Team'] == ateam_name].shape[0]
                            total_passes = home_passes + away_passes
                            
                            if total_passes == 0:
                                home_tilt.append(50) # Neutral if no passes
                            else:
                                home_tilt.append((home_passes / total_passes) * 100)

                        tilt_series = pd.Series(home_tilt)
                        smoothed_tilt = tilt_series.rolling(window=10, min_periods=1, center=True).mean()

                        fig, ax = plt.subplots(figsize=(22, 12))
                        ax.plot(minutes, smoothed_tilt, color='#ff6361', lw=3)
                        ax.plot(minutes, 100 - smoothed_tilt, color='#003f5c', lw=3)
                        ax.axhline(50, color='grey', linestyle='--', lw=2)

                        ax.fill_between(minutes, 50, smoothed_tilt, where=smoothed_tilt > 50, interpolate=True, color='#ff6361', alpha=0.6, label=f'{hteam_name} Dominance')
                        ax.fill_between(minutes, 50, smoothed_tilt, where=smoothed_tilt < 50, interpolate=True, color='#003f5c', alpha=0.6, label=f'{ateam_name} Dominance')

                        plt.ylim(0, 100)
                        plt.xlim(0, max(minutes) -1)
                        plt.ylabel('Field Tilt (%)', fontsize=18)
                        plt.xlabel('Minute', fontsize=18)
                        plt.legend()
                        st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"The specified folder for JSON files ('{json_folder}') was not found.")

# ==============================================================================
# 6. PASS NETWORK PAGE (JSON)
# ==============================================================================
elif selected_page == "Pass Network":
    st.header("Pass Network Visualization")

    json_folder = 'WSL 2024-2025'
    try:
        json_files = sorted(
            [f for f in os.listdir(json_folder) if f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(json_folder, f)),
            reverse=True
        )
        if not json_files:
            st.warning(f"No JSON files found in the '{json_folder}' folder.")
        else:
            selected_match = st.selectbox("Select a match", json_files, key="pass_network_match")

            if selected_match:
                file_path = os.path.join(json_folder, selected_match)
                df = load_and_process_json(file_path, id_to_name_dict, event_map_df)
                
                if not df.empty and not df['Team'].isnull().any():
                    team_names = df['Team'].dropna().unique().tolist()
                    selected_team = st.selectbox("Select a team", team_names, key="pass_network_team")

                    if selected_team:
                        team_data = df.loc[df['Team'] == selected_team].copy()
                        team_data['recipient'] = team_data['playerName'].shift(-1)
                        
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

                            for _, row in passes_between.iterrows():
                                if row['pass_count'] > 2:
                                    pitch.lines(row.x, row.y, row.x_end, row.y_end,
                                                alpha=1, lw=row.pass_count/2, color='white', ax=ax, zorder=1)

                            pitch.scatter(avg_locs.x, avg_locs.y, s=600, color='#d3d3d3', edgecolors='black', ax=ax, zorder=2)
                            for i, row in avg_locs.iterrows():
                                pitch.annotate(i.split()[-1], xy=(row.x, row.y), c='black', va='center', ha='center', size=12, ax=ax, zorder=3)
                            
                            st.pyplot(fig)
    except FileNotFoundError:
        st.error(f"The specified folder for JSON files ('{json_folder}') was not found.")