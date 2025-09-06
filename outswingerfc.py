import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

# --- Configuration & Setup ---

st.set_page_config(
    page_title="Outswinger FC | Women's Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching ---
@st.cache_data(ttl=3600)
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

# --- Helper Functions ---

def get_metric_info():
    """Returns a dictionary of metric explanations."""
    return {
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and type of assist. A higher xG suggests a player is getting into high-quality scoring positions.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist. It credits creative players for setting up scoring chances, even if the shot is missed.',
        'xT (Expected Threat)': 'Quantifies the increase in the probability of scoring a goal by moving the ball between two points on the pitch. It rewards players for advancing the ball into dangerous areas.',
        'Expected Disruption (xDisruption)': 'Measures a defensive player\'s ability to break up opposition plays. It values tackles and interceptions that prevent high-probability scoring chances for the opponent.',
        'Goal Probability Added (GPA/G+)': 'Measures the change in goal probability from a player\'s actions on the ball. A positive GPA indicates that the player\'s actions increased the team\'s chances of scoring.'
    }

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def calculate_derived_metrics(df):
    """Calculates per 90 and per shot metrics if applicable."""
    df = df.copy()
    if 'Minutes Played' in df.columns and 'Shots' in df.columns:
        df['Minutes Played'] = pd.to_numeric(df['Minutes Played'], errors='coerce').replace(0, np.nan)
        df['Shots'] = pd.to_numeric(df['Shots'], errors='coerce').replace(0, np.nan)
        
        for col in ['xG', 'xAG', 'xT', 'xDisruption', 'GPA']:
            if col in df.columns:
                df[f'{col} per 90'] = (df[col] / df['Minutes Played'] * 90).round(2)
        
        if 'xG' in df.columns:
            df['xG per Shot'] = (df['xG'] / df['Shots']).round(2)
    return df

# --- Page Display Functions ---

def display_metrics_page(data_config, metric_info):
    """Renders the main metrics leaderboard page."""
    # --- League Selection ---
    leagues_row1 = ["WSL", "WSL 2", "Frauen-Bundesliga"]
    leagues_row2 = ["Liga F", "NWSL"]
    
    cols_row1 = st.columns(len(leagues_row1))
    for i, league in enumerate(leagues_row1):
        if cols_row1[i].button(league, use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    cols_row2 = st.columns(len(leagues_row2) + 1)
    for i, league in enumerate(leagues_row2):
        if cols_row2[i].button(league, use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    selected_league = st.session_state.selected_league
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info.get(selected_metric_key, '')}")
    
    if selected_league == "Frauen-Bundesliga":
        st.info("Note: Data of FC Köln - RB Leipzig is not present as of 06-09-2025")

    metric_config = data_config.get(selected_league, {}).get(selected_metric_key)

    if metric_config:
        try:
            file_path = resource_path(os.path.join("data", metric_config["file"]))
            df_raw = load_data(file_path)

            rename_map = {'playerName': 'Player', 'ActualDisruptions': 'Actual disruption', 'ExpectedDisruptions': 'expected disruptions'}
            df_raw.rename(columns=rename_map, inplace=True)
            
            df_processed = calculate_derived_metrics(df_raw)
            sort_by_col = metric_config["sort"]
            
            search_placeholders = {"WSL": "e.g., Sam Kerr", "WSL 2": "e.g., Melissa Johnson", "Frauen-Bundesliga": "e.g., Alexandra Popp", "Liga F": "e.g., Alexia Putellas", "NWSL": "e.g., Sophia Smith"}
            placeholder = search_placeholders.get(selected_league, "Search for a player...")

            st.markdown("---")
            col1, col2 = st.columns([2, 1.5])
            with col1:
                search_term = st.text_input("Search for a player:", placeholder=placeholder)
            with col2:
                top_n = st.slider("Number of players to display:", 5, 50, 15, 5)
            
            display_option = st.radio("Display format:", ("📄 Data Table", "📊 Visualization"), horizontal=True, label_visibility="collapsed")

            if search_term:
                df_processed = df_processed[df_processed['Player'].str.contains(search_term, case=False, na=False)]

            if not df_processed.empty and sort_by_col in df_processed.columns:
                display_df = df_processed.sort_values(by=sort_by_col, ascending=False).head(top_n).reset_index(drop=True)
                display_df.index = display_df.index + 1

                if display_option == "📊 Visualization":
                    st.subheader("Top Performers Chart")
                    max_val = display_df[sort_by_col].max()
                    x_domain = [0, max_val]
                    chart = alt.Chart(display_df).mark_bar().encode(x=alt.X(f'{sort_by_col}:Q', title=selected_metric_key, scale=alt.Scale(domain=x_domain)), y=alt.Y('Player:N', sort='-x', title="Player")).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.subheader("Detailed Data Table")
                    existing_cols = [col for col in metric_config["cols"] if col in display_df.columns]
                    st.dataframe(display_df[existing_cols], use_container_width=True)
            elif not df_processed.empty:
                st.warning(f"The metric '{sort_by_col}' is not available in the loaded data file.")
            else:
                 st.info("No matching players found.")
        except FileNotFoundError:
            st.error(f"Error: The data file `{metric_config['file']}` was not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("No data configuration found.")

def display_corners_page(data_config):
    """Renders the corner analysis page."""
    leagues = list(data_config.keys())
    leagues_with_total = ["Total"] + leagues
    
    selected_league_corners = st.selectbox("Select a league to analyze:", leagues_with_total)
    
    st.header(f"Corner Analysis for {selected_league_corners}")
    
    df_full = pd.DataFrame()
    
    try:
        if selected_league_corners == "Total":
            all_dfs = []
            for league in leagues:
                corner_config = data_config.get(league, {}).get('Corners')
                if corner_config:
                    try:
                        file_path = resource_path(os.path.join("data", corner_config["file"]))
                        df_league = load_data(file_path)
                        all_dfs.append(df_league)
                    except FileNotFoundError:
                        st.warning(f"Corner data for {league} not found. Skipping.")
            if all_dfs:
                df_full = pd.concat(all_dfs, ignore_index=True)
        else:
            corner_config = data_config.get(selected_league_corners, {}).get('Corners')
            if corner_config:
                file_path = resource_path(os.path.join("data", corner_config["file"]))
                df_full = load_data(file_path)
            else:
                st.error(f"Corner data configuration not found for {selected_league_corners}.")
                return

    except FileNotFoundError:
        st.error(f"Error: The specified corner data file was not found.")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    if df_full.empty:
        st.warning("No corner data could be loaded for the selected league(s).")
        return

    # Filter for corners just in case the source file contains other data types
    df_corners = df_full[df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
    if df_corners.empty:
        st.warning("No events of type 'FromCorner' found in the dataset.")
        return

    st.sidebar.header("Corner Filters")
    teams = ["All"] + sorted(df_corners['TeamId'].unique().tolist())
    players = ["All"] + sorted(df_corners['PlayerId'].unique().tolist())
    is_goal_options = ["All", True, False]
    min_time, max_time = int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max())

    selected_team = st.sidebar.selectbox("Filter by Team:", teams)
    selected_player = st.sidebar.selectbox("Filter by Player:", players)

    # Safely add GameState filter only if the column exists
    if 'GameState' in df_corners.columns:
        game_states = ["All"] + sorted(df_corners['GameState'].unique().tolist())
        selected_state = st.sidebar.selectbox("Filter by Game State:", game_states)
    else:
        selected_state = "All" # Default to "All" if column is missing

    selected_goal = st.sidebar.selectbox("Filter by Goal:", is_goal_options, format_func=lambda x: "All" if x=="All" else ("Yes" if x else "No"))
    selected_time = st.sidebar.slider("Filter by Time (minutes):", min_time, max_time, (min_time, max_time))

    df_filtered = df_corners.copy()
    if selected_team != "All": df_filtered = df_filtered[df_filtered['TeamId'] == selected_team]
    if selected_player != "All": df_filtered = df_filtered[df_filtered['PlayerId'] == selected_player]
    if selected_state != "All" and 'GameState' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['GameState'] == selected_state]
    if selected_goal != "All": df_filtered = df_filtered[df_filtered['isGoal'] == selected_goal]
    df_filtered = df_filtered[df_filtered['timeMin'].between(selected_time[0], selected_time[1])]

    st.markdown("---")
    st.write(f"#### Displaying `{len(df_filtered)}` corner events based on filters.")

    if not df_filtered.empty and all(c in df_filtered.columns for c in ['x', 'y', 'xG']):
        pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(8, 6))
        fig.set_facecolor('#22312b')
        
        sizes = df_filtered['xG'] * 900
        pitch.scatter(df_filtered.x, df_filtered.y, s=sizes, ax=ax, alpha=0.7, ec='white', c='#e789e7')
        st.pyplot(fig)
    elif not df_filtered.empty:
        st.warning("Required columns ('X', 'Y', 'xG') not found for plotting.")
    else:
        st.info("No data available for the selected filters.")


# --- Main App Logic ---
def main():
    metric_info = get_metric_info()
    metric_pages = list(metric_info.keys())

    if 'selected_league' not in st.session_state: st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state: st.session_state.selected_metric = metric_pages[0]

    st.sidebar.title("Outswinger FC")
    st.sidebar.image("https://placehold.co/400x200/2d3748/e2e8f0?text=Outswinger+FC", use_container_width=True)
    
    page_view = st.sidebar.radio("Select a view:", ["Metrics Leaderboard", "Corners Analysis"])
    st.sidebar.markdown("---")

    data_config = {
        "WSL": {
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "WSL_corners.csv"}
        },
        "WSL 2": {
            'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL2_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "WSL2_corners.csv"}
        },
        "Frauen-Bundesliga": {
            'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "FBL_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "FBL_corners.csv"}
        },
        "Liga F": {
            'xG (Expected Goals)': {"file": "LigaF.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "LigaF_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "LigaF_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "LigaF_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "LigaF_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "LigaF_corners.csv"}
        },
        "NWSL": {
            'xG (Expected Goals)': {"file": "NWSL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "NWSL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "NWSL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "NWSL_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "NWSL_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "NWSL_corners.csv"}
        }
    }

    if page_view == "Metrics Leaderboard":
        st.title(f"📊 {st.session_state.selected_league} Advanced Metrics Leaderboard")
        st.sidebar.header("Metric Leaderboards")
        for metric in metric_pages:
            if st.sidebar.button(metric, use_container_width=True, disabled=(st.session_state.selected_metric == metric)):
                st.session_state.selected_metric = metric
                st.rerun()
        display_metrics_page(data_config, metric_info)
    
    elif page_view == "Corners Analysis":
        st.title("⛳️ Corner Analysis")
        display_corners_page(data_config)

    st.markdown("---")
    st.markdown(f"© {datetime.now().year} Outswinger FC | All rights reserved.")

if __name__ == "__main__":
    main()

