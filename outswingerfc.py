import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

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
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def calculate_derived_metrics(df):
    """Calculates per 90, per shot, and other derived metrics."""
    df = df.copy()
    if 'Minutes Played' not in df.columns or 'Shots' not in df.columns:
        return df

    df['Minutes Played'] = pd.to_numeric(df['Minutes Played'], errors='coerce').replace(0, np.nan)
    df['Shots'] = pd.to_numeric(df['Shots'], errors='coerce').replace(0, np.nan)

    for col in ['xG', 'xAG', 'xT', 'xDisruption', 'GPA']:
        if col in df.columns:
            df[f'{col} per 90'] = (df[col] / df['Minutes Played']) * 90

    if 'xG' in df.columns and 'Shots' in df.columns:
        df['xG per Shot'] = df['xG'] / df['Shots']
        
    return df

def load_and_process_data(selected_league, selected_metric_key):
    """Loads, processes, and standardizes data based on user selection."""
    data_config = {
        "WSL": {
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}
        },
        "WSL 2": {
            'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL2_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}
        },
        "Frauen-Bundesliga": {
            'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['Player', 'Team', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "FBL_gpa.csv", "cols": ['Player', 'Team', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}
        }
    }
    
    metric_config = data_config.get(selected_league, {}).get(selected_metric_key)

    if not metric_config:
        st.warning("No data configuration found for the selected league and metric.")
        return pd.DataFrame(), [], ''

    try:
        file_path = resource_path(os.path.join("data", metric_config["file"]))
        df_raw = pd.read_csv(file_path)

        # Standardize column names for consistency
        rename_map = {
            'playerName': 'Player',
            'ActualDisruptions': 'Actual disruption',
            'ExpectedDisruptions': 'expected disruptions'
        }
        df_raw.rename(columns=rename_map, inplace=True)

        df_processed = calculate_derived_metrics(df_raw)
        cols_to_show = metric_config["cols"]
        sort_by_col = metric_config["sort"]
        
        # Dynamically add 'per 90' metrics if available
        for base_col in ['xAG', 'xT']:
            per_90_col = f'{base_col} per 90'
            if per_90_col in df_processed.columns and base_col in cols_to_show:
                cols_to_show.append(per_90_col)
        
        return df_processed, cols_to_show, sort_by_col

    except FileNotFoundError:
        st.error(f"Error: The data file `{metric_config['file']}` was not found in the 'data' directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}.")
    
    return pd.DataFrame(), [], ''

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Outswinger FC | Women's Football Analytics", layout="wide", initial_sidebar_state="expanded")

    metric_info = get_metric_info()
    metric_pages = list(metric_info.keys())

    # --- Initialize Session State ---
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = metric_pages[0]

    # --- Sidebar ---
    st.sidebar.title("Outswinger FC")
    st.sidebar.image("https://placehold.co/400x200/2d3748/e2e8f0?text=Outswinger+FC", use_container_width=True)
    st.sidebar.info("This app displays player stats for the WSL, WSL 2, and Frauen-Bundesliga.")
    st.sidebar.header("Metric Leaderboards")

    for metric in metric_pages:
        if st.sidebar.button(metric, use_container_width=True, disabled=(st.session_state.selected_metric == metric)):
            st.session_state.selected_metric = metric
            st.rerun()

    # --- Main Page ---
    st.title(f"📊 {st.session_state.selected_league} Advanced Metrics Leaderboard")

    leagues = ["WSL", "WSL 2", "Frauen-Bundesliga"]
    cols = st.columns(len(leagues))
    for i, league in enumerate(leagues):
        if cols[i].button(league, use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    selected_league = st.session_state.selected_league
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info.get(selected_metric_key, '')}")
    
    if selected_league == "Frauen-Bundesliga":
        st.info("Note: Data of FC Köln - RB Leipzig is not present as of 06-09-2025")

    with st.spinner("Loading data..."):
        df_processed, cols_to_show, sort_by_col = load_and_process_data(selected_league, selected_metric_key)

        if not df_processed.empty and sort_by_col and sort_by_col in df_processed.columns:
            existing_cols = [col for col in cols_to_show if col in df_processed.columns]
            if not existing_cols:
                 st.warning("None of the required columns for this metric were found in the data file.")
            else:
                display_df = df_processed[existing_cols]
                display_df = display_df.sort_values(by=sort_by_col, ascending=False).reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, use_container_width=True)
        elif not df_processed.empty:
            st.warning(f"The required sorting metric '{sort_by_col}' is not available in the loaded data file.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"© {datetime.now().year} Outswinger FC | All rights reserved.")

if __name__ == "__main__":
    main()

