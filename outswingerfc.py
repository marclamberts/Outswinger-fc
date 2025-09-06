import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

def get_metric_info():
    """Returns a dictionary of metric explanations."""
    return {
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and type of assist. A higher xG suggests a player is getting into high-quality scoring positions.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist. It credits creative players for setting up scoring chances, even if the shot is missed.',
        'xT (Expected Threat)': 'Quantifies the increase in the probability of scoring a goal by moving the ball between two points on the pitch. It rewards players for advancing the ball into dangerous areas.',
        'Expected Disruption (xDisruption)': 'Measures a defensive player\'s ability to break up opposition plays. It values tackles and interceptions that prevent high-probability scoring chances for the opponent.'
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
    # Ensure a copy is made to avoid SettingWithCopyWarning
    df = df.copy()

    # Check if essential columns exist before calculations. If not, return original df.
    if 'Minutes Played' not in df.columns or 'Shots' not in df.columns:
        return df

    # Avoid division by zero
    df['Minutes Played'] = df['Minutes Played'].replace(0, np.nan)
    df['Shots'] = df['Shots'].replace(0, np.nan)

    # Calculate per 90 metrics for the remaining core metrics
    for col in ['xG', 'xAG', 'xT', 'xDisruption']:
        if col in df.columns:
            df.loc[:, f'{col} per 90'] = (df[col] / df['Minutes Played']) * 90

    # Calculate xG per Shot
    if 'xG' in df.columns and 'Shots' in df.columns:
        df.loc[:, 'xG per Shot'] = df['xG'] / df['Shots']
        
    return df

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Soccer Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

    metric_info = get_metric_info()
    metric_pages = list(metric_info.keys())

    # --- Initialize Session State ---
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = metric_pages[0] # Default to the first metric

    # --- Sidebar Navigation ---
    st.sidebar.title("Women's Footy Data")
    st.sidebar.image("https://placehold.co/400x200/2d3748/ffffff?text=SOCCER+ANALYSIS", use_container_width=True)
    
    st.sidebar.info(
        """
        This app displays player stats for the WSL, WSL 2, and Frauen-Bundesliga.
        """
    )

    st.sidebar.header("Metric Leaderboards")
    for metric in metric_pages:
        if st.sidebar.button(metric, use_container_width=True):
            st.session_state.selected_metric = metric
            st.rerun()

    # --- Main Page ---
    st.title(f"📊 {st.session_state.selected_league} Advanced Metrics Leaderboard")

    # --- League Selection Buttons ---
    leagues = ["WSL", "WSL 2", "Frauen-Bundesliga"]
    cols = st.columns(len(leagues))
    for i, league in enumerate(leagues):
        if cols[i].button(league, use_container_width=True):
            st.session_state.selected_league = league
            st.rerun()

    # --- Display Selected Metric Page ---
    selected_league = st.session_state.selected_league
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info.get(selected_metric_key, '')}")

    # --- Data Configuration ---
    data_config = {
        "WSL": {
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['playerName', 'Team', 'ActualDisruptions', 'ExpectedDisruptions'], "sort": 'ExpectedDisruptions'}
        },
        "WSL 2": {
            'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['playerName', 'Team', 'ActualDisruptions', 'ExpectedDisruptions'], "sort": 'ExpectedDisruptions'}
        },
        "Frauen-Bundesliga": {
            'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['playerName', 'Team', 'ActualDisruptions', 'ExpectedDisruptions'], "sort": 'ExpectedDisruptions'}
        }
    }

    df_processed = pd.DataFrame()
    sort_by_col = ''
    cols_to_show = []

    # --- Data Loading and Processing ---
    metric_config = data_config.get(selected_league, {}).get(selected_metric_key)

    if metric_config:
        local_csv_path = resource_path(os.path.join("data", metric_config["file"]))
        try:
            df_raw = pd.read_csv(local_csv_path)
            df_processed = calculate_derived_metrics(df_raw)
            cols_to_show = metric_config["cols"]
            sort_by_col = metric_config["sort"]

            # Add per 90 columns if they exist after processing
            for col in ['xAG per 90', 'xT per 90', 'xDisruption per 90']:
                 if col in df_processed.columns and col.split(' ')[0] in cols_to_show:
                    cols_to_show.append(col)

        except FileNotFoundError:
            st.error(f"Error: The file `{local_csv_path}` was not found.")
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}.")
    else:
        st.warning("No data configuration found for the selected league and metric.")

    # --- Filter for necessary columns, sort, and display ---
    if not df_processed.empty and sort_by_col in df_processed.columns:
        existing_cols = [col for col in cols_to_show if col in df_processed.columns]
        if not existing_cols:
             st.warning(f"None of the required columns for this metric are in the data file.")
        else:
            display_df = df_processed[existing_cols]
            display_df = display_df.sort_values(by=sort_by_col, ascending=False).reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, use_container_width=True)
    elif not df_processed.empty:
        st.warning(f"The metric '{sort_by_col}' is not available in the loaded data file.")
    else:
        st.info("No data to display. This might be because the corresponding data file is missing.")


if __name__ == "__main__":
    main()

