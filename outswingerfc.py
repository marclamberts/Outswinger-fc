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

    # Check if essential columns exist before calculations
    if 'Minutes Played' not in df.columns or 'Shots' not in df.columns:
        st.warning("'Minutes Played' and/or 'Shots' columns not found in the data. Cannot calculate per 90 or per shot metrics.")
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
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = metric_pages[0] # Default to the first metric

    # --- Sidebar Navigation ---
    st.sidebar.title("Women's Footy Data")
    st.sidebar.image("https://placehold.co/400x200/2d3748/ffffff?text=SOCCER+ANALYSIS", use_container_width=True)
    
    st.sidebar.info(
        """
        This app displays player stats for the WSL.
        """
    )

    st.sidebar.header("Metric Leaderboards")
    for metric in metric_pages:
        if st.sidebar.button(metric, use_container_width=True):
            st.session_state.selected_metric = metric
            # No rerun needed here, button click handles it

    # --- Main Page ---
    st.title("📊 WSL Advanced Metrics Leaderboard")

    # --- Display Selected Metric Page ---
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 WSL - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info[selected_metric_key]}")

    df_processed = pd.DataFrame() # Initialize an empty dataframe
    sort_by_col = '' # Initialize sort_by_col to avoid reference before assignment error

    # --- Data Loading and Logic based on selected metric ---
    if selected_metric_key == 'xG (Expected Goals)':
        local_csv_path = resource_path(os.path.join("data", "WSL.csv"))
        try:
            df_raw = pd.read_csv(local_csv_path)
            st.success(f"Successfully loaded data from `{local_csv_path}`.")
            df_processed = calculate_derived_metrics(df_raw)
        except FileNotFoundError:
            st.error(f"Error: The file `{local_csv_path}` was not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")
            
        cols_to_show = ['Player', 'Team', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG']
        sort_by_col = 'xG'

    elif selected_metric_key == 'xAG (Expected Assisted Goals)':
        local_csv_path = resource_path(os.path.join("data", "WSL_assists.csv"))
        try:
            df_raw = pd.read_csv(local_csv_path)
            st.success(f"Successfully loaded data from `{local_csv_path}`.")
            df_processed = calculate_derived_metrics(df_raw) # Process for per 90 stats
        except FileNotFoundError:
            st.error(f"Error: The file `{local_csv_path}` was not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")

        base_metric_name = 'xAG'
        cols_to_show = ['Player', 'Team', 'Assists', 'ShotAssists', base_metric_name]
        if f'{base_metric_name} per 90' in df_processed.columns:
            cols_to_show.append(f'{base_metric_name} per 90')
        sort_by_col = base_metric_name
        
    elif selected_metric_key == 'xT (Expected Threat)':
        local_csv_path = resource_path(os.path.join("data", "WSL_xt.csv"))
        try:
            df_raw = pd.read_csv(local_csv_path)
            st.success(f"Successfully loaded data from `{local_csv_path}`.")
            df_processed = calculate_derived_metrics(df_raw) # Process for per 90 stats
        except FileNotFoundError:
            st.error(f"Error: The file `{local_csv_path}` was not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")

        base_metric_name = 'xT'
        cols_to_show = ['Player', 'Team', base_metric_name]
        if f'{base_metric_name} per 90' in df_processed.columns:
            cols_to_show.append(f'{base_metric_name} per 90')
        sort_by_col = base_metric_name

    elif selected_metric_key == 'Expected Disruption (xDisruption)':
        local_csv_path = resource_path(os.path.join("data", "WSL_xDisruption.csv"))
        try:
            df_raw = pd.read_csv(local_csv_path)
            st.success(f"Successfully loaded data from `{local_csv_path}`.")
            df_processed = calculate_derived_metrics(df_raw)
        except FileNotFoundError:
            st.error(f"Error: The file `{local_csv_path}` was not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}.")

        cols_to_show = ['playerName', 'Team', 'Actual disruption', 'expected disruptions']
        sort_by_col = 'expected disruptions'

    # --- Filter for necessary columns, sort, and display ---
    if not df_processed.empty and sort_by_col in df_processed.columns:
        # Ensure all columns to show actually exist in the dataframe before trying to select them
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
        st.warning("No data to display. Please check the data source.")


if __name__ == "__main__":
    main()

