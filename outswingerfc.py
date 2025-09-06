import streamlit as st
import pandas as pd
import numpy as np

def get_metric_info():
    """Returns a dictionary of metric explanations."""
    return {
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and type of assist. A higher xG suggests a player is getting into high-quality scoring positions.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist. It credits creative players for setting up scoring chances, even if the shot is missed.',
        'xT (Expected Threat)': 'Quantifies the increase in the probability of scoring a goal by moving the ball between two points on the pitch. It rewards players for advancing the ball into dangerous areas.',
        'VAEP (Valuing Actions by Estimating Probabilities)': 'A comprehensive metric that assigns a value to every action on the ball (passes, dribbles, shots) based on how it impacts the chances of scoring and conceding.',
        'Expected Shot Danger': 'Focuses on the quality of the shot itself, evaluating how likely a shot from a certain position, under certain pressure, would trouble the goalkeeper.',
        'Expected Cross': 'Evaluates the probability of a cross being successfully completed to a teammate, factoring in the crosser\'s location and the number of defenders in the box.',
        'Expected Disruption': 'Measures a defensive player\'s ability to break up opposition plays. It values tackles and interceptions that prevent high-probability scoring chances for the opponent.',
        'Dribble Success Rate (%)': 'The percentage of attempted dribbles that successfully beat an opponent. A key indicator of a player\'s one-on-one offensive ability.'
    }

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

    # Calculate per 90 metrics
    for col in ['xG', 'xG Open Play', 'xG Set Piece', 'xG Build-up', 'xAG', 'xT', 'VAEP']:
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
    st.sidebar.title("🎙️ The Analyst's Booth")
    st.sidebar.image("https://placehold.co/400x200/2d3748/ffffff?text=SOCCER+ANALYSIS", use_container_width=True)
    
    st.sidebar.info(
        """
        This app displays player stats loaded from the local `data/test.csv` file.
        """
    )

    st.sidebar.header("Metric Leaderboards")
    for metric in metric_pages:
        if st.sidebar.button(metric, use_container_width=True):
            st.session_state.selected_metric = metric
            # No rerun needed here, button click handles it

    # --- Main Page ---
    st.title("📊 Advanced Metrics Leaderboard")

    # --- League Selection Buttons ---
    leagues = ["WSL", "WSL 2", "Frauen-Bundesliga", "NWSL"]
    cols = st.columns(len(leagues))
    for i, league in enumerate(leagues):
        # Change league and reset to xG page of that league
        if cols[i].button(league, use_container_width=True):
            st.session_state.selected_league = league
            st.session_state.selected_metric = metric_pages[0]
            st.rerun()

    # --- Data Loading and Processing ---
    local_csv_path = "data/2025-09-05_Chelsea FC Women - Manchester City WFC.csv"
    df_raw = pd.DataFrame() # Start with an empty DataFrame

    try:
        # Load the data from the local CSV
        df_raw = pd.read_csv(local_csv_path)
        st.success(f"Successfully loaded data from `{local_csv_path}`.")

    except FileNotFoundError:
        st.error(f"Error: The file `{local_csv_path}` was not found. Please make sure it's in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while processing `{local_csv_path}`: {e}.")


    df_processed = calculate_derived_metrics(df_raw)

    # --- Display Selected Metric Page ---
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {st.session_state.selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info[selected_metric_key]}")

    if selected_metric_key == 'xG (Expected Goals)':
        cols_to_show = [
            'PlayerId', 'TeamId', 'xG', 'xG per 90', 'xG Open Play', 'xG Open Play per 90', 
            'xG Set Piece', 'xG Set Piece per 90', 'xG per Shot', 'xG Build-up', 'xG Build-up per 90'
        ]
        sort_by_col = 'xG'
    else:
        # Handle other metrics dynamically
        base_metric_name = selected_metric_key.split(' (')[0]
        cols_to_show = ['PlayerId', 'TeamId', base_metric_name]
        if f'{base_metric_name} per 90' in df_processed.columns:
            cols_to_show.append(f'{base_metric_name} per 90')
        sort_by_col = base_metric_name

    # --- Filter for necessary columns, sort, and display ---
    if not df_processed.empty and sort_by_col in df_processed.columns:
        display_df = df_processed[[col for col in cols_to_show if col in df_processed.columns]]
        display_df = display_df.sort_values(by=sort_by_col, ascending=False).reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)
    elif not df_processed.empty:
        st.warning(f"The metric '{sort_by_col}' is not available in the loaded data file.")
    else:
        st.warning("No data to display. Please check the data source.")


if __name__ == "__main__":
    main()

