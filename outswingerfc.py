import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from matplotlib.patches import Circle
import io

# --- App Configuration ---
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
    if 'Minutes' in df.columns:
        df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce').replace(0, np.nan)
        
        for col in ['xG', 'xAG', 'xT', 'xDisruption', 'GPA']:
            if col in df.columns:
                df[f'{col} per 90'] = (df[col] / df['Minutes'] * 90).round(2)
        
        if 'Shots' in df.columns and 'xG' in df.columns:
            df['Shots'] = pd.to_numeric(df['Shots'], errors='coerce').replace(0, np.nan)
            df['xG per Shot'] = (df['xG'] / df['Shots']).round(2)
    return df

def create_detailed_shot_map(df, title_text="Corner Shots"):
    """Creates a detailed shot map using mplsoccer."""
    
    # Calculate summary statistics
    total_shots = df.shape[0]
    if total_shots == 0:
        return None, "No shots to plot."

    total_goals = int(df['isGoal'].sum())
    total_xg = df['xG'].sum()
    xg_per_shot = total_xg / total_shots if total_shots > 0 else 0
    
    # Define colors
    colors = {"missed": "#003f5c", "goal": "#bc5090"}

    # Create the pitch
    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=False, line_zorder=2, linewidth=0.5)
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor("white")
    ax.set_ylim(49.8, 105) # Cut pitch at halfway line

    # Plot the shots
    for i in range(len(df['x'])):
        row = df.iloc[i]
        color = colors["goal"] if row['isGoal'] else colors["missed"]
        size = row['xG'] * 500
        ax.scatter(row['y'], row['x'], color=color, s=size, alpha=0.7, zorder=3)

    # Add title and subtitle
    ax.text(50, 108, title_text, fontsize=24, weight='bold', color='black', ha='center', va='top')
    ax.text(50, 104, "Shot Map from Corners", fontsize=12, style='italic', color='black', ha='center', va='top')
    
    # Adjust plot for additional space at the bottom
    plt.subplots_adjust(bottom=0.3)

    # --- Circles for Stats ---
    circle_positions = [(0.2, -0.10), (0.2, -0.25), (0.35, -0.10), (0.35, -0.25)]
    circle_texts = ["Shots", "xG/Shot", "Goals", "xG"]
    values = [total_shots, round(xg_per_shot, 2), total_goals, round(total_xg, 2)]
    circle_colors = [colors["missed"], colors["missed"], colors["goal"], colors["goal"]]

    for pos, text, value, color in zip(circle_positions, circle_texts, values, circle_colors):
        circle = Circle(pos, 0.04, transform=ax.transAxes, color=color, zorder=5, clip_on=False)
        ax.add_artist(circle)
        ax.text(pos[0], pos[1] + 0.06, text, transform=ax.transAxes, color='black', fontsize=12, ha='center', va='center', zorder=6)
        ax.text(pos[0], pos[1], value, transform=ax.transAxes, color='white', fontsize=12, weight='bold', ha='center', va='center', zorder=6)

    # --- xG Size Legend ---
    ax.text(0.75, -0.05, "xG Size", transform=ax.transAxes, fontsize=12, color='black', ha='center', va='center', weight='bold')
    ax.scatter([0.72, 0.75, 0.78], [-0.12, -0.12, -0.12], s=[0.1*500, 0.4*500, 0.7*500], color=colors['missed'], transform=ax.transAxes, clip_on=False)
    ax.text(0.75, -0.18, "Low → High", transform=ax.transAxes, fontsize=10, color='black', ha='center', va='center')

    # --- Branding ---
    ax.text(0.75, -0.25, "OUTSWINGERFC.COM", transform=ax.transAxes, fontsize=12, color='black', ha='center', va='center', weight='bold')

    return fig, None

# --- Component Rendering Functions for Data Scouting Page ---

def render_metrics_leaderboard(data_config, metric_info):
    """Renders the player metrics leaderboard component."""
    
    # --- League Selection ---
    leagues_row1 = ["WSL", "WSL 2", "Frauen-Bundesliga"]
    leagues_row2 = ["Liga F", "NWSL", "Premiere Ligue"]
    
    cols_row1 = st.columns(len(leagues_row1))
    for i, league in enumerate(leagues_row1):
        if cols_row1[i].button(league, key=f"league_btn_1_{i}", use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    cols_row2 = st.columns(len(leagues_row2))
    for i, league in enumerate(leagues_row2):
        if cols_row2[i].button(league, key=f"league_btn_2_{i}", use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    selected_league = st.session_state.selected_league
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info.get(selected_metric_key, '')}")
    
    if selected_league == "Frauen-Bundesliga":
        st.info("Note: Data of FC Köln - RB Leipzig is not present as of 06-09-2025")

    metric_config = data_config.get(selected_league, {}).get(selected_metric_key)
    league_config = data_config.get(selected_league, {})

    if metric_config and league_config.get("minutes_file"):
        try:
            metric_file_path = resource_path(os.path.join("data", metric_config["file"]))
            df_metric = load_data(metric_file_path)

            minutes_file_path = resource_path(os.path.join("data", league_config["minutes_file"]))
            df_minutes = load_data(minutes_file_path)

            rename_map = {
                'playerName': 'Player', 'ActualDisruptions': 'Actual disruption', 
                'ExpectedDisruptions': 'xDisruption', 'expected disruptions': 'xDisruption'
            }
            df_metric.rename(columns=rename_map, inplace=True)
            
            df_raw = pd.merge(df_metric, df_minutes[['Player', 'Minutes']], on='Player', how='left')
            df_raw.rename(columns={'Minutes ': 'Minutes'}, inplace=True)
            
            df_processed = calculate_derived_metrics(df_raw)
            sort_by_col = metric_config["sort"]
            
            search_placeholders = {"WSL": "e.g., Sam Kerr", "WSL 2": "e.g., Melissa Johnson", "Frauen-Bundesliga": "e.g., Alexandra Popp", "Liga F": "e.g., Alexia Putellas", "NWSL": "e.g., Sophia Smith", "Premiere Ligue": "e.g., Ada Hegerberg"}
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
        except FileNotFoundError as e:
            st.error(f"Error: A required data file was not found. Details: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("No data configuration found.")

def render_corner_analysis(data_config):
    """Renders the corner analysis component."""
    leagues = list(data_config.keys())
    leagues_with_total = ["Total"] + leagues
    
    selected_league_corners = st.session_state.get('corner_league_selection', 'Total')
    st.selectbox("Select League:", leagues_with_total, key='corner_league_selection')
    
    df_full = pd.DataFrame()
    try:
        if st.session_state.corner_league_selection == "Total":
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
            corner_config = data_config.get(st.session_state.corner_league_selection, {}).get('Corners')
            if corner_config:
                file_path = resource_path(os.path.join("data", corner_config["file"]))
                df_full = load_data(file_path)
            else:
                st.error(f"Corner data configuration not found for {st.session_state.corner_league_selection}.")
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

    df_corners = df_full[df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
    if df_corners.empty:
        st.warning("No events of type 'FromCorner' found in the dataset.")
        return

    # Sidebar Filters (managed by the session state from the sidebar)
    selected_team = st.session_state.get('corner_team', 'All')
    selected_player = st.session_state.get('corner_player', 'All')
    selected_state = st.session_state.get('corner_gamestate', 'All')
    selected_goal = st.session_state.get('corner_isgoal', 'All')
    selected_time = st.session_state.get('corner_time', (int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max())))
    
    df_filtered = df_corners.copy()
    if selected_team != "All": df_filtered = df_filtered[df_filtered['TeamId'] == selected_team]
    if selected_player != "All": df_filtered = df_filtered[df_filtered['PlayerId'] == selected_player]
    if selected_state != "All" and 'GameState' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['GameState'] == selected_state]
    if selected_goal != "All": df_filtered = df_filtered[df_filtered['isGoal'] == selected_goal]
    df_filtered = df_filtered[df_filtered['timeMin'].between(selected_time[0], selected_time[1])]
    
    plot_title = selected_team if selected_team != "All" else selected_player if selected_player != "All" else f"{st.session_state.corner_league_selection} Corners"

    st.markdown("---")
    if not df_filtered.empty and all(c in df_filtered.columns for c in ['x', 'y', 'xG', 'isGoal']):
        fig, error_message = create_detailed_shot_map(df_filtered, title_text=plot_title)
        if fig:
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button("📥 Download Image", data=buf, file_name=f"{plot_title.replace(' ', '_')}_shot_map.png", mime="image/png", use_container_width=True)
            with dl_col2:
                st.download_button("📥 Download Data", data=csv, file_name=f"{plot_title.replace(' ', '_')}_corner_data.csv", mime="text/csv", use_container_width=True)
        else:
            st.info(error_message)
    elif not df_filtered.empty:
        st.warning("Required columns ('x', 'y', 'xG', 'isGoal') not found for plotting.")
    else:
        st.info("No data available for the selected filters.")


# --- Main Page Display Functions ---

def display_landing_page():
    """Renders the initial landing page."""
    # This function should be called before any other Streamlit elements.
    # To enforce a centered layout, this logic must run first.
    
    st.markdown("""
        <style>
            .block-container {
                max-width: 750px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Welcome to Outswinger FC Analytics")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.image("https://placehold.co/400x400/2d3748/e2e8f0?text=Partner+Logo+1", use_container_width=True)
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 150px;'>X</h1>", unsafe_allow_html=True)
    with col3:
        st.image("https://placehold.co/400x400/e2e8f0/2d3748?text=Partner+Logo+2", use_container_width=True)

    st.markdown("---")
    st.subheader("A collaboration in Women's Football Analytics")

    if st.button("Enter Analytics Platform", use_container_width=True, type="primary"):
        st.session_state.app_mode = "MainApp"
        st.rerun()

def display_data_scouting_page(data_config, metric_info):
    """Renders the data scouting page with tabs for different tools."""
    st.title("📊 Data Scouting")

    tab1, tab2 = st.tabs(["Player Metrics Leaderboard", "Corner Analysis"])

    with tab1:
        render_metrics_leaderboard(data_config, metric_info)
    
    with tab2:
        render_corner_analysis(data_config)

def display_match_analysis_page():
    """Placeholder for the Match Analysis page."""
    st.title("🎯 Match Analysis")
    st.markdown("---")
    st.info("This section is currently under development. Tools for in-depth match analysis will be available here soon.")
    st.image("https://placehold.co/800x400/4a5568/e2e8f0?text=Match+Breakdown+Tools+Coming+Soon", use_container_width=True)

def display_player_profiles_page():
    """Placeholder for the Player Profiles page."""
    st.title("👤 Player Profiles")
    st.markdown("---")
    st.info("This section is currently under development. Soon you'll be able to search for individual players and view their detailed statistical profiles.")
    st.image("https://placehold.co/800x400/2d3748/e2e8f0?text=Player+Profile+Dashboard+Coming+Soon", use_container_width=True)

# --- Main App Logic ---
def main():
    # --- Data and Config Initialization ---
    metric_info = get_metric_info()
    data_config = {
        "WSL": {
            "minutes_file": "WSL_minutes.csv",
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "WSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "WSL_corners.csv"}
        },
        "WSL 2": {
            "minutes_file": "WSL2_minutes.csv",
            'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "WSL2_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "WSL2_corners.csv"}
        },
        "Frauen-Bundesliga": {
            "minutes_file": "FBL_minutes.csv",
            'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "FBL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "FBL_corners.csv"}
        },
        "Liga F": {
            "minutes_file": "LigaF_minutes.csv",
            'xG (Expected Goals)': {"file": "LigaF.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "LigaF_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "LigaF_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "LigaF_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "LigaF_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "LigaF_corners.csv"}
        },
        "NWSL": {
            "minutes_file": "NWSL_minutes.csv",
            'xG (Expected Goals)': {"file": "NWSL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "NWSL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "NWSL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "NWSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "NWSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "NWSL_corners.csv"}
        },
        "Premiere Ligue": {
            "minutes_file": "PremiereLigue_minutes.csv",
            'xG (Expected Goals)': {"file": "PremiereLigue.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "PremiereLigue_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "PremiereLigue_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "PremiereLigue_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "PremiereLigue_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "Premiere_Ligue_corners.csv"}
        }
    }
    
    # --- Session State Initialization ---
    if 'app_mode' not in st.session_state: st.session_state.app_mode = "Landing"
    if 'selected_league' not in st.session_state: st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state: st.session_state.selected_metric = list(metric_info.keys())[0]
    if 'page_view' not in st.session_state: st.session_state.page_view = "Data Scouting"

    # --- Page Routing ---
    if st.session_state.app_mode == "Landing":
        display_landing_page()
    else:
        # --- Main App Sidebar and Page Rendering ---
        st.sidebar.title("Outswinger FC")
        st.sidebar.image("https://placehold.co/400x200/2d3748/e2e8f0?text=Outswinger+FC", use_container_width=True)
        
        st.sidebar.header("Main Menu")
        if st.sidebar.button("📊 Data Scouting", use_container_width=True, type="primary" if st.session_state.page_view == "Data Scouting" else "secondary"):
            st.session_state.page_view = "Data Scouting"
            st.rerun()
        if st.sidebar.button("🎯 Match Analysis", use_container_width=True, type="primary" if st.session_state.page_view == "Match Analysis" else "secondary"):
            st.session_state.page_view = "Match Analysis"
            st.rerun()
        if st.sidebar.button("👤 Player Profiles", use_container_width=True, type="primary" if st.session_state.page_view == "Player Profiles" else "secondary"):
            st.session_state.page_view = "Player Profiles"
            st.rerun()
        st.sidebar.markdown("---")

        # --- Contextual Sidebar for Data Scouting ---
        if st.session_state.page_view == "Data Scouting":
            st.sidebar.header("Metric Leaderboard Filters")
            metric_pages = list(metric_info.keys())
            for metric in metric_pages:
                if st.sidebar.button(metric, key=f"metric_btn_{metric}", use_container_width=True, disabled=(st.session_state.selected_metric == metric)):
                    st.session_state.selected_metric = metric
                    st.rerun()
            
            st.sidebar.markdown("---")
            st.sidebar.header("Corner Analysis Filters")
            # These filters are controlled via session_state keys, which are read by the render_corner_analysis function
            # This is a dummy load to get filter options
            try:
                temp_df_full = load_data(resource_path(os.path.join("data", "WSL_corners.csv")))
                df_corners = temp_df_full[temp_df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
                teams = ["All"] + sorted(df_corners['TeamId'].unique().tolist())
                players = ["All"] + sorted(df_corners['PlayerId'].unique().tolist())
                is_goal_options = ["All", True, False]
                min_time, max_time = int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max())
                game_states = ["All"] + sorted(df_corners['GameState'].unique().tolist()) if 'GameState' in df_corners.columns else ["All"]

                st.selectbox("Filter by Team:", teams, key='corner_team')
                st.selectbox("Filter by Player:", players, key='corner_player')
                st.selectbox("Filter by Game State:", game_states, key='corner_gamestate')
                st.selectbox("Filter by Goal:", is_goal_options, key='corner_isgoal', format_func=lambda x: "All" if x=="All" else ("Yes" if x else "No"))
                st.slider("Filter by Time (minutes):", min_time, max_time, (min_time, max_time), key='corner_time')
            except Exception:
                 st.sidebar.warning("Could not load corner filter options.")


        st.sidebar.info("This app displays player stats for the WSL, WSL 2, Frauen-Bundesliga, Liga F, NWSL, and Premiere Ligue.")

        # --- Display Selected Page ---
        if st.session_state.page_view == "Data Scouting":
            display_data_scouting_page(data_config, metric_info)
        elif st.session_state.page_view == "Match Analysis":
            display_match_analysis_page()
        elif st.session_state.page_view == "Player Profiles":
            display_player_profiles_page()

        st.markdown("---")
        st.markdown(f"© {datetime.now().year} Outswinger FC | All rights reserved.")


if __name__ == "__main__":
    main()