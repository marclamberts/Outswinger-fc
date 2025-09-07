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
    fig, ax = pitch.draw(figsize=(10, 7)) # Reduced figsize for a smaller plot
    fig.set_facecolor("white")
    ax.set_ylim(49.8, 105) # Cut pitch at halfway line

    # Plot the shots
    for i in range(len(df['x'])):
        row = df.iloc[i]
        color = colors["goal"] if row['isGoal'] else colors["missed"]
        size = row['xG'] * 500
        ax.scatter(row['y'], row['x'], color=color, s=size, alpha=0.7, zorder=3)

    # Add title and subtitle
    ax.text(50, 108, title_text, fontsize=24, weight='bold', color='black', ha='center', va='top') # Adjusted font size
    ax.text(50, 104, "Shot Map from Corners", fontsize=12, style='italic', color='black', ha='center', va='top') # Adjusted font size
    
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

# --- Page Display Functions ---

def display_metrics_page(data_config, metric_info):
    """Renders the main metrics leaderboard page."""
    # --- League Selection ---
    leagues_row1 = ["WSL", "WSL 2", "Frauen-Bundesliga"]
    leagues_row2 = ["Liga F", "NWSL", "Premiere Ligue"]
    
    cols_row1 = st.columns(len(leagues_row1))
    for i, league in enumerate(leagues_row1):
        if cols_row1[i].button(league, use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league
            st.rerun()

    cols_row2 = st.columns(len(leagues_row2))
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
    league_config = data_config.get(selected_league, {})

    if metric_config and league_config.get("minutes_file"):
        try:
            # Load primary metric data
            metric_file_path = resource_path(os.path.join("data", metric_config["file"]))
            df_metric = load_data(metric_file_path)

            # Load minutes data
            minutes_file_path = resource_path(os.path.join("data", league_config["minutes_file"]))
            df_minutes = load_data(minutes_file_path)

            rename_map = {'playerName': 'Player', 'ActualDisruptions': 'Actual disruption', 'ExpectedDisruptions': 'expected disruptions'}
            df_metric.rename(columns=rename_map, inplace=True)
            
            # Merge the two dataframes
            df_raw = pd.merge(df_metric, df_minutes[['Player', 'Minutes Played']], on='Player', how='left')
            
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

def display_corners_page(data_config):
    """Renders the corner analysis page."""
    st.sidebar.header("Corner Filters")
    leagues = list(data_config.keys())
    leagues_with_total = ["Total"] + leagues
    
    selected_league_corners = st.sidebar.selectbox("Select League:", leagues_with_total)
    
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

    df_corners = df_full[df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
    if df_corners.empty:
        st.warning("No events of type 'FromCorner' found in the dataset.")
        return

    teams = ["All"] + sorted(df_corners['TeamId'].unique().tolist())
    players = ["All"] + sorted(df_corners['PlayerId'].unique().tolist())
    is_goal_options = ["All", True, False]
    min_time, max_time = int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max())

    selected_team = st.sidebar.selectbox("Filter by Team:", teams)
    selected_player = st.sidebar.selectbox("Filter by Player:", players)

    if 'GameState' in df_corners.columns:
        game_states = ["All"] + sorted(df_corners['GameState'].unique().tolist())
        selected_state = st.sidebar.selectbox("Filter by Game State:", game_states)
    else:
        selected_state = "All"

    selected_goal = st.sidebar.selectbox("Filter by Goal:", is_goal_options, format_func=lambda x: "All" if x=="All" else ("Yes" if x else "No"))
    selected_time = st.sidebar.slider("Filter by Time (minutes):", min_time, max_time, (min_time, max_time))

    df_filtered = df_corners.copy()
    if selected_team != "All": df_filtered = df_filtered[df_filtered['TeamId'] == selected_team]
    if selected_player != "All": df_filtered = df_filtered[df_filtered['PlayerId'] == selected_player]
    if selected_state != "All" and 'GameState' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['GameState'] == selected_state]
    if selected_goal != "All": df_filtered = df_filtered[df_filtered['isGoal'] == selected_goal]
    df_filtered = df_filtered[df_filtered['timeMin'].between(selected_time[0], selected_time[1])]
    
    # Determine the title for the shot map
    if selected_team != "All":
        plot_title = selected_team
    elif selected_player != "All":
        plot_title = selected_player
    else:
        plot_title = f"{selected_league_corners} Corners"

    st.markdown("---")

    if not df_filtered.empty and all(c in df_filtered.columns for c in ['x', 'y', 'xG', 'isGoal']):
        fig, error_message = create_detailed_shot_map(df_filtered, title_text=plot_title)
        if fig:
            st.pyplot(fig)
            
            # --- Download Buttons ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
            
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Download Image",
                    data=buf,
                    file_name=f"{plot_title.replace(' ', '_')}_shot_map.png",
                    mime="image/png",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="📥 Download Data",
                    data=csv,
                    file_name=f"{plot_title.replace(' ', '_')}_corner_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info(error_message)
            
    elif not df_filtered.empty:
        st.warning("Required columns ('x', 'y', 'xG', 'isGoal') not found for plotting.")
    else:
        st.info("No data available for the selected filters.")


# --- Main App Logic ---
def main():
    metric_info = get_metric_info()
    metric_pages = list(metric_info.keys())

    # Initialize session state
    if 'selected_league' not in st.session_state: st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state: st.session_state.selected_metric = metric_pages[0]
    if 'page_view' not in st.session_state: st.session_state.page_view = "Metrics Leaderboard"

    st.sidebar.title("Outswinger FC")
    st.sidebar.image("https://placehold.co/400x200/2d3748/e2e8f0?text=Outswinger+FC", use_container_width=True)
    
    # --- View Selection Buttons ---
    st.sidebar.write("Select a view:")
    view_cols = st.sidebar.columns(2)
    with view_cols[0]:
        if st.button("📈 Leaderboard", use_container_width=True, disabled=(st.session_state.page_view == "Metrics Leaderboard")):
            st.session_state.page_view = "Metrics Leaderboard"
            st.rerun()
    with view_cols[1]:
        if st.button("⛳ Corners", use_container_width=True, disabled=(st.session_state.page_view == "Corners Analysis")):
            st.session_state.page_view = "Corners Analysis"
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("This app displays player stats for the WSL, WSL 2, Frauen-Bundesliga, Liga F, NWSL, and Premiere Ligue.")

    data_config = {
        "WSL": {
            "minutes_file": "WSL_minutes.csv",
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "WSL_corners.csv"}
        },
        "WSL 2": {
            "minutes_file": "WSL2_minutes.csv",
            'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "WSL2_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "WSL2_corners.csv"}
        },
        "Frauen-Bundesliga": {
            "minutes_file": "FBL_minutes.csv",
            'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "FBL_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "FBL_corners.csv"}
        },
        "Liga F": {
            "minutes_file": "LigaF_minutes.csv",
            'xG (Expected Goals)': {"file": "LigaF.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "LigaF_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "LigaF_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "LigaF_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "LigaF_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "LigaF_corners.csv"}
        },
        "NWSL": {
            "minutes_file": "NWSL_minutes.csv",
            'xG (Expected Goals)': {"file": "NWSL.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "NWSL_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "NWSL_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "NWSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "NWSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "NWSL_corners.csv"}
        },
        "Premiere Ligue": {
            "minutes_file": "PremiereLigue_minutes.csv",
            'xG (Expected Goals)': {"file": "PremiereLigue.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Shots', 'xG', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "PremiereLigue_assists.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Assists', 'ShotAssists', 'xAG'], "sort": 'xAG'},
            'xT (Expected Threat)': {"file": "PremiereLigue_xT.csv", "cols": ['Player', 'Team', 'Minutes Played', 'xT'], "sort": 'xT'},
            'Expected Disruption (xDisruption)': {"file": "PremiereLigue_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes Played', 'Actual disruption', 'expected disruptions'], "sort": 'expected disruptions'},
            'Goal Probability Added (GPA/G+)': {"file": "PremiereLigue_gpa.csv", "cols": ['Player', 'Team', 'Minutes Played', 'GPA', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'},
            'Corners': {"file": "PremiereLigue_corners.csv"}
        }
    }

    if st.session_state.page_view == "Metrics Leaderboard":
        st.title(f"📊 {st.session_state.selected_league} Advanced Metrics Leaderboard")
        st.sidebar.header("Metric Leaderboards")
        for metric in metric_pages:
            if st.sidebar.button(metric, use_container_width=True, disabled=(st.session_state.selected_metric == metric)):
                st.session_state.selected_metric = metric
                st.rerun()
        display_metrics_page(data_config, metric_info)
    
    elif st.session_state.page_view == "Corners Analysis":
        display_corners_page(data_config)

    st.markdown("---")
    st.markdown(f"© {datetime.now().year} Outswinger FC | All rights reserved.")

if __name__ == "__main__":
    main()

