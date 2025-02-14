import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import io
import os
import numpy as np
from datetime import datetime

# Streamlit app layout and settings
st.set_page_config(page_title="Outswinger FC - Data visualisation app", layout="wide")

# Title of the app
st.title("Football Shot Map App")

# Sidebar Menu
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ("Home", "Shot Map", "Flow Map", "Field Tilt"))

if selected_page == "Shot Map":
    st.title("Expected Goals (xG) Shotmap")

    # Path to the xgCSV folder
    xg_csv_folder = 'xgCSV'

    # List all CSV files in the xgCSV folder and sort by modification time (most recent first)
    csv_files = sorted(
        [f for f in os.listdir(xg_csv_folder) if f.endswith('.csv')],
        key=lambda f: os.path.getmtime(os.path.join(xg_csv_folder, f)),
        reverse=True  # Sort by most recent first
    )

    # Let the user select a CSV file
    selected_file = st.selectbox("Select a CSV file", csv_files)

    # If a file is selected, process and display it
    if selected_file:
        # Load the data
        file_path = os.path.join(xg_csv_folder, selected_file)
        df = pd.read_csv(file_path)

        # Extract team names from the file name
        teams = selected_file.split('_')[-1].split(' - ')  # Extract team names from the file name

        # Define the home and away teams
        team1_name = teams[0]
        team2_name = teams[1].split('.')[0]

        # Count goals for each team
        team1_goals = df.loc[(df['TeamId'] == team1_name) & (df['isGoal'] == True), 'isGoal'].sum()
        team2_goals = df.loc[(df['TeamId'] == team2_name) & (df['isGoal'] == True), 'isGoal'].sum()

        team1 = df.loc[df['TeamId'] == team1_name].reset_index()
        team2 = df.loc[df['TeamId'] == team2_name].reset_index()

        # Calculate total xG and PsxG for each team
        team1_xg = team1['xG'].sum()
        team2_xg = team2['xG'].sum()
        team1_psxg = team1['PsxG'].sum() if 'PsxG' in team1.columns else 0
        team2_psxg = team2['PsxG'].sum() if 'PsxG' in team1.columns else 0

        # Calculate Win Probabilities and Expected Points for each team
        total_xg = team1_xg + team2_xg
        team1_win_prob = team1_xg / total_xg
        team2_win_prob = team2_xg / total_xg
        draw_prob = 1 - (team1_win_prob + team2_win_prob)

        # Expected Points Calculation
        team1_xp = (3 * team1_win_prob) + (1 * draw_prob)
        team2_xp = (3 * team2_win_prob) + (1 * draw_prob)

        # Plot the pitch
        pitch = Pitch(pitch_type='opta', pitch_width=68, pitch_length=105, pad_bottom=1.5, pad_top=5, pitch_color='white',
                      line_color='black', half=False, goal_type='box', goal_alpha=0.8)
        fig, ax = plt.subplots(figsize=(16, 10))
        pitch.draw(ax=ax)
        fig.set_facecolor('white')  # Set background to white
        plt.gca().invert_xaxis()

        # Add xG for each team in the title
        plt.text(80, 90, f"{team1_xg:.2f} xG", color='#ff6361', ha='center', fontsize=30, fontweight='bold')
        plt.text(20, 90, f"{team2_xg:.2f} xG", color='#003f5c', ha='center', fontsize=30, fontweight='bold')

        # Create the title with goals and expected points included
        title = f"{team1_name} vs {team2_name} ({team1_goals} - {team2_goals})"
        subtitle = f"xG Shot Map"

        # Title text
        plt.text(0.40, 1.05, title, ha='center', va='bottom', fontsize=25, fontweight='bold', transform=ax.transAxes)
        plt.text(0.16, 1.02, subtitle, ha='right', va='bottom', fontsize=18, transform=ax.transAxes)

        # Add logo in the top-right corner
        logo_path = 'logo.png'  # Replace with the path to your logo file
        logo_img = mpimg.imread(logo_path)  # Read the logo image

        # Create the logo image and place it at the top-right corner of the plot
        imagebox = OffsetImage(logo_img, zoom=0.06)  # Adjust zoom for scaling the logo
        ab = AnnotationBbox(imagebox, (0.99, 1.15), frameon=False, xycoords='axes fraction', box_alignment=(1, 1))

        # Add the logo to the plot
        ax.add_artist(ab)

        # Add win probability text at the bottom-left
        win_text = f"Win Probability:\n{team1_name}: {team1_win_prob*100:.2f}% | {team2_name}: {team2_win_prob*100:.2f}%"
        plt.text(0.02, -0.03, win_text, ha='left', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

        # Add Expected Points text in the bottom-left corner
        xp_text = f"Expected Points:\n{team1_name}: {team1_xp:.2f} | {team2_name}: {team2_xp:.2f}"
        plt.text(0.02, -0.10, xp_text, ha='left', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

        # Add text in the bottom-right corner
        text = "OUTSWINGERFC.COM\nData via Opta | Women's Super League 2024-2025"
        plt.text(0.98, -0.03, text, ha='right', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

        # Function to add jitter to the scatter plot (random small shift in position)
        def add_jitter(x, y, jitter_strength=0.5):
            return x + np.random.uniform(-jitter_strength, jitter_strength), y + np.random.uniform(-jitter_strength, jitter_strength)

        # Scatter plot code for team1 (home team)
        for x in range(len(team1['x'])):
            x_pos, y_pos = add_jitter(team1['x'][x], 100 - team1['y'][x])  # Apply jitter
            if team1['Type_of_play'][x] == 'FromCorner' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'FromCorner' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team1['Type_of_play'][x] == 'RegularPlay' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'RegularPlay' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team1['Type_of_play'][x] == 'FastBreak' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'FastBreak' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team1['Type_of_play'][x] == 'ThrowinSetPiece' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'ThrowinSetPiece' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team1['Type_of_play'][x] == 'SetPiece' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'SetPiece' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team1['Type_of_play'][x] == 'Penalty' and team1['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team1['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team1['Type_of_play'][x] == 'Penalty' and team1['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#ff6361', s=team1['xG'][x] * 800, alpha=0.9, zorder=2)

        # Scatter plot code for team2 (away team)
        for x in range(len(team2['x'])):
            x_pos, y_pos = add_jitter(100 - team2['x'][x], team2['y'][x])  # Apply jitter
            if team2['Type_of_play'][x] == 'FromCorner' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'FromCorner' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team2['Type_of_play'][x] == 'RegularPlay' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'RegularPlay' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team2['Type_of_play'][x] == 'FastBreak' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'FastBreak' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team2['Type_of_play'][x] == 'ThrowinSetPiece' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'ThrowinSetPiece' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team2['Type_of_play'][x] == 'SetPiece' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'SetPiece' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)
            elif team2['Type_of_play'][x] == 'Penalty' and team2['isGoal'][x] == True:
                plt.scatter(x_pos, y_pos, color='#ffa600', s=team2['xG'][x] * 800, alpha=0.9, zorder=3)
            elif team2['Type_of_play'][x] == 'Penalty' and team2['isGoal'][x] == False:
                plt.scatter(x_pos, y_pos, color='#003f5c', s=team2['xG'][x] * 800, alpha=0.9, zorder=2)

        
        # Save the figure to a PNG image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Create download button
        st.download_button(
            label="Download Shot Map as PNG",
            data=img_buf,
            file_name=f"shot_map_{team1_name}_{team2_name}_outswingerfc.png",
            mime="image/png"
        )
        st.pyplot(fig)


import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Flow Map page
if selected_page == "Flow Map":
    st.title("Expected Goals (xG) Flow Map")

    # Path to the xgCSV folder
    xg_csv_folder = 'xgCSV'

    # List all CSV files in the xgCSV folder and sort by modification time (most recent first)
    csv_files = sorted(
        [f for f in os.listdir(xg_csv_folder) if f.endswith('.csv')],
        key=lambda f: os.path.getmtime(os.path.join(xg_csv_folder, f)),
        reverse=True  # Sort by most recent first
    )

    # Let the user select a CSV file
    selected_file = st.selectbox("Select a CSV file", csv_files)

    # If a file is selected, process and display it
    if selected_file:
        # Load the data
        file_path = os.path.join(xg_csv_folder, selected_file)
        df = pd.read_csv(file_path)

        # Extract team names from the file name
        teams = selected_file.replace('.csv', '').split('_')[-1].split(' - ')
        if len(teams) < 2:
            st.error("Error extracting team names. Check filename format.")
        else:
            hteam, ateam = teams

            # Initialize lists to store xG, PsxG, and goal data for both teams
            a_xG, h_xG = [], []
            a_min, h_min = [], []
            a_psxg, h_psxg = [], []
            a_goals_min, h_goals_min = [], []

            # Process each row to track xG, PsxG, and goals for both teams
            for _, row in df.iterrows():
                team_id = row['TeamId']
                xg_value = row['xG']
                psxg_value = row['PsxG'] if 'PsxG' in df.columns else 0
                minute = row['timeMin']

                if team_id == ateam:
                    a_xG.append(xg_value)
                    a_min.append(minute)
                    a_psxg.append(psxg_value)
                    if row['isGoal'] == 1:
                        a_goals_min.append(minute)

                elif team_id == hteam:
                    h_xG.append(xg_value)
                    h_min.append(minute)
                    h_psxg.append(psxg_value)
                    if row['isGoal'] == 1:
                        h_goals_min.append(minute)

            # Cumulative xG and PsxG calculations
            def nums_cumulative_sum(nums_list):
                return [sum(nums_list[:i+1]) for i in range(len(nums_list))]

            a_cumulative = nums_cumulative_sum([0] + a_xG)
            h_cumulative = nums_cumulative_sum([0] + h_xG)
            a_psxg_cumulative = nums_cumulative_sum([0] + a_psxg)
            h_psxg_cumulative = nums_cumulative_sum([0] + h_psxg)

            # Ensure xG and time lists have the same starting values
            a_min = [0] + a_min
            h_min = [0] + h_min

            # Last xG and PsxG values
            alast, hlast = round(a_cumulative[-1], 2), round(h_cumulative[-1], 2)
            a_psxg_last, h_psxg_last = round(a_psxg_cumulative[-1], 2), round(h_psxg_cumulative[-1], 2)

            # Calculate home and away goals from the dataset
            home_goals = df.loc[(df['TeamId'] == hteam) & (df['isGoal'] == 1)].shape[0]
            away_goals = df.loc[(df['TeamId'] == ateam) & (df['isGoal'] == 1)].shape[0]

            # Plot setup
            fig, ax = plt.subplots(figsize=(16, 10))
            fig.set_facecolor('white')
            ax.patch.set_facecolor('white')

            # Set up grid and axis labels
            ax.grid(ls='dotted', lw=1, color='black', axis='y', zorder=1, which='both', alpha=0.6)
            ax.grid(ls='dotted', lw=1, color='black', axis='x', zorder=1, which='both', alpha=0.6)
            ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            plt.xticks([0, 15, 30, 45, 60, 75, 90])
            plt.xlabel('Minute', fontsize=16)
            plt.ylabel('xG', fontsize=16)

            # Plot cumulative xG and PsxG fill
            ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.3)
            ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.3)

            # Ensure both teams are plotted fully
            a_min.append(95)
            a_cumulative.append(alast)
            h_min.append(95)
            h_cumulative.append(hlast)

            ax.step(a_min, a_cumulative, color='#003f5c', linewidth=5, where='post')
            ax.step(h_min, h_cumulative, color='#ff6361', linewidth=5, where='post')

            # Label the lines on the plot (make labels black and adjusted positioning)
            ax.text(92, a_cumulative[-1] + 0.05, ateam, color='black', fontsize=18, fontweight='bold')  # Adjust y position to avoid overlap
            ax.text(92, h_cumulative[-1] + 0.05, hteam, color='black', fontsize=18, fontweight='bold')  # Adjust y position to avoid overlap

            # Goal annotations
            for goal in a_goals_min:
                ax.scatter(goal, a_cumulative[a_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)
            for goal in h_goals_min:
                ax.scatter(goal, h_cumulative[h_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)

            # Title (remove 'WFC' from title only)
            title_hteam = hteam.replace('WFC', '').strip()  # Remove WFC from home team name for title
            title_ateam = ateam.replace('WFC', '').strip()  # Remove WFC from away team name for title
            ax.text(0.4, 1.1, f"{title_hteam} vs {title_ateam} ({home_goals} - {away_goals})", fontsize=35, color="black", fontweight='bold', ha='center', transform=ax.transAxes)

            # Subtitle (adjusted y position to avoid overlap)
            subtitle = f"{hteam} xG: {hlast:.2f} | PsxG: {h_psxg_last:.2f}\n{ateam} xG: {alast:.2f} | PsxG: {a_psxg_last:.2f}"
            ax.text(0.2, 1.02, subtitle, fontsize=18, color="black", ha='center', transform=ax.transAxes)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Footer and logo (placed under the plot)
            st.markdown("OUTSWINGERFC.COM\nData via Opta | Women's Super League 2024-2025")
            # Show match score and expected points
            # Calculate win probabilities and expected points
            total_xg = alast + hlast
            team1_win_prob = alast / total_xg
            team2_win_prob = hlast / total_xg
            draw_prob = 1 - (team1_win_prob + team2_win_prob)

# Expected Points Calculation
            team1_xp = (3 * team1_win_prob) + (1 * draw_prob)
            team2_xp = (3 * team2_win_prob) + (1 * draw_prob)

# Add win probability and expected points in bottom-left corner
            win_prob_text = f"{hteam} Win Probability: {team1_win_prob * 100:.2f}%\n{ateam} Win Probability: {team2_win_prob * 100:.2f}%\n"
            xp_text = f"{ateam} Expected Points: {team1_xp:.2f}\n{hteam} Expected Points: {team2_xp:.2f}"

# Show win probability and expected points
            st.write(win_prob_text)
            st.write(xp_text)
    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st

# Streamlit page selection
selected_page = st.selectbox("Select a page", ["Field Tilt"])

# Field Tilt page
if selected_page == "Field Tilt":
    st.title("Field Tilt Analysis")

    # Folder containing CSV match data (change to your path)
    match_data_folder = 'WSL 2024-2025'  # Replace with your folder path
    csv_files = sorted([f for f in os.listdir(match_data_folder) if f.endswith('.csv')],
                       key=lambda f: os.path.getmtime(os.path.join(match_data_folder, f)),
                       reverse=True)

    # Let the user select a match
    selected_match = st.selectbox("Select a match", csv_files)

    if selected_match:
        file_path = os.path.join(match_data_folder, selected_match)
        df = pd.read_csv(file_path)

        # Show the first few rows of the selected match data
        st.write("Preview of the selected match data:")
        st.write(df.head())

        # Check if 'contestantId' is in the CSV columns
        if 'contestantId' in df.columns:
            # Extract and display unique contestantIds (home and away teams)
            contestant_ids = df['contestantId'].unique()
            st.write(f"Unique contestantIds in the selected match: {contestant_ids}")

            # You could also map contestantId to team names if the mapping is available
            mapping_file_path = 'opta_club_rankings_womens_14022025.xlsx'  # Change to your mapping file path
            mapping_df = pd.read_excel(mapping_file_path)
            id_to_team = dict(zip(mapping_df['id'], mapping_df['team']))

            # Show team names based on contestantId
            team_names = [id_to_team.get(str(contestant_id), 'Unknown Team') for contestant_id in contestant_ids]
            st.write("Corresponding Team Names:")
            for cid, team in zip(contestant_ids, team_names):
                st.write(f"Contestant ID: {cid} - Team Name: {team}")

            # Assuming the first two contestant IDs are the home and away teams
            hteam_id = contestant_ids[0]
            ateam_id = contestant_ids[1]

            # Get the corresponding team names
            hteam_name = id_to_team.get(str(hteam_id), 'Unknown Team')
            ateam_name = id_to_team.get(str(ateam_id), 'Unknown Team')

            # Filter final third passes
            df_final_third = df.loc[(df['typeId'] == 1) & (df['x'] > 70)]  # Passes in the final third

            # Group by teams and count final third passes
            final_third_count = df_final_third.groupby('contestantId').size().reset_index(name='Final third passes')

            # Calculate total final third passes
            total_final_third_passes = final_third_count['Final third passes'].sum()

            # Add Field Tilt for each team
            final_third_count['Field Tilt'] = (final_third_count['Final third passes'] / total_final_third_passes) * 100

            # Initialize lists for minute-by-minute Field Tilt
            minutes = list(range(0, 96))  # 0-95 minutes
            home_tilt = []
            away_tilt = []

            # Compute Field Tilt per minute
            for minute in minutes:
                home_passes = df_final_third[(df_final_third['contestantId'] == hteam_id) & (df_final_third['timeMin'] == minute)].shape[0]
                away_passes = df_final_third[(df_final_third['contestantId'] == ateam_id) & (df_final_third['timeMin'] == minute)].shape[0]
                
                total_passes = home_passes + away_passes
                if total_passes == 0:
                    home_tilt.append(0)  # Neutral if no attacking passes
                    away_tilt.append(0)
                else:
                    home_tilt.append((home_passes / total_passes) * 100)
                    away_tilt.append(-1 * (away_passes / total_passes) * 100)  # Make away team values negative

            # Smooth the Field Tilt using a larger moving average window
            def moving_average(values, window):
                return np.convolve(values, np.ones(window) / window, mode='same')

            home_tilt_smoothed = moving_average(home_tilt, 15)  # Larger window for rounder curves
            away_tilt_smoothed = moving_average(away_tilt, 15)

            # Calculate total attacking contributions for percentages
            home_total = sum([x for x in home_tilt if x > 0])
            away_total = sum([-x for x in away_tilt if x < 0])  # Convert negative values to positive
            overall_total = home_total + away_total

            home_percentage = (home_total / overall_total) * 100
            away_percentage = (away_total / overall_total) * 100

            # Calculate net dominance
            net_tilt = np.maximum(home_tilt_smoothed + away_tilt_smoothed, 0)  # Positive area for home
            net_tilt_away = np.minimum(home_tilt_smoothed + away_tilt_smoothed, 0)  # Negative area for away

            # Goals for context
            home_goals_min = df[(df['contestantId'] == hteam_id) & (df['typeId'] == 16)]['timeMin'].tolist()
            away_goals_min = df[(df['contestantId'] == ateam_id) & (df['typeId'] == 16)]['timeMin'].tolist()

            # Plot Field Dominance
            fig, ax = plt.subplots(figsize=(22, 12))  # Increased plot size here
            fig.set_facecolor('white')
            ax.patch.set_facecolor('white')

            # Grid and aesthetics
            ax.grid(ls='dotted', lw=1, color='black', alpha=0.4, zorder=1)
            ax.axhline(0, color='black', linestyle='dashed', linewidth=2, alpha=0.7)  # 0% line

            # Fill regions based on net dominance
            ax.fill_between(minutes, 0, net_tilt, where=(net_tilt > 0), interpolate=True, color='#ff6361', alpha=0.6, label=f'{hteam_name} Dominance')
            ax.fill_between(minutes, 0, net_tilt_away, where=(net_tilt_away < 0), interpolate=True, color='#003f5c', alpha=0.6, label=f'{ateam_name} Dominance')

            # Add goal markers
            for goal in home_goals_min:
                ax.scatter(goal, 0, color='black', marker='*', s=200, label=f'{hteam_name} Goal')

            for goal in away_goals_min:
                ax.scatter(goal, 0, color='black', marker='*', s=200, label=f'{ateam_name} Goal')

            # Add logo in the top-right corner
            logo_path = 'logo.png'  # Replace with the path to your logo file
            logo_img = mpimg.imread(logo_path)  # Read the logo image
            imagebox = OffsetImage(logo_img, zoom=0.7)  # Adjust zoom to control logo size
            ab = AnnotationBbox(imagebox, (0.95, 1.25), frameon=False, xycoords='axes fraction', box_alignment=(1, 1))
            ax.add_artist(ab)

            # Title and subtitles
            title = f"{hteam_name} ({len(home_goals_min)}) - {ateam_name} ({len(away_goals_min)}) - Field Tilt"
            subtitle = f"{hteam_name}: {home_percentage:.1f}% | {ateam_name}: {away_percentage:.1f}%"
            footer = "Field Tilt is the share of touches in the final third.\n15 Minute Moving Average | Data via Opta"

            # Adjust the title and subtitle positions by increasing the `y` value
            plt.title(title, fontsize=40, color='black', weight='bold', loc='left', y=1.12)  # Higher title
            plt.suptitle(subtitle, fontsize=25, color='black', style='italic', x=0.27, y=0.93)  # Higher subtitle
            fig.text(0.12, 0.02, footer, fontsize=20, color='black', style='italic')  # Larger footer text

            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Labels and limits
            plt.xlabel('Minute', fontsize=18)  # Larger font size for labels
            plt.ylabel('Field tilt (%)', fontsize=18)
            plt.ylim(-100, 100)
            plt.xlim(0, 95)

            # Legend
            ax.legend(loc='upper right', fontsize=14, frameon=False)  # Larger legend

            # Save and display
            plt.savefig('field_dominance_chart_with_logo.png', dpi=300, bbox_inches='tight', facecolor='white')
            st.pyplot(fig)

        else:
            st.write("The selected CSV does not contain a 'contestantId' column.")

