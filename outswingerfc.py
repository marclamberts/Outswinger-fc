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
selected_page = st.sidebar.radio("Go to", ("Home", "Shot Map", "Flow Map", "Field Tilt", "Passnetwork"))

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
from mplsoccer import VerticalPitch
from matplotlib.colors import Normalize


# Flow Map page
if selected_page == "Passnetwork":
    st.title("Expected Goals (xG) Flow Map")

# 1. Function to calculate endX and endY based on the type of pass
def get_end_coordinates(row, type_cols):
    endX, endY = 0.0, 0.0
    for j in range(len(type_cols)):
        col = row[type_cols[j]]
        if col == 140:
            endX = row[f'qualifier/{j}/value']
        elif col == 141:
            endY = row[f'qualifier/{j}/value']
    return endX, endY

# 2. Function to calculate EPV values
def calculate_epv(df, epv):
    epv_rows, epv_cols = epv.shape

    # Bin coordinates for x and y based on EPV grid
    df['x1_bin'] = pd.cut(df['x'], bins=epv_cols, labels=False).astype('Int64')
    df['y1_bin'] = pd.cut(df['y'], bins=epv_rows, labels=False).astype('Int64')
    df['x2_bin'] = pd.cut(df['endX'], bins=epv_cols, labels=False).astype('Int64')
    df['y2_bin'] = pd.cut(df['endY'], bins=epv_rows, labels=False).astype('Int64')

    # Function to get EPV value based on the bin indices
    def get_epv_value(bin_indices, epv_grid):
        if pd.notnull(bin_indices[0]) and pd.notnull(bin_indices[1]):
            return epv_grid[int(bin_indices[1])][int(bin_indices[0])]
        return np.nan

    # Assign EPV values for start and end zones
    df['start_zone_value'] = df[['x1_bin', 'y1_bin']].apply(lambda x: get_epv_value(x, epv), axis=1)
    df['end_zone_value'] = df[['x2_bin', 'y2_bin']].apply(lambda x: get_epv_value(x, epv), axis=1)
    
    # Calculate EPV as the difference between start and end zones
    df['epv'] = df['end_zone_value'] - df['start_zone_value']

    return df

# 3. Function to plot the pass network with an optional logo
def add_logo(ax, logo_path):
    logo = mpimg.imread(logo_path)
    imagebox = OffsetImage(logo, zoom=0.6)
    ab = AnnotationBbox(imagebox, (0.95, 1.1), frameon=False, 
                        xycoords='axes fraction', boxcoords="axes fraction")
    ax.add_artist(ab)

def plot_pass_network_with_logo(ax, pitch, average_locs_and_count, passes_between, title, add_logo_to_this_plot=False, logo_path=None):
    pitch.draw(ax=ax)

    norm = Normalize(vmin=average_locs_and_count['epv'].min(), vmax=average_locs_and_count['epv'].max())
    cmap = plt.cm.viridis
    colors = cmap(norm(average_locs_and_count['epv']))

    max_pass_count = passes_between['pass_count'].max()
    passes_between['zorder'] = passes_between['pass_count'] / max_pass_count * 10
    passes_between['alpha'] = passes_between['pass_count'] / max_pass_count

    for _, row in passes_between.iterrows():
        pitch.lines(row['x'], row['y'], row['x_end'], row['y_end'],
                    color='grey', alpha=row['alpha'], lw=4, ax=ax, zorder=row['zorder'])

    pitch.scatter(average_locs_and_count['x'], average_locs_and_count['y'], s=500,
                  color=colors, edgecolors="black", linewidth=1, alpha=1, ax=ax, zorder=11)

    for _, row in average_locs_and_count.iterrows():
        pitch.annotate(row.name, xy=(row.x, row.y), ax=ax, ha='center', va='bottom',
                       fontsize=12, color='black', zorder=12, xytext=(0, -35), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=18, color="black", fontweight='bold', pad=20)

    if add_logo_to_this_plot:
        add_logo(ax, logo_path)

# 4. Function to create pass network data
def create_pass_network_data(df, team_id):
    team_data = df[df['contestantId'] == team_id].reset_index()
    team_data["newsecond"] = 60 * team_data["timeMin"] + team_data["timeSec"]
    team_data.sort_values(by=['newsecond'], inplace=True)
    team_data['passer'] = team_data['playerName']
    team_data['recipient'] = team_data['passer'].shift(-1)
    passes = team_data[team_data['typeId'] == 1]
    completions = passes[passes['outcome'] == 1]

    subs = team_data[team_data['typeId'] == 18]
    sub_times = subs["newsecond"]
    sub_one = sub_times.min()
    completions = completions[completions['newsecond'] < sub_one]

    average_locs_and_count = completions.groupby('passer').agg({'x': ['mean'], 'y': ['mean', 'count'], 'epv': ['mean']})
    average_locs_and_count.columns = ['x', 'y', 'count', 'epv']

    passes_between = completions.groupby(['passer', 'recipient']).id.count().reset_index()
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    return average_locs_and_count, passes_between

# 5. Streamlit interface: Let the user select a match
st.title("Select a Match and Contestant")

# Folder containing CSV match data
match_data_folder = 'WSL 2024-2025'  # Update to your match data folder path
csv_files = sorted([f for f in os.listdir(match_data_folder) if f.endswith('.csv')],
                   key=lambda f: os.path.getmtime(os.path.join(match_data_folder, f)),
                   reverse=True)

# Let the user select a match
selected_match = st.selectbox("Select a match", csv_files)

if selected_match:
    # Load the match data from the selected CSV file
    file_path = os.path.join(match_data_folder, selected_match)
    df = pd.read_csv(file_path)

    # Extract contestantId's (teams) for the match
    team_ids = df['contestantId'].unique()
    selected_team = st.selectbox("Select a Team", team_ids)

    # Calculate the endX and endY using the `type_cols` columns
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Identify type columns (columns with /qualifierId in the name)
    type_cols = [col for col in df.columns if '/qualifierId' in col]
    df[['endX', 'endY']] = df.apply(lambda row: get_end_coordinates(row, type_cols), axis=1, result_type="expand")

    # Load the EPV grid
    epv = pd.read_csv("epv_grid.csv", header=None).to_numpy()

    # Calculate EPV values
    df = calculate_epv(df, epv)

    # Create pass network data for the selected team
    data_team = create_pass_network_data(df, selected_team)

    # Plotting the pass network
    fig, axs = plt.subplots(figsize=(20, 16))
    fig.set_facecolor("white")
    pitch = VerticalPitch(pitch_type='opta', pad_top=5, pitch_color='white', line_color='black')

    plot_pass_network_with_logo(axs, pitch, data_team[0], data_team[1], f"{selected_team} Passing Network")

    # Display the plot in Streamlit
    st.pyplot(fig)
