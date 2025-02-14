import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import io
import os
from datetime import datetime

# Streamlit app layout and settings
st.set_page_config(page_title="Football Shot Map", layout="wide")

# Title of the app
st.title("Football Shot Map with xG, Win Probability, and Expected Points")

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
    pitch = Pitch(pitch_type='opta', pitch_width=68, pitch_length=105, pad_bottom=0.5, pad_top=5, pitch_color='white',
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
    imagebox = OffsetImage(logo_img, zoom=0.05)  # Adjust zoom for scaling the logo
    ab = AnnotationBbox(imagebox, (0.98, 1.2), frameon=False, xycoords='axes fraction', box_alignment=(1, 1))

    # Add the logo to the plot
    ax.add_artist(ab)

    # Add win probability text at the bottom-left
    win_text = f"Win Probability:\n{team1_name}: {team1_win_prob*100:.2f}%\n{team2_name}: {team2_win_prob*100:.2f}%"
    plt.text(0.02, -0.03, win_text, ha='left', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

    # Add Expected Points text in the bottom-left corner
    xp_text = f"Expected Points:\n{team1_name}: {team1_xp:.2f} | {team2_name}: {team2_xp:.2f}"
    plt.text(0.02, -0.10, xp_text, ha='left', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

    # Add text in the bottom-right corner
    text = "OUTSWINGERFC.COM\nData via Opta | Women's Super League 2024-2025"
    plt.text(0.98, -0.03, text, ha='right', va='top', fontsize=12, color='black', weight='bold', transform=ax.transAxes)

    # Save the plot to a file (in memory buffer)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", dpi=300, bbox_inches='tight', facecolor='white')
    img_buf.seek(0)

    # Provide a download button for the image
    st.download_button(
        label="Download Shot Map",
        data=img_buf,
        file_name=f"{team1_name}_vs_{team2_name}_shot_map.png",
        mime="image/png"
    )

    # Optionally, display the plot in Streamlit
    st.image(img_buf, use_container_width=True)

    # Display the win probability and expected points as text
    st.markdown(win_text)
    st.markdown(xp_text)
