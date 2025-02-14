import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from mplsoccer.pitch import Pitch

# Streamlit App Title
st.title("⚽ Football xG Analysis: Shot Map & Flow Map")

# Load all CSV files from 'xgCSV' folder
folder_path = "xgCSV"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# File Selection Dropdown
selected_file = st.selectbox("Select Match Data File", csv_files)

if selected_file:
    # Load Data
    df = pd.read_csv(os.path.join(folder_path, selected_file))

    # Extract Teams
    hteam = df['HomeTeam'].iloc[0]
    ateam = df['AwayTeam'].iloc[-1]

    # --- xG Shot Map ---
    st.subheader("🎯 xG Shot Map")

    pitch = Pitch(pitch_type='opta', pitch_width=68, pitch_length=105, pad_bottom=0.5, pad_top=5, pitch_color='white',
                  line_color='black', half=False, goal_type='box', goal_alpha=0.8)

    fig, ax = plt.subplots(figsize=(16, 10))
    pitch.draw(ax=ax)
    plt.gca().invert_xaxis()

    # Plot Team1 (Home)
    for _, row in df[df['TeamId'] == hteam].iterrows():
        color = '#ffa600' if row['isGoal'] else '#ff6361'
        plt.scatter(row['x'], 100 - row['y'], color=color, s=row['xG'] * 800, alpha=0.9, zorder=3)

    # Plot Team2 (Away)
    for _, row in df[df['TeamId'] == ateam].iterrows():
        color = '#ffa600' if row['isGoal'] else '#003f5c'
        plt.scatter(100 - row['x'], row['y'], color=color, s=row['xG'] * 800, alpha=0.9, zorder=3)

    # Display Shot Map
    st.pyplot(fig)

    # --- xG Flow Map ---
    st.subheader("🔄 xG Flow Map")

    # Lists to store xG over time
    a_xG, h_xG = [0], [0]
    a_min, h_min = [0], [0]
    a_goals_min, h_goals_min = [], []

    for x in range(len(df['xG'])):
        if df['TeamId'][x] == ateam:
            a_xG.append(df['xG'][x])
            a_min.append(df['timeMin'][x])
            if df['isGoal'][x] == 1:
                a_goals_min.append(df['timeMin'][x])
        if df['TeamId'][x] == hteam:
            h_xG.append(df['xG'][x])
            h_min.append(df['timeMin'][x])
            if df['isGoal'][x] == 1:
                h_goals_min.append(df['timeMin'][x])

    # Cumulative xG function
    def nums_cumulative_sum(nums_list):
        return [sum(nums_list[:i+1]) for i in range(len(nums_list))]

    a_cumulative = nums_cumulative_sum(a_xG)
    h_cumulative = nums_cumulative_sum(h_xG)

    alast, hlast = round(a_cumulative[-1], 2), round(h_cumulative[-1], 2)

    # Create figure
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    fig2.set_facecolor('white')
    ax2.patch.set_facecolor('white')

    ax2.grid(ls='dotted', lw=1, color='black', axis='y', zorder=1, which='both', alpha=0.6)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_visible(False)

    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute', fontsize=16)
    plt.ylabel('xG', fontsize=16)

    if hlast > alast:
        ax2.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.3)
        ax2.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.1)
    else:
        ax2.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.1)
        ax2.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.3)

    a_min.append(95)
    a_cumulative.append(alast)
    h_min.append(95)
    h_cumulative.append(hlast)

    ax2.set_xlim(0, 95)
    ax2.set_ylim(0, max(alast, hlast) + 0.5)

    ax2.step(x=a_min, y=a_cumulative, color='#003f5c', linewidth=5, where='post', label=f'{ateam} xG: {alast:.2f}')
    ax2.step(x=h_min, y=h_cumulative, color='#ff6361', linewidth=5, where='post', label=f'{hteam} xG: {hlast:.2f}')

    for goal in a_goals_min:
        ax2.scatter(goal, a_cumulative[a_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)

    for goal in h_goals_min:
        ax2.scatter(goal, h_cumulative[h_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)

    home_goals = df[(df['TeamId'] == hteam) & (df['isGoal'] == 1)].shape[0]
    away_goals = df[(df['TeamId'] == ateam) & (df['isGoal'] == 1)].shape[0]

    ax2.text(0.13, 1.1, f"{hteam}", fontsize=35, color="#ff6361", fontweight='bold', ha='center', va='bottom', transform=ax2.transAxes)
    ax2.text(0.30, 1.1, "vs", fontsize=35, color="black", fontweight='bold', ha='center', va='bottom', transform=ax2.transAxes)
    ax2.text(0.6, 1.1, f"{ateam}", fontsize=35, color="#003f5c", fontweight='bold', ha='center', va='bottom', transform=ax2.transAxes)
    
    score = f"{home_goals} - {away_goals}"
    ax2.text(0.95, 1.1, score, fontsize=35, color="black", ha='center', va='bottom', transform=ax2.transAxes)

    # Display xG Flow Map
    st.pyplot(fig2)
