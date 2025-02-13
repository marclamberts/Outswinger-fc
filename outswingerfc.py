import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_xg_flow(df, logo_path):
    hteam = df['HomeTeam'].iloc[0]
    ateam = df['AwayTeam'].iloc[-1]
    
    a_xG, h_xG, a_min, h_min, a_psxg, h_psxg = [0], [0], [0], [0], [0], [0]
    a_goals_min, h_goals_min = [], []
    
    for x in range(len(df['xG'])):
        if df['TeamId'][x] == ateam:
            a_xG.append(df['xG'][x])
            a_min.append(df['timeMin'][x])
            a_psxg.append(df['PsxG'][x] if 'PsxG' in df.columns else 0)
            if df['isGoal'][x] == 1:
                a_goals_min.append(df['timeMin'][x])
        if df['TeamId'][x] == hteam:
            h_xG.append(df['xG'][x])
            h_min.append(df['timeMin'][x])
            h_psxg.append(df['PsxG'][x] if 'PsxG' in df.columns else 0)
            if df['isGoal'][x] == 1:
                h_goals_min.append(df['timeMin'][x])
    
    def nums_cumulative_sum(nums_list):
        return [sum(nums_list[:i+1]) for i in range(len(nums_list))]
    
    a_cumulative = nums_cumulative_sum(a_xG)
    h_cumulative = nums_cumulative_sum(h_xG)
    a_psxg_cumulative = nums_cumulative_sum(a_psxg)
    h_psxg_cumulative = nums_cumulative_sum(h_psxg)
    
    alast, hlast = round(a_cumulative[-1], 2), round(h_cumulative[-1], 2)
    a_psxg_last, h_psxg_last = round(a_psxg_cumulative[-1], 2), round(h_psxg_cumulative[-1], 2)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    ax.grid(ls='dotted', lw=1, color='black', axis='y', alpha=0.6)
    ax.grid(ls='dotted', lw=1, color='black', axis='x', alpha=0.6)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute', fontsize=16)
    plt.ylabel('xG', fontsize=16)
    
    if hlast > alast:
        ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.3)
        ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.1)
    else:
        ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.1)
        ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.3)
    
    ax.step(a_min, a_cumulative, color='#003f5c', linewidth=5, where='post')
    ax.step(h_min, h_cumulative, color='#ff6361', linewidth=5, where='post')
    
    for goal in a_goals_min:
        ax.scatter(goal, a_cumulative[a_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)
    for goal in h_goals_min:
        ax.scatter(goal, h_cumulative[h_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)
    
    home_goals = df[(df['TeamId'] == hteam) & (df['isGoal'] == 1)].shape[0]
    away_goals = df[(df['TeamId'] == ateam) & (df['isGoal'] == 1)].shape[0]
    ax.text(0.13, 1.1, f"{hteam}", fontsize=35, color="#ff6361", fontweight='bold', transform=ax.transAxes)
    ax.text(0.30, 1.1, "vs", fontsize=35, color="black", fontweight='bold', transform=ax.transAxes)
    ax.text(0.6, 1.1, f"{ateam}", fontsize=35, color="#003f5c", fontweight='bold', transform=ax.transAxes)
    
    score = f"{home_goals} - {away_goals}"
    ax.text(0.95, 1.1, score, fontsize=35, color="black", transform=ax.transAxes)
    
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=0.7)
        ab = AnnotationBbox(imagebox, (1.15, 1.2), frameon=False, xycoords='axes fraction')
        ax.add_artist(ab)
    
    plt.show()
    return fig

st.title("xG Flow Analysis")

csv_folder = "xgCSV"  # Folder where CSV files are stored
files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
selected_file = st.selectbox("Select a match CSV file:", files)

if selected_file:
    file_path = os.path.join(csv_folder, selected_file)
    df = load_data(file_path)
    logo_path = "Outswinger FC (3).png"
    st.pyplot(plot_xg_flow(df, logo_path))
