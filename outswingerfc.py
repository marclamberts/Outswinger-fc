import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Function to plot xG Flow visualization
def plot_xg_flow(df, hteam, ateam):
    # Initialize lists for xG flow
    a_xG, h_xG = [0], [0]
    a_min, h_min = [0], [0]
    a_psxg, h_psxg = [0], [0]
    a_goals_min, h_goals_min = [], []

    # Process data
    for x in range(len(df['xG'])):
        if df['TeamId'][x] == ateam:
            a_xG.append(df['xG'][x])
            a_min.append(df['timeMin'][x])
            a_psxg.append(df.get('PsxG', [0])[x])
            if df['isGoal'][x] == 1:
                a_goals_min.append(df['timeMin'][x])
        if df['TeamId'][x] == hteam:
            h_xG.append(df['xG'][x])
            h_min.append(df['timeMin'][x])
            h_psxg.append(df.get('PsxG', [0])[x])
            if df['isGoal'][x] == 1:
                h_goals_min.append(df['timeMin'][x])

    # Cumulative sums
    def cumulative_sum(nums_list):
        return [sum(nums_list[:i+1]) for i in range(len(nums_list))]

    a_cumulative = cumulative_sum(a_xG)
    h_cumulative = cumulative_sum(h_xG)
    a_psxg_cumulative = cumulative_sum(a_psxg)
    h_psxg_cumulative = cumulative_sum(h_psxg)

    # Final xG values
    alast, hlast = round(a_cumulative[-1], 2), round(h_cumulative[-1], 2)
    a_psxg_last, h_psxg_last = round(a_psxg_cumulative[-1], 2), round(h_psxg_cumulative[-1], 2)

    # Extract the actual goals from the dataset
    home_goals = sum((df['TeamId'] == hteam) & (df['isGoal'] == 1))
    away_goals = sum((df['TeamId'] == ateam) & (df['isGoal'] == 1))

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.set_facecolor('white')
    ax.patch.set_facecolor('white')

    # Grid settings
    ax.grid(ls='dotted', lw=1, color='black', axis='y', zorder=1, alpha=0.6)
    ax.grid(ls='dotted', lw=1, color='black', axis='x', zorder=1, alpha=0.6)

    # Labels
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute', fontsize=16)
    plt.ylabel('xG', fontsize=16)

    # Fill areas
    ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.3 if hlast > alast else 0.1)
    ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.3 if alast > hlast else 0.1)

    # Step plot
    ax.set_xlim(0, 95)
    ax.set_ylim(0, max(alast, hlast) + 0.5)
    ax.step(a_min, a_cumulative, color='#003f5c', label=f'{ateam} xG: {alast}', linewidth=5, where='post')
    ax.step(h_min, h_cumulative, color='#ff6361', label=f'{hteam} xG: {hlast}', linewidth=5, where='post')

    # Goals as stars
    for goal in a_goals_min:
        ax.scatter(goal, a_cumulative[a_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)
    for goal in h_goals_min:
        ax.scatter(goal, h_cumulative[h_min.index(goal)], color='#ffa600', marker='*', s=500, zorder=3)

    # Title & score
    ax.text(0.95, 1.1, f"{home_goals} - {away_goals}", fontsize=35, color="black", ha='center', transform=ax.transAxes)

    # Save plot
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return img_bytes

# Function to plot xG Shot Map
def plot_xg_shot_map(df, hteam, ateam):
    required_columns = {'LocationX', 'LocationY', 'xG', 'isGoal', 'TeamId'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        st.error(f"Missing columns in CSV: {', '.join(missing_columns)}")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the shots
    for _, shot in df.iterrows():
        x, y = shot.get('LocationX', None), shot.get('LocationY', None)
        if x is None or y is None:
            continue

        color = 'yellow' if shot.get('isGoal', 0) == 1 else plt.cm.viridis(shot.get('xG', 0))
        ax.scatter(x, y, c=color, s=100, alpha=0.8)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_title(f'{hteam} vs {ateam} - xG Shot Map')

    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return img_bytes

# Streamlit app
st.title("xG Flow and Shot Map Visualizations")

# Sidebar
st.sidebar.header("Select Match File")
csv_folder = "xgCSV"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

selected_file = st.sidebar.selectbox("Select a match file:", csv_files)

if st.sidebar.button("Generate xG Visualizations"):
    file_path = os.path.join(csv_folder, selected_file)
    df = pd.read_csv(file_path)

    if not {'HomeTeam', 'AwayTeam'}.issubset(df.columns):
        st.error("CSV is missing 'HomeTeam' or 'AwayTeam' columns.")
    else:
        hteam, ateam = df['HomeTeam'].iloc[0], df['AwayTeam'].iloc[-1]

        # Plot xG Flow
        xg_flow_img = plot_xg_flow(df, hteam, ateam)
        if xg_flow_img:
            st.image(xg_flow_img, use_container_width=True)
            st.download_button("Download xG Flow", xg_flow_img.getvalue(), "xg_flow.png", "image/png")

        # Plot xG Shot Map
        xg_shot_map_img = plot_xg_shot_map(df, hteam, ateam)
        if xg_shot_map_img:
            st.image(xg_shot_map_img, use_container_width=True)
            st.download_button("Download xG Shot Map", xg_shot_map_img.getvalue(), "xg_shot_map.png", "image/png")
