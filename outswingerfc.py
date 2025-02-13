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

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.set_facecolor('white')
    ax.patch.set_facecolor('white')

    # Grid settings
    ax.grid(ls='dotted', lw=1, color='black', axis='y', zorder=1, alpha=0.6)
    ax.grid(ls='dotted', lw=1, color='black', axis='x', zorder=1, alpha=0.6)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    # Labels
    plt.xticks([0, 15, 30, 45, 60, 75, 90])
    plt.xlabel('Minute', fontsize=16)
    plt.ylabel('xG', fontsize=16)

    # Fill areas
    if hlast > alast:
        ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.3)
        ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.1)
    else:
        ax.fill_between(h_min, h_cumulative, color='#ff6361', alpha=0.1)
        ax.fill_between(a_min, a_cumulative, color='#003f5c', alpha=0.3)

    # Step plot
    a_min.append(95)
    a_cumulative.append(alast)
    h_min.append(95)
    h_cumulative.append(hlast)

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
    ax.text(0.13, 1.1, f"{hteam}", fontsize=35, color="#ff6361", fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.30, 1.1, "vs", fontsize=35, color="black", fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.6, 1.1, f"{ateam}", fontsize=35, color="#003f5c", fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.95, 1.1, f"{home_goals} - {away_goals}", fontsize=35, color="black", ha='center', transform=ax.transAxes)

    # Subtitle
    subtitle = f"{hteam} xG: {hlast:.2f} | PsxG: {h_psxg_last:.2f}\n{ateam} xG: {alast:.2f} | PsxG: {a_psxg_last:.2f}"
    fig.text(0.12, 0.9, subtitle, ha='left', fontsize=14, color='black', fontstyle='italic')

    # Footer
    fig.text(0.80, 0.01, 'OUTSWINGERFC.COM\nData via Opta | Eredivisie 2024-2025', fontstyle='italic', fontsize=14, color='black')

    # Logo
    logo_path = 'Data visuals/Outswinger FC (3).png'
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        imagebox = OffsetImage(logo_img, zoom=0.7)
        ab = AnnotationBbox(imagebox, (1.15, 1.2), frameon=False, xycoords='axes fraction', box_alignment=(1, 1))
        ax.add_artist(ab)

    # Save plot
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return img_bytes


# Function to plot xG Shot Map
def plot_xg_shot_map(df, hteam, ateam):
    # Filter the shots
    shots = df[(df['TeamId'] == hteam) | (df['TeamId'] == ateam)]

    # Create the plot for the shot map
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the soccer field outline (dimensions of a soccer field in meters)
    ax.plot([0, 105], [0, 0], color='green', lw=2)  # Bottom
    ax.plot([0, 105], [68, 68], color='green', lw=2)  # Top
    ax.plot([0, 0], [0, 68], color='green', lw=2)  # Left
    ax.plot([105, 105], [0, 68], color='green', lw=2)  # Right

    # Plot the shots
    for _, shot in shots.iterrows():
        if shot['isGoal'] == 1:
            ax.scatter(shot['LocationX'], shot['LocationY'], c='yellow', s=100, label='Goal' if shot['TeamId'] == ateam else '', alpha=0.8)
        else:
            ax.scatter(shot['LocationX'], shot['LocationY'], c=plt.cm.viridis(shot['xG']), s=100, alpha=0.8)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_xlabel('Field Length (meters)')
    ax.set_ylabel('Field Width (meters)')
    ax.set_title(f'{hteam} vs {ateam} - xG Shot Map')

    # Return image
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

# Dropdown to select CSV file
selected_file = st.sidebar.selectbox("Select a match file:", csv_files)

# Button to show visualization
if st.sidebar.button("Generate xG Visualizations"):
    file_path = os.path.join(csv_folder, selected_file)

    # Load the data
    df = pd.read_csv(file_path)

    # Extract home and away teams
    hteam = df['HomeTeam'].iloc[0]
    ateam = df['AwayTeam'].iloc[-1]

    # Plot xG Flow
    xg_flow_img = plot_xg_flow(df, hteam, ateam)
    st.image(xg_flow_img, use_column_width=True)
    st.download_button(label="Download xG Flow as PNG", data=xg_flow_img.getvalue(), file_name="xg_flow.png", mime="image/png")

    # Plot xG Shot Map
    xg_shot_map_img = plot_xg_shot_map(df, hteam, ateam)
    st.image(xg_shot_map_img, use_column_width=True)
    st.download_button(label="Download xG Shot Map as PNG", data=xg_shot_map_img.getvalue(), file_name="xg_shot_map.png", mime="image/png")
