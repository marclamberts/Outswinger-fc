import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from io import BytesIO
import os

# Streamlit app
st.title("xG Flow Visualization")

# Set up file path (modify as needed)
csv_folder = "xgCSV"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Dropdown to select CSV file
selected_file = st.selectbox("Select a match file:", csv_files)
file_path = os.path.join(csv_folder, selected_file)

# Load the data
df = pd.read_csv(file_path)

# Extract home and away teams
hteam = df['HomeTeam'].iloc[0]
ateam = df['AwayTeam'].iloc[-1]

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

# Goals count
home_goals = df[(df['TeamId'] == hteam) & (df['isGoal'] == 1)].shape[0]
away_goals = df[(df['TeamId'] == ateam) & (df['isGoal'] == 1)].shape[0]

# Win probabilities
total_xg = alast + hlast
team1_win_prob = alast / total_xg
team2_win_prob = hlast / total_xg
draw_prob = 1 - (team1_win_prob + team2_win_prob)

# Expected points
team1_xp = (3 * team1_win_prob) + (1 * draw_prob)
team2_xp = (3 * team2_win_prob) + (1 * draw_prob)

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

# Win probabilities & Expected Points
win_prob_text = f"{hteam} Win Probability: {team1_win_prob * 100:.2f}%\n{ateam} Win Probability: {team2_win_prob * 100:.2f}%\n"
xp_text = f"{ateam} Expected Points: {team1_xp:.2f}\n{hteam} Expected Points: {team2_xp:.2f}"
fig.text(0.12, 0.001, win_prob_text + xp_text, ha='left', fontsize=12, fontstyle='italic', color='black')

# Save plot
img_bytes = BytesIO()
plt.savefig(img_bytes, format="png", dpi=300, bbox_inches='tight')
plt.close(fig)

# Download button
st.download_button(label="Download xG Flow as PNG", data=img_bytes.getvalue(), file_name="xg_flow.png", mime="image/png")

st.pyplot(fig)
