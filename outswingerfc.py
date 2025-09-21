import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.patches import Circle
import io
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# --- App Configuration ---
st.set_page_config(
    page_title="She Plots FC x Outswinger FC | Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# --- Caching ---
@st.cache_data(ttl=3600)
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

@st.cache_data(ttl=3600)
def load_profile_data(file_path):
    """Loads an Excel file for player profiles."""
    return pd.read_excel(file_path)

# --- Helper Functions ---

def get_metric_info():
    """Returns a dictionary of metric explanations."""
    return {
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and type of assist. A higher xG suggests a player is getting into high-quality scoring positions.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist. It credits creative players for setting up scoring chances, even if the shot is missed.',
        'xT (Expected Threat)': 'Quantifies the increase in the probability of scoring a goal by moving the ball between two points on the pitch. It rewards players for advancing the ball into dangerous areas.',
        'Expected Disruption (xDisruption)': "Measures a defensive player's ability to break up opposition plays. It values tackles and interceptions that prevent high-probability scoring chances for the opponent.",
        'Goal Probability Added (GPA/G+)': "Measures the change in goal probability from a player's actions on the ball. A positive GPA indicates that the player's actions increased the team's chances of scoring."
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
    """Creates a detailed shot map for corners using mplsoccer."""
    total_shots = df.shape[0]
    if total_shots == 0: return None, "No shots to plot."
    total_goals, total_xg = int(df['isGoal'].sum()), df['xG'].sum()
    xg_per_shot = total_xg / total_shots if total_shots > 0 else 0
    colors = {"missed": "#003f5c", "goal": "#bc5090"}
    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black', half=False, line_zorder=2, linewidth=0.5)
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor("white")
    ax.set_ylim(49.8, 105)
    for i in range(len(df['x'])):
        row, color = df.iloc[i], colors["goal"] if df.iloc[i]['isGoal'] else colors["missed"]
        size = row['xG'] * 500
        ax.scatter(row['y'], row['x'], color=color, s=size, alpha=0.7, zorder=3)
    ax.text(50, 108, title_text, fontsize=24, weight='bold', color='black', ha='center', va='top')
    ax.text(50, 104, "Shot Map from Corners", fontsize=12, style='italic', color='black', ha='center', va='top')
    plt.subplots_adjust(bottom=0.3)
    circle_positions, circle_texts = [(0.2, -0.10), (0.2, -0.25), (0.35, -0.10), (0.35, -0.25)], ["Shots", "xG/Shot", "Goals", "xG"]
    values = [total_shots, round(xg_per_shot, 2), total_goals, round(total_xg, 2)]
    circle_colors = [colors["missed"], colors["missed"], colors["goal"], colors["goal"]]
    for pos, text, value, color in zip(circle_positions, circle_texts, values, circle_colors):
        circle = Circle(pos, 0.04, transform=ax.transAxes, color=color, zorder=5, clip_on=False)
        ax.add_artist(circle)
        ax.text(pos[0], pos[1] + 0.06, text, transform=ax.transAxes, color='black', fontsize=12, ha='center', va='center', zorder=6)
        ax.text(pos[0], pos[1], value, transform=ax.transAxes, color='white', fontsize=12, weight='bold', ha='center', va='center', zorder=6)
    ax.text(0.75, -0.05, "xG Size", transform=ax.transAxes, fontsize=12, color='black', ha='center', va='center', weight='bold')
    ax.scatter([0.72, 0.75, 0.78], [-0.12, -0.12, -0.12], s=[0.1*500, 0.4*500, 0.7*500], color=colors['missed'], transform=ax.transAxes, clip_on=False)
    ax.text(0.75, -0.18, "Low → High", transform=ax.transAxes, fontsize=10, color='black', ha='center', va='center')
    ax.text(0.75, -0.25, "SHE PLOTS FC", transform=ax.transAxes, fontsize=12, color='black', ha='center', va='center', weight='bold')
    return fig, None

def create_player_profile_fig(df, player_name, position_group):
    """Generates a player profile visualization and returns the matplotlib figure."""
    df.rename(columns={"Nationality": "Passport country"}, inplace=True)
    if position_group == 'Centre-back':
        positions, categories, roles = ['CB', 'RCB', 'LCB'], {"Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"], "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Forward passes per 90", "Accurate forward passes, %", "Passes to final third per 90", "Accurate passes to final third, %"], "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"], "Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"], "Proactive Defending": ["Interceptions per 90", "PAdj Interceptions", "Sliding tackles per 90", "PAdj Sliding tackles"], "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %"], "Box Defending": ["Shots blocked per 90"], "Sweeping": []}, {"Ball Player": ["Progressive Passing", "Security"], "Libero": ["Progressive Passing", "Ball Carrying", "Creativity"], "Wide Creator": ["Creativity", "Ball Carrying"], "Aggressor": ["Proactive Defending", "Duelling"], "Physical Dominator": ["Duelling", "Box Defending"], "Box Defender": ["Box Defending", "Duelling"]}
    elif position_group == 'Full-back':
        positions, categories, roles = ['LB', 'RB', 'LWB', 'RWB'], {"Box Defending": ["Shots blocked per 90"], "Duelling": ["Duels per 90", "Duels won, %", "Aerial duels per 90", "Aerial duels won, %", "Defensive duels per 90", "Defensive duels won, %"], "Pressing": ["PAdj Interceptions", "PAdj Sliding tackles", "Counterpressing recoveries per 90"], "Security": ["Accurate passes, %", "Back passes per 90", "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %"], "Playmaking": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"], "Final Third": ["Passes to final third per 90", "Accurate passes to final third, %", "Crosses per 90", "Accurate crosses, %", "Touches in box per 90"], "Overlapping": ["Accelerations per 90", "Fouls suffered per 90"], "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"]}, {"False Wing": ["Playmaking", "Ball Carrying", "Final Third"], "Flyer": ["Overlapping", "Final Third", "Ball Carrying"], "Playmaker": ["Playmaking", "Security", "Final Third"], "Safety": ["Security", "Box Defending", "Pressing"], "Ballwinner": ["Duelling", "Pressing"], "Defensive FB": ["Box Defending", "Duelling", "Pressing"]}
    elif position_group == 'Midfielder':
        positions, categories, roles = ['LCMF', 'RCMF', 'CFM', 'LDMF', 'RDMF', 'RAMF', 'LAMF', 'AMF', 'DMF'], {"Creativity": ["Key passes per 90", "Shot assists per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"], "Box Crashing": ["Touches in box per 90", "Shots per 90", "xG per 90", "Non-penalty goals per 90"], "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %", "Accelerations per 90"], "Progressive Passing": ["Progressive passes per 90", "Accurate progressive passes, %", "Passes to final third per 90", "Accurate passes to final third, %"], "Dictating": ["Passes per 90", "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %", "Lateral passes per 90"], "Ball Winning": ["PAdj Interceptions", "Counterpressing recoveries per 90", "Defensive duels won, %"], "Destroying": ["Duels per 90", "Defensive duels per 90", "PAdj Sliding tackles", "Fouls per 90"]}, {"Anchor": ["Destroying", "Ball Winning", "Dictating"], "DLP": ["Dictating", "Progressive Passing", "Creativity"], "Ball Winner": ["Ball Winning", "Destroying"], "Box to Box": ["Progressive Passing", "Ball Carrying", "Box Crashing", "Ball Winning"], "Box Crasher": ["Box Crashing", "Ball Carrying", "Creativity"], "Playmaker": ["Creativity", "Progressive Passing", "Dictating"], "Attacking mid": ["Creativity", "Box Crashing", "Progressive Passing"]}
    elif position_group == 'Attacking Midfielder':
        positions, categories, roles = ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'], {"Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"], "Build up": ["Passes per 90", "Accurate passes, %", "Progressive passes per 90"], "Final ball": ["Key passes per 90", "xA per 90", "Deep completions per 90"], "Wide creation": ["Crosses per 90", "Accurate crosses, %", "Passes to penalty area per 90"], "Movement": ["Accelerations per 90", "Touches in box per 90"], "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90"], "1v1 ability": ["Dribbles per 90", "Successful dribbles, %", "Fouls suffered per 90"], "Box presence": ["Touches in box per 90", "xG per 90", "Shots per 90"], "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]}, {"Winger": ["Wide creation", "1v1 ability", "Ball Carrying"], "Direct Dribbler": ["1v1 ability", "Ball Carrying", "Movement"], "Industrious Winger": ["Pressing", "Ball Carrying", "Wide creation"], "Shadow Striker": ["Box presence", "Finishing", "Movement"], "Wide playmaker": ["Wide creation", "Final ball", "Build up"], "Playmaker": ["Final ball", "Build up", "1v1 ability"]}
    elif position_group == 'Attacker':
        positions, categories, roles = ['CF', 'LW', 'RW'], {"Pressing": ["Counterpressing recoveries per 90", "PAdj Interceptions", "Defensive duels per 90"], "Ball Carrying": ["Progressive runs per 90", "Dribbles per 90", "Successful dribbles, %"], "Creativity": ["Key passes per 90", "xA per 90", "Smart passes per 90", "Accurate smart passes, %"], "Link Play": ["Passes per 90", "Accurate passes, %", "Deep completions per 90", "Passes to final third per 90"], "Movement": ["Accelerations per 90", "Touches in box per 90"], "Box Presence": ["Touches in box per 90", "Shots per 90", "Aerial duels per 90", "Aerial duels won, %"], "Finishing": ["Non-penalty goals per 90", "xG per 90", "Shots on target, %"]}, {"Poacher": ["Finishing", "Box Presence", "Movement"], "Second Striker": ["Creativity", "Ball Carrying", "Finishing"], "Link Forward": ["Link Play", "Creativity", "Ball Carrying"], "False 9": ["Link Play", "Creativity", "Ball Carrying"], "Complete Forward": ["Finishing", "Box Presence", "Ball Carrying", "Link Play"], "Power Forward": ["Box Presence", "Finishing"], "Pressing Forward": ["Pressing", "Movement", "Box Presence"]}
    else:
        return None, f"Invalid position_group: '{position_group}'."
    
    position_df = df[df['Position'].str.contains('|'.join(positions), na=False)].copy()
    data = position_df[["Player", "Team", "Age", "Position", "Minutes played", "Passport country"]].copy()
    for cat, metrics in categories.items():
        for metric in metrics:
            if metric in position_df.columns: data[f"{cat}_{metric}"] = position_df[metric]
    data.fillna(data.mean(numeric_only=True), inplace=True)
    for cat, metrics in categories.items():
        for metric in metrics:
            col_name = f"{cat}_{metric}"
            if col_name in data.columns and pd.api.types.is_numeric_dtype(data[col_name]): data[f"{col_name}_z"] = zscore(data[col_name])
    for cat in categories:
        z_cols = [col for col in data.columns if col.startswith(cat) and col.endswith('_z')]
        if z_cols:
            data[f"{cat}_z_avg"] = data[z_cols].mean(axis=1)
            data[f"{cat}_percentile"] = data[f"{cat}_z_avg"].apply(lambda x: norm.cdf(x) * 100)
    for role, cats in roles.items():
        cat_cols = [f"{c}_percentile" for c in cats if f"{c}_percentile" in data.columns]
        if cat_cols: data[role] = data[cat_cols].mean(axis=1)

    try:
        player_row = data[data["Player"] == player_name].iloc[0]
    except IndexError:
        return None, f"Player '{player_name}' not found in the {position_group} dataset."

    age, pos, team = player_row.get("Age", "N/A"), player_row.get("Position", "N/A"), player_row.get("Team", "N/A")
    nation, minutes = player_row.get("Passport country", "N/A"), player_row.get("Minutes played", 0)
    cat_percentiles = {cat: player_row.get(f"{cat}_percentile", 0) for cat in categories}
    cat_percentiles = {k: v for k, v in cat_percentiles.items() if v is not None and not pd.isna(v)}
    role_scores = {role: player_row.get(role, 0) for role in roles}
    top_roles = sorted(role_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    category_percentile_cols = [f"{cat}_percentile" for cat in categories if f"{cat}_percentile" in data.columns]
    avg_rating = player_row[category_percentile_cols].mean() if category_percentile_cols else np.nan
    category_cols = [col for col in category_percentile_cols if col in data.columns]
    player_vector = player_row[category_cols].values.reshape(1, -1)
    all_vectors = data[category_cols].values
    similarities = cosine_similarity(player_vector, all_vectors).flatten()
    sim_df = data.copy()
    sim_df['Similarity'] = similarities
    sim_df = sim_df[sim_df['Player'] != player_name]
    top_similar = sim_df.sort_values('Similarity', ascending=False).head(4)

    fig = plt.figure(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, player_name, fontsize=24, weight='bold', ha='left')
    fig.text(0.05, 0.88, "Biography", fontsize=14, weight='bold', ha='left')
    bio_text = f"Age: {age}\nPosition: {pos}\nTeam: {team}\nPassport country: {nation}\nMinutes: {int(minutes)}"
    fig.text(0.05, 0.85, bio_text, fontsize=11, ha='left', va='top', linespacing=1.5)
    x0, y0, box_size = 0.55, 0.92, 0.012
    sorted_role_scores = sorted(role_scores.items(), key=lambda item: list(roles.keys()).index(item[0]))
    for i, (role, score) in enumerate(sorted_role_scores):
        fig.text(x0, y0 - i*0.045, role, fontsize=10, ha='left')
        for j in range(10):
            color = "gold" if j < round(score/10) else "lightgrey"
            rect = plt.Rectangle((x0+0.22 + j*0.022, y0 - i*0.045 - 0.005), box_size, box_size, transform=fig.transFigure, facecolor=color, edgecolor="black", lw=0.3)
            fig.add_artist(rect)
    if not pd.isna(avg_rating):
        rect = plt.Rectangle((0.33, 0.80), 0.10, 0.08, transform=fig.transFigure, facecolor="lightblue", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(0.335, 0.84, "Rating:", fontsize=10, weight="bold", ha='left')
        fig.text(0.38, 0.82, f"{avg_rating:.2f}", fontsize=12, weight="bold", ha='center')
    for i, (role, score) in enumerate(top_roles):
        tile_x = 0.05 + i * (0.18 + 0.02)
        rect = plt.Rectangle((tile_x, 0.60), 0.18, 0.08, transform=fig.transFigure, facecolor="gold", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(tile_x + 0.02, 0.64, role, fontsize=11, weight="bold", ha='left', va='center')
        fig.text(tile_x + 0.16, 0.64, f"{score:.0f}", fontsize=13, weight="bold", ha='right', va='center')
    ax_bar = fig.add_axes([0.05, 0.20, 0.9, 0.35])
    bar_data = dict(sorted(cat_percentiles.items(), key=lambda x: x[1]))
    bars = ax_bar.barh(list(bar_data.keys()), list(bar_data.values()), color='gold', edgecolor='black')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_title("Positional Responsibilities", fontsize=12, weight='bold', loc='left')
    ax_bar.tick_params(axis='y', labelsize=9)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.6)
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va='center', fontsize=9)
    fig.text(0.05, 0.12, "Similar Player Profiles", fontsize=12, weight='bold', ha='left')
    for i, (_, row) in enumerate(top_similar.iterrows()):
        tile_x = 0.05 + i * (0.22 + 0.02)
        rect = plt.Rectangle((tile_x, 0.02), 0.22, 0.08, transform=fig.transFigure, facecolor="lightgreen", edgecolor="black", lw=1)
        fig.add_artist(rect)
        fig.text(tile_x + 0.01, 0.075, f"{row['Player']}", fontsize=10, weight="bold", ha='left', va='top')
        fig.text(tile_x + 0.01, 0.055, f"{row['Team']}", fontsize=9, ha='left', va='top')
        fig.text(tile_x + 0.01, 0.035, f"{row['Similarity']*100:.1f}% Similarity", fontsize=9, ha='left', va='top')
    return fig, "Profile generated successfully."

def create_match_shot_map_fig(df, file_path):
    """Generates the xG shot map for a given match."""
    df = df[~((df['timeMin'] >= 120) & (df['Type_of_play'] == 'Penalty'))]
    file_name = os.path.basename(file_path)
    try:
        teams = file_name.split('_')[-1].replace('.csv', '').split(' - ')
        team1_name, team2_name = teams[0].strip(), teams[1].strip()
    except IndexError:
        return None, "Filename format is incorrect. Expected 'Date_TeamA - TeamB.csv'"

    team1 = df[df['TeamId'] == team1_name].reset_index(drop=True)
    team2 = df[df['TeamId'] == team2_name].reset_index(drop=True)

    team1_goals, team2_goals = team1['isGoal'].sum(), team2['isGoal'].sum()
    team1_xg, team2_xg = team1['xG'].sum(), team2['xG'].sum()

    pitch = Pitch(pitch_type='opta', pitch_width=68, pitch_length=105, pad_bottom=0.5, pad_top=5,
                  pitch_color='#001f3f', line_color='white', half=False, goal_type='box', goal_alpha=0.8)
    fig, ax = plt.subplots(figsize=(16, 10))
    pitch.draw(ax=ax)
    fig.set_facecolor('#001f3f')
    ax.invert_xaxis()

    team1_miss, team2_miss, goal_color, own_goal_color = '#7FDBFF', '#FFDC00', '#2ECC40', 'red'
    def is_true(val):
        if isinstance(val, str): return val.strip().lower() in ('1', 'true', 'yes', 'y', 't')
        return bool(val)

    def plot_team_shots(team_df, is_team1, color_miss, color_goal):
        for _, row in team_df.iterrows():
            x_raw, y_raw = row['x'], row['y']
            x_plot, y_plot = (x_raw, 100 - y_raw) if is_team1 else (100 - x_raw, y_raw)
            og = is_true(row.get('isOwnGoal', False))
            if og:
                x_plot, y_plot = 100 - x_plot, 100 - y_plot
                color, size, z = own_goal_color, 300, 4
            else:
                size = row['xG'] * 800
                color = color_goal if row['isGoal'] else color_miss
                z = 3 if row['isGoal'] else 2
            ax.scatter(x_plot, y_plot, color=color, s=size, alpha=0.9, zorder=z)

    plot_team_shots(team1, True, team1_miss, goal_color)
    plot_team_shots(team2, False, team2_miss, goal_color)
    ax.text(80, 90, f"{team1_xg:.2f} xG", color=team1_miss, ha='center', fontsize=30, fontweight='bold')
    ax.text(20, 90, f"{team2_xg:.2f} xG", color=team2_miss, ha='center', fontsize=30, fontweight='bold')
    title = f"{team1_name} vs {team2_name} ({int(team1_goals)} - {int(team2_goals)})"
    ax.text(0.5, 1.06, title, ha='center', va='bottom', fontsize=25, fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 1.02, "xG Shot Map", ha='center', va='bottom', fontsize=17, color='white', transform=ax.transAxes)

    LOGO_DIR = resource_path("data/logos")
    badge1_path, badge2_path = os.path.join(LOGO_DIR, f"{team1_name}.png"), os.path.join(LOGO_DIR, f"{team2_name}.png")
    if os.path.exists(badge1_path): ax.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(badge1_path), zoom=0.12), (0.07, 1.1), xycoords='axes fraction', frameon=False))
    if os.path.exists(badge2_path): ax.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(badge2_path), zoom=0.07), (0.93, 1.1), xycoords='axes fraction', frameon=False))
    logo_path = os.path.join(LOGO_DIR, 'sheplotsfc.png')
    if os.path.exists(logo_path): ax.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(logo_path), zoom=0.1), (0.9, 0.1), frameon=False, xycoords='axes fraction'))

    total_xg = team1_xg + team2_xg
    team1_win_prob = team1_xg / total_xg if total_xg != 0 else 0.5
    team2_win_prob = 1 - team1_win_prob
    ax.text(0.98, -0.03, "Data via Opta | FBL 2025-2026", ha='right', va='top', fontsize=12, color='white', weight='bold', transform=ax.transAxes)
    win_text = f"Win Probability:\n{team1_name}: {team1_win_prob*100:.1f}%\n{team2_name}: {team2_win_prob*100:.1f}%"
    ax.text(0.02, -0.03, win_text, ha='left', va='top', fontsize=12, color='white', weight='bold', transform=ax.transAxes)

    ax.scatter([], [], color=goal_color, s=200, label='Goal')
    ax.scatter([], [], color=team1_miss, s=200, label=f'{team1_name} Shot')
    ax.scatter([], [], color=team2_miss, s=200, label=f'{team2_name} Shot')
    ax.scatter([], [], color=own_goal_color, s=200, label='Own Goal')
    ax.legend(frameon=False, fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.15), labelcolor='white', handletextpad=0.5, ncol=4)
    
    return fig, "Match analysis generated."


# --- Page Display Functions ---
def display_landing_page():
    st.markdown("""<style>.block-container { max-width: 750px; padding-top: 5rem; } div[data-testid="stSidebar"] { display: none; }</style>""", unsafe_allow_html=True)
    st.title("She Plots FC - Analytics")
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1: st.image(resource_path("sheplotsfc2.png"), use_container_width=True)
    with col2: st.markdown("<h1 style='text-align: center; margin-top: 150px;'>X</h1>", unsafe_allow_html=True)
    with col3: st.image(resource_path("Outswinger FC.png"), use_container_width=True)
    st.markdown("---")
    st.subheader("Opening the world of WoSo data")
    if st.button("Enter Analytics Platform", use_container_width=True, type="primary"):
        st.session_state.app_mode = "MainApp"; st.rerun()

def display_data_scouting_page(data_config, metric_info):
    st.title("📊 Data Scouting")
    leagues_row1, leagues_row2 = ["WSL", "WSL 2", "Frauen-Bundesliga"], ["Liga F", "NWSL", "Premiere Ligue"]
    cols_row1, cols_row2 = st.columns(len(leagues_row1)), st.columns(len(leagues_row2))
    for i, league in enumerate(leagues_row1):
        if cols_row1[i].button(league, key=f"league_btn_1_{i}", use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league; st.rerun()
    for i, league in enumerate(leagues_row2):
        if cols_row2[i].button(league, key=f"league_btn_2_{i}", use_container_width=True, disabled=(st.session_state.selected_league == league)):
            st.session_state.selected_league = league; st.rerun()
    selected_league, selected_metric_key = st.session_state.selected_league, st.session_state.selected_metric
    st.header(f"📈 {selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info.get(selected_metric_key, '')}")
    if selected_league == "Frauen-Bundesliga" and datetime.now().date() < datetime(2025, 9, 6).date():
        st.info("Note: Data of FC Köln - RB Leipzig is not present.")
    metric_config, league_config = data_config.get(selected_league, {}).get(selected_metric_key), data_config.get(selected_league, {})
    if metric_config and league_config.get("minutes_file"):
        try:
            df_metric, df_minutes = load_data(resource_path(os.path.join("data", metric_config["file"]))), load_data(resource_path(os.path.join("data", league_config["minutes_file"])))
            df_metric.rename(columns={'playerName': 'Player', 'ActualDisruptions': 'Actual disruption', 'ExpectedDisruptions': 'xDisruption', 'expected disruptions': 'xDisruption'}, inplace=True)
            df_raw = pd.merge(df_metric, df_minutes[['Player', 'Minutes']], on='Player', how='left')
            df_raw.rename(columns={'Minutes ': 'Minutes'}, inplace=True)
            df_processed = calculate_derived_metrics(df_raw)
            sort_by_col = metric_config["sort"]
            placeholder = {"WSL": "e.g., Sam Kerr", "WSL 2": "e.g., Melissa Johnson", "Frauen-Bundesliga": "e.g., Alexandra Popp", "Liga F": "e.g., Alexia Putellas", "NWSL": "e.g., Sophia Smith", "Premiere Ligue": "e.g., Ada Hegerberg"}.get(selected_league, "Search for a player...")
            with st.container():
                col1, col2 = st.columns([2, 1.5])
                with col1: search_term = st.text_input("Search for a player:", placeholder=placeholder)
                with col2: top_n = st.slider("Number of players to display:", 5, 50, 15, 5)
                display_option = st.radio("Display format:", ("📄 Data Table", "📊 Visualization"), horizontal=True, label_visibility="collapsed")
            if search_term: df_processed = df_processed[df_processed['Player'].str.contains(search_term, case=False, na=False)]
            if not df_processed.empty and sort_by_col in df_processed.columns:
                display_df = df_processed.sort_values(by=sort_by_col, ascending=False).head(top_n).reset_index(drop=True)
                display_df.index = display_df.index + 1
                if display_option == "📊 Visualization":
                    st.subheader("Top Performers Chart")
                    max_val, x_domain = display_df[sort_by_col].max(), [0, display_df[sort_by_col].max()]
                    chart = alt.Chart(display_df).mark_bar().encode(x=alt.X(f'{sort_by_col}:Q', title=selected_metric_key, scale=alt.Scale(domain=x_domain)), y=alt.Y('Player:N', sort='-x', title="Player")).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.subheader("Detailed Data Table")
                    st.dataframe(display_df[[col for col in metric_config["cols"] if col in display_df.columns]], use_container_width=True)
            elif not df_processed.empty: st.warning(f"The metric '{sort_by_col}' is not available.")
            else: st.info("No matching players found.")
        except FileNotFoundError as e: st.error(f"Error: A required data file was not found. Details: {e}")
        except Exception as e: st.error(f"An error occurred: {e}")
    else: st.warning("No data configuration found.")

def display_corners_page(data_config):
    st.title("⛳ Corners")
    leagues, selected_league_corners = list(data_config.keys()), st.session_state.get('corner_league_selection', 'Total')
    df_full = pd.DataFrame()
    try:
        if selected_league_corners == "Total":
            all_dfs = []
            for league in leagues:
                corner_config = data_config.get(league, {}).get('Corners')
                if corner_config:
                    try: all_dfs.append(load_data(resource_path(os.path.join("data", corner_config["file"]))))
                    except FileNotFoundError: st.warning(f"Corner data for {league} not found. Skipping.")
            if all_dfs: df_full = pd.concat(all_dfs, ignore_index=True)
        else:
            corner_config = data_config.get(selected_league_corners, {}).get('Corners')
            if corner_config: df_full = load_data(resource_path(os.path.join("data", corner_config["file"])))
            else: st.error(f"Corner data configuration not found for {selected_league_corners}."); return
    except FileNotFoundError: st.error(f"Error: The specified corner data file was not found."); return
    except Exception as e: st.error(f"An error occurred: {e}"); return
    if df_full.empty: st.warning("No corner data could be loaded for the selected league(s)."); return
    df_corners = df_full[df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
    if df_corners.empty: st.warning("No events of type 'FromCorner' found in the dataset."); return
    selected_team, selected_player, selected_state, selected_goal, selected_time = st.session_state.get('corner_team', 'All'), st.session_state.get('corner_player', 'All'), st.session_state.get('corner_gamestate', 'All'), st.session_state.get('corner_isgoal', 'All'), st.session_state.get('corner_time', (int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max())))
    df_filtered = df_corners.copy()
    if selected_team != "All": df_filtered = df_filtered[df_filtered['TeamId'] == selected_team]
    if selected_player != "All": df_filtered = df_filtered[df_filtered['PlayerId'] == selected_player]
    if selected_state != "All" and 'GameState' in df_filtered.columns: df_filtered = df_filtered[df_filtered['GameState'] == selected_state]
    if selected_goal != "All": df_filtered = df_filtered[df_filtered['isGoal'] == selected_goal]
    df_filtered = df_filtered[df_filtered['timeMin'].between(selected_time[0], selected_time[1])]
    plot_title = selected_team if selected_team != "All" else selected_player if selected_player != "All" else f"{selected_league_corners} Corners"
    if not df_filtered.empty and all(c in df_filtered.columns for c in ['x', 'y', 'xG', 'isGoal']):
        fig, error_message = create_detailed_shot_map(df_filtered, title_text=plot_title)
        if fig:
            st.pyplot(fig)
            buf, csv = io.BytesIO(), df_filtered.to_csv(index=False).encode('utf-8')
            fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1: st.download_button("📥 Download Image", data=buf, file_name=f"{plot_title.replace(' ', '_')}_shot_map.png", mime="image/png", use_container_width=True)
            with dl_col2: st.download_button("📥 Download Data", data=csv, file_name=f"{plot_title.replace(' ', '_')}_corner_data.csv", mime="text/csv", use_container_width=True)
        else: st.info(error_message)
    elif not df_filtered.empty: st.warning("Required columns ('x', 'y', 'xG', 'isGoal') not found for plotting.")
    else: st.info("No data available for the selected filters.")

def display_match_analysis_page():
    st.title("🎯 Match Analysis")
    MATCH_DIR = resource_path("data/matchxg")
    
    try:
        leagues = [name for name in os.listdir(MATCH_DIR) if os.path.isdir(os.path.join(MATCH_DIR, name))]
        if not leagues:
            st.error(f"No league folders found in '{MATCH_DIR}'. Please create subfolders for each league.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            selected_league = st.selectbox("Select League", sorted(leagues))

        league_path = os.path.join(MATCH_DIR, selected_league)
        match_files = [f for f in os.listdir(league_path) if f.endswith('.csv')]
        
        if not match_files:
            st.warning(f"No match files found for {selected_league}.")
            return

        # Extract all unique team names from filenames in the selected league
        all_teams = set()
        for f in match_files:
            try:
                teams = os.path.splitext(f)[0].split('_')[-1].split(' - ')
                all_teams.add(teams[0].strip())
                all_teams.add(teams[1].strip())
            except IndexError:
                continue # Skip malformed filenames
        
        with col2:
            team_filter = st.selectbox("Filter by Team (Optional)", ["All Teams"] + sorted(list(all_teams)))

        # Filter match list based on team selection
        if team_filter != "All Teams":
            filtered_matches = [f for f in match_files if team_filter in f]
        else:
            filtered_matches = match_files
            
        if not filtered_matches:
            st.warning(f"No matches found for {team_filter} in {selected_league}.")
            return

        # Clean names for display
        match_display_names = sorted([os.path.splitext(f)[0] for f in filtered_matches])
        selected_match_name = st.selectbox("Select a Match to Analyze", match_display_names)

        if st.button("Generate Analysis", use_container_width=True, type="primary"):
            if selected_match_name:
                with st.spinner("Generating match analysis..."):
                    file_path = os.path.join(league_path, f"{selected_match_name}.csv")
                    df = load_data(file_path)
                    fig, message = create_match_shot_map_fig(df, file_path)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error(f"Could not generate visualization: {message}")

    except FileNotFoundError:
        st.error(f"Match data directory not found at '{MATCH_DIR}'. Please create 'data/matchxg' and league subfolders.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def display_player_profiling_page():
    st.title("👤 Player Profiling")
    PROFILES_DIR = resource_path("data/profiles")
    position_map = { 'Centre-back': ['CB', 'RCB', 'LCB'], 'Full-back': ['LB', 'RB', 'LWB', 'RWB'], 'Midfielder': ['LCMF', 'RCMF', 'CFM', 'LDMF', 'RDMF', 'RAMF', 'LAMF', 'AMF', 'DMF'], 'Attacking Midfielder': ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'], 'Attacker': ['CF', 'LW', 'RW'] }

    try:
        league_files = [f for f in os.listdir(PROFILES_DIR) if f.endswith('.xlsx')]
        if not league_files: st.error(f"No Excel files found in '{PROFILES_DIR}'."); return
        league_names = sorted([os.path.splitext(f)[0] for f in league_files])
    except FileNotFoundError: st.error(f"Directory not found at '{PROFILES_DIR}'. Please create 'data/profiles' and add Excel files."); return

    col1, col2 = st.columns(2)
    with col1: selected_league_name = st.selectbox("Select League", league_names)
    with col2: position_group = st.selectbox("Select Position Group", list(position_map.keys()))

    if selected_league_name:
        try:
            full_path = os.path.join(PROFILES_DIR, f"{selected_league_name}.xlsx")
            df = load_profile_data(full_path)
            positions_to_check, df['Position'] = position_map[position_group], df['Position'].astype(str)
            position_df = df[df['Position'].str.contains('|'.join(positions_to_check), na=False)]
            player_list = sorted(position_df['Player'].unique())
            if not player_list: st.warning(f"No players found for '{position_group}' in {selected_league_name}."); return
            selected_player = st.selectbox("Select Player", player_list)
            if st.button("Generate Profile", use_container_width=True, type="primary"):
                if selected_player:
                    with st.spinner("Generating profile..."):
                        fig, message = create_player_profile_fig(df, selected_player, position_group)
                        if fig: st.pyplot(fig)
                        else: st.error(message)
        except FileNotFoundError: st.error(f"Could not find the file: {selected_league_name}.xlsx")
        except Exception as e: st.error(f"An error occurred while processing the file: {e}")

# --- Main App Logic ---
def main():
    """Main function to run the Streamlit app."""
    metric_info = get_metric_info()
    data_config = { "WSL": { "minutes_file": "WSL_minutes.csv", 'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "WSL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "WSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "WSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "WSL_corners.csv"} }, "WSL 2": { "minutes_file": "WSL2_minutes.csv", 'xG (Expected Goals)': {"file": "WSL2.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "WSL2_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "WSL2_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "WSL2_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "WSL2_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "WSL2_corners.csv"} }, "Frauen-Bundesliga": { "minutes_file": "FBL_minutes.csv", 'xG (Expected Goals)': {"file": "FBL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "FBL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "FBL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "FBL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "FBL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "FBL_corners.csv"} }, "Liga F": { "minutes_file": "LigaF_minutes.csv", 'xG (Expected Goals)': {"file": "LigaF.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "LigaF_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "LigaF_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "LigaF_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "LigaF_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "LigaF_corners.csv"} }, "NWSL": { "minutes_file": "NWSL_minutes.csv", 'xG (Expected Goals)': {"file": "NWSL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "NWSL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "NWSL_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "NWSL_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "NWSL_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "NWSL_corners.csv"} }, "Premiere Ligue": { "minutes_file": "PremiereLigue_minutes.csv", 'xG (Expected Goals)': {"file": "PremiereLigue.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90', 'OpenPlay_xG', 'SetPiece_xG'], "sort": 'xG'}, 'xAG (Expected Assisted Goals)': {"file": "PremiereLigue_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'ShotAssists', 'xAG', 'xAG per 90'], "sort": 'xAG'}, 'xT (Expected Threat)': {"file": "PremiereLigue_xT.csv", "cols": ['Player', 'Team', 'Minutes', 'xT', 'xT per 90'], "sort": 'xT'}, 'Expected Disruption (xDisruption)': {"file": "PremiereLigue_xDisruption.csv", "cols": ['Player', 'Team', 'Minutes', 'Actual disruption', 'xDisruption', 'xDisruption per 90'], "sort": 'xDisruption'}, 'Goal Probability Added (GPA/G+)': {"file": "PremiereLigue_gpa.csv", "cols": ['Player', 'Team', 'Minutes', 'GPA', 'GPA per 90', 'Avg GPA', 'GPA Rating'], "sort": 'GPA'}, 'Corners': {"file": "Premiere_Ligue_corners.csv"} } }
    
    # --- Session State Initialization ---
    if 'app_mode' not in st.session_state: st.session_state.app_mode = "Landing"
    if 'page_view' not in st.session_state: st.session_state.page_view = "Data Scouting"
    if 'selected_league' not in st.session_state: st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state: st.session_state.selected_metric = list(metric_info.keys())[0]

    # --- Page Routing ---
    if st.session_state.app_mode == "Landing":
        display_landing_page()
    else:
        st.markdown("""<style>div[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Data Scouting", use_container_width=True, type="primary" if st.session_state.page_view == "Data Scouting" else "secondary"):
                st.session_state.page_view = "Data Scouting"; st.rerun()
        with col2:
            if st.button("🎯 Match Analysis", use_container_width=True, type="primary" if st.session_state.page_view == "Match Analysis" else "secondary"):
                st.session_state.page_view = "Match Analysis"; st.rerun()
        with col3:
            if st.button("👤 Player Profiling", use_container_width=True, type="primary" if st.session_state.page_view == "Player Profiling" else "secondary"):
                st.session_state.page_view = "Player Profiling"; st.rerun()
        with col4:
            if st.button("⛳ Corners", use_container_width=True, type="primary" if st.session_state.page_view == "Corners" else "secondary"):
                st.session_state.page_view = "Corners"; st.rerun()
        st.markdown("---")

        if st.session_state.page_view == "Data Scouting":
            with st.expander("Show Metric Leaderboard Filters", expanded=True):
                st.selectbox("Select Metric:", list(metric_info.keys()), key='selected_metric')
        elif st.session_state.page_view == "Corners":
            with st.expander("Show Corner Analysis Filters", expanded=True):
                try:
                    leagues_with_total = ["Total"] + list(data_config.keys())
                    temp_df_full = load_data(resource_path(os.path.join("data", "WSL_corners.csv")))
                    df_corners = temp_df_full[temp_df_full['Type_of_play'].str.strip().str.lower() == 'fromcorner'].copy()
                    teams, players = ["All"] + sorted(df_corners['TeamId'].unique().tolist()), ["All"] + sorted(df_corners['PlayerId'].unique().tolist())
                    is_goal_options, (min_time, max_time) = ["All", True, False], (int(df_corners['timeMin'].min()), int(df_corners['timeMin'].max()))
                    game_states = ["All"] + sorted(df_corners['GameState'].unique().tolist()) if 'GameState' in df_corners.columns else ["All"]
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.selectbox("Select League:", leagues_with_total, key='corner_league_selection')
                        st.selectbox("Filter by Team:", teams, key='corner_team')
                    with c2:
                        st.selectbox("Filter by Player:", players, key='corner_player')
                        st.selectbox("Filter by Game State:", game_states, key='corner_gamestate')
                    with c3:
                        st.selectbox("Filter by Goal:", is_goal_options, key='corner_isgoal', format_func=lambda x: "All" if x=="All" else ("Yes" if x else "No"))
                        st.slider("Filter by Time (minutes):", min_time, max_time, (min_time, max_time), key='corner_time')
                except Exception: st.warning("Could not load corner filter options.")
        
        if st.session_state.page_view not in ["Data Scouting", "Corners", "Player Profiling"]:
            st.markdown("---")
        
        if st.session_state.page_view == "Data Scouting": display_data_scouting_page(data_config, metric_info)
        elif st.session_state.page_view == "Match Analysis": display_match_analysis_page()
        elif st.session_state.page_view == "Player Profiling": display_player_profiling_page()
        elif st.session_state.page_view == "Corners": display_corners_page(data_config)

        st.markdown("---")
        st.markdown(f"© {datetime.now().year} She Plots FC x Outswinger FC | All rights reserved.")

if __name__ == "__main__":
    main()