import streamlit as st
import pandas as pd
import numpy as np
import math
from mplsoccer import PyPizza
import matplotlib.pyplot as plt

# Load your data
@st.cache
def load_data():
    df = pd.read_excel("T5 Women.xlsx")
    df = df[df['Pos'].str.contains('MF')]
    df = df[df['Min'] > 450]
    df['NpxG+xAG per 90'] = df['npxGPer90'] + df['xAGPer90']
    df["completion%"] = (df["PassesCompletedPer90"] / df["PassesAttemptedPer90"]) * 100
    df = df[['Player', 'Squad', 'G+A', 'GoalsPer90', 'npxGPer90', 'Sh/90', 'AssistsPer90', 'xAGPer90',  'NpxG+xAG per 90', 'SCAPer90',
             'PassesAttemptedPer90', 'completion%',
             'ProgPassesPer90', 'ProgCarriesPer90', 'SuccDrbPer90', 'Att3rdTouchPer90', 'ProgPassesRecPer90',
             'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90', 'AerialWinsPer90'
              ]]

    df['Goals'] = df['GoalsPer90']
    df['Non-penalty xG'] = df['npxGPer90']
    df['Shots'] = df['Sh/90']
    df['Assists'] = df['AssistsPer90']
    df['xG assisted'] = df['xAGPer90']
    df['NpxG+xAG '] = df['NpxG+xAG per 90']
    df['SCA'] = df['SCAPer90']
    df['Passes'] = df['PassesAttemptedPer90']
    df['Pass%'] = df['completion%']
    df['Prog Pass'] = df['ProgPassesPer90']
    df['Prog Carries'] = df['ProgCarriesPer90']
    df['Dribble%'] = df['SuccDrbPer90']
    df['Final 3rd touch'] = df['Att3rdTouchPer90']
    df['Prog pass rec'] = df['ProgPassesRecPer90']
    df['Tackles'] = df['TklPer90']
    df['Interceptions'] = df['IntPer90']
    df['Blocks'] = df['BlocksPer90']
    df['Cleared'] = df['ClrPer90']
    df['Aerial%'] = df['AerialWinsPer90']

    df = df.drop(['G+A', 'GoalsPer90', 'npxGPer90', 'Sh/90', 'AssistsPer90', 'xAGPer90',  'NpxG+xAG per 90', 'SCAPer90',
             'PassesAttemptedPer90', 'completion%',
             'ProgPassesPer90', 'ProgCarriesPer90', 'SuccDrbPer90', 'Att3rdTouchPer90', 'ProgPassesRecPer90',
             'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90', 'AerialWinsPer90'], axis=1)
    return df

# Define a function to calculate percentile ranks (without decimals)
def percentile_rank(data, score):
    count = len(data)
    below = np.sum(data < score)
    equal = np.sum(data == score)
    percentile = (below + 0.5 * equal) / count * 100
    return int(round(percentile))  # Convert to integer

# Define a function to generate the radar chart
def generate_radar_chart(df, player_name, squad_name):
    params = list(df.columns[2:])
    player = df.loc[df['Player'] == player_name].reset_index().loc[0, params].tolist()
    values = [percentile_rank(df[param].fillna(0).values, val) for param, val in zip(params, player)]
    values = [99 if val == 100 else val for val in values]  # Cap at 99

    baker = PyPizza(
        params=params,
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-."
    )
    slice_colors = ["#008000"] * 7 + ["#FF9300"] * 7 + ["#D70232"] * 5

    fig, ax = baker.make_pizza(
        values, figsize=(8, 8.5), param_location=110, color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor="black", zorder=2, linewidth=1),
        kwargs_params=dict(color="white", fontsize=12, va="center", alpha=.5),
        kwargs_values=dict(color="white", fontsize=12, zorder=3,
                           bbox=dict(edgecolor="white", facecolor="#e5e5e5", boxstyle="round,pad=0.2", lw=1))
    )
    fig.text(0.515, 0.97, f"{player_name} - {squad_name}\n\n", size=25, ha="center", color="black")
    fig.text(0.515, 0.932, "Per 90 Percentile Rank T5 EU\n\n", size=15, ha="center", color="black")
    fig.text(0.09, 0.005, f"Minimal 450 minutes", color="black")
    fig.text(0.75, 0.005, f"Marc Lamberts - Outswinger FC" , color="black")
    return fig

# Streamlit App
st.title("Player Radar Chart")
st.sidebar.header("Select Options")

# Load data
df = load_data()

# Team selection
team_selected = st.sidebar.selectbox("Select Team", sorted(df['Squad'].unique()))
# Player selection
player_selected = st.sidebar.selectbox("Select Player", sorted(df[df['Squad'] == team_selected]['Player'].unique()))

if st.sidebar.button("Generate Radar Chart"):
    squad_name = df.loc[df['Player'] == player_selected, 'Squad'].iloc[0]
    fig = generate_radar_chart(df, player_selected, squad_name)
    
    # Display radar chart
    st.pyplot(fig)

    # Option to download the image
    file_name = f'{player_selected} - {squad_name}.png'
    plt.savefig(file_name, dpi=750, bbox_inches='tight', facecolor='#e5e5e5')
    with open(file_name, "rb") as img_file:
        st.download_button(label="Download Image", data=img_file, file_name=file_name, mime="image/png")
