import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import math
from mplsoccer import PyPizza, add_image, FontManager
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(df):
    df = df[df['Pos'].str.contains('MF')]
    df = df[df['Min'] > 450]
    df['NpxG+xAG per 90'] = df['npxGPer90'] + df['xAGPer90']
    df["completion%"] = (df["PassesCompletedPer90"] / df["PassesAttemptedPer90"]) * 100
    df = df[['Player', 'Squad', 'G+A', 'GoalsPer90', 'npxGPer90', 'Sh/90', 'AssistsPer90', 'xAGPer90', 'NpxG+xAG per 90', 'SCAPer90',
             'PassesAttemptedPer90', 'completion%',
             'ProgPassesPer90', 'ProgCarriesPer90', 'SuccDrbPer90', 'Att3rdTouchPer90', 'ProgPassesRecPer90',
             'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90', 'AerialWinsPer90']]

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

    df = df.drop(['G+A', 'GoalsPer90', 'npxGPer90', 'Sh/90', 'AssistsPer90', 'xAGPer90', 'NpxG+xAG per 90', 'SCAPer90',
                  'PassesAttemptedPer90', 'completion%', 'ProgPassesPer90', 'ProgCarriesPer90', 'SuccDrbPer90',
                  'Att3rdTouchPer90', 'ProgPassesRecPer90', 'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90',
                  'AerialWinsPer90'], axis=1)
    return df

def generate_radar_chart(df, player_name, squad_name):
    params = list(df.columns[2:])
    player = df.loc[df['Player'] == player_name].reset_index().loc[0, params].tolist()
    values = [math.floor(stats.percentileofscore(df[param].fillna(0), val)) for param, val in zip(params, player)]
    values = [99 if val == 100 else val for val in values]

    baker = PyPizza(
        params=params,
        straight_line_color="white",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-.")
    slice_colors = ["#008000"] * 7 + ["#FF9300"] * 7 + ["#D70232"] * 5
    text_colors = ["#000000"] * 8 + ["white"] * 5

    fig, ax = baker.make_pizza(
        values, figsize=(8, 8.5), param_location=110, color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor="white", zorder=2, linewidth=1),
        kwargs_params=dict(color="white", fontsize=12, va="center", alpha=.5),
        kwargs_values=dict(color="white", fontsize=12, zorder=3,
                           bbox=dict(edgecolor="white", facecolor="#242424", boxstyle="round,pad=0.2", lw=1)))
    fig.text(0.515, 0.97, f"{player_name} - {squad_name}\n\n", size=25, ha="center", color="white")
    fig.text(0.515, 0.932, "Per 90 Percentile Rank T5 EU\n\n", size=15, ha="center", color="white")
    fig.text(0.09, 0.005, f"Minimal 450 minutes \ midfielders", color="white")

    notes = '@Lambertsmarc'
    CREDIT_1 = "by Marc Lamberts | @ShePlotsFC \ndata: Opta\nAll units per 90"
    CREDIT_2 = '@lambertsmarc'
    CREDIT_2 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"
    CREDIT_3 = "by Alina Ruprecht | @alina_rxp"
    fig.text(0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9, color="white", ha="right")
    fig.text(0.34, 0.935, "Attacking      Progression     Defending                ", size=14, color="white")

    fig.patches.extend([
        plt.Rectangle((0.31, 0.9325), 0.025, 0.021, fill=True, color="#008000", transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.475, 0.9325), 0.025, 0.021, fill=True, color="#ff9300", transform=fig.transFigure, figure=fig),
        plt.Rectangle((0.652, 0.9325), 0.025, 0.021, fill=True, color="#d70232", transform=fig.transFigure, figure=fig),
    ])

    return fig

# Load data
df = load_data("/Users/marclambertes/T5 Women.xlsx")
df = preprocess_data(df)

st.title("Player Radar Chart")
st.sidebar.header("Select Options")

# Team selection
team_selected = st.sidebar.selectbox("Select Team", sorted(df['Squad'].unique()))
# Player selection
player_selected = st.sidebar.selectbox("Select Player", sorted(df[df['Squad'] == team_selected]['Player'].unique()))

if st.sidebar.button("Generate Radar Chart"):
    squad_name = df.loc[df['Player'] == player_selected, 'Squad'].iloc[0]
    fig = generate_radar_chart(df, player_selected, squad_name)
    
    # Display radar chart
    st.pyplot(fig)

    # Save radar chart
    file_name = f"{player_selected} - {squad_name}.png"
    fig.savefig(file_name, dpi=750, bbox_inches='tight', facecolor='#242424')

    # Display download button
    with open(file_name, "rb") as file:
        btn = st.download_button(
            label="Download Radar Chart",
            data=file,
            file_name=file_name,
            mime="image/png"
        )
