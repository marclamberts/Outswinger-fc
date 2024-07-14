import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load your data
@st.cache
def load_data():
    df = pd.read_excel("T5 Women.xlsx")
    df = df[df['Pos'].str.contains('MF')]
    df = df[df['Min'] > 450]
    df['NpxG+xAG per 90'] = df['npxGPer90'] + df['xAGPer90']
    df['completion%'] = (df['PassesCompletedPer90'] / df['PassesAttemptedPer90']) * 100
    return df

# Function to calculate percentile rank
def percentile_rank(data, score):
    count = len(data)
    below = np.sum(data < score)
    equal = np.sum(data == score)
    return (below + 0.5 * equal) / count * 100

# Function to create a radar chart
def create_radar_chart(player_name, values, params):
    num_vars = len(params)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='green', alpha=0.25)
    ax.set_yticklabels([])
    
    plt.xticks(angles[:-1], params, color='black', size=10)
    plt.title(player_name, size=20, color='black', weight='bold')
    plt.tight_layout()

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
    params = list(df.columns[3:])  # Assuming params start from the 4th column
    player_data = df.loc[df['Player'] == player_selected].values.flatten()[3:]

    # Calculate percentile values
    values = [percentile_rank(df[param].fillna(0).values, val) for param, val in zip(params, player_data)]
    values = [99 if val == 100 else val for val in values]

    fig = create_radar_chart(player_selected, values, params)

    # Display radar chart
    st.pyplot(fig)

    # Option to download the image
    file_name = f'{player_selected} - {squad_name}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    with open(file_name, "rb") as img_file:
        st.download_button(label="Download Image", data=img_file, file_name=file_name, mime="image/png")
