import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import PyPizza
import matplotlib.pyplot as plt
from PIL import Image  # To load the logo

# Load your data
@st.cache_data
def load_data():
    df = pd.read_excel("T5 Women.xlsx")
    # [data loading and processing code here, same as before]

# Define a function to calculate percentile ranks (without decimals)
def percentile_rank(data, score):
    # [percentile calculation code here, same as before]

# Define a function to generate the radar chart with a logo in the middle
def generate_radar_chart(params, values, title, subtitle, logo_path):
    num_params = len(params)
    slice_colors = ["#008000" if i < num_params // 3 else "#FF9300" if i < 2 * num_params // 3 else "#D70232" for i in range(num_params)]
    
    baker = PyPizza(
        params=params,
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-.",
        inner_circle_size=10
    )
    
    fig, ax = baker.make_pizza(
        values, figsize=(8, 8.5), param_location=110, color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor="black", zorder=2, linewidth=1),
        kwargs_params=dict(color="black", fontsize=12, va="center", alpha=.5),
        kwargs_values=dict(color="black", fontsize=12, zorder=3,
                           bbox=dict(edgecolor="black", facecolor="#e5e5e5", boxstyle="round,pad=0.2", lw=1))
    )

    # Add title and subtitle
    fig.text(0.515, 0.97, title, size=25, ha="center", color="black")
    fig.text(0.515, 0.932, subtitle, size=15, ha="center", color="black")
    fig.text(0.515, -0.05, "Marc Lamberts - Outswinger FC - @lambertsmarc/@ShePlotsFC", size=12, ha="center", color="black")
    
    # Add logo in the middle
    logo = Image.open(logo_path)
    ax.figure.figimage(logo, xo=350, yo=220, alpha=0.6, zorder=5)  # Adjust xo and yo to center

    return fig

# Streamlit App
st.title("Outswinger FC Analysis App")
st.sidebar.header("Navigation")

# Navigation
page = st.sidebar.radio("Go to", ("Welcome", "Player Analysis"))

if page == "Welcome":
    st.write("""
    # Welcome to the Outswinger FC Analysis App
    This app provides data for women's football in the Top-5 European leagues and others.
    The app was last updated on 31-10-2024.
    Please credit my work when using this in public articles, podcasts, videos, or other forms of media.
    Marc Lamberts - @lambertsmarc on X/Twitter. - marclambertsanalysis@gmail.com
    """)
    st.write("Choose an option from the sidebar to get started.")

elif page == "Player Analysis":
    st.header("Player Radar Chart")
    st.sidebar.header("Select Options")
    
    df = load_data()
    
    # League selection with 'All' option
    league_options = ['All'] + sorted(df['Comp'].unique())
    league_selected = st.sidebar.selectbox("Select League", league_options)
    
    # Position selection with 'All' option
    position_options = ['All'] + sorted(df['Pos'].unique())
    position_selected = st.sidebar.selectbox("Select Position", position_options)
    
    # Minimum minutes selection
    min_minutes_options = [180, 300, 450, 600, 750, 900]
    min_minutes = st.sidebar.selectbox("Select Minimum Minutes", min_minutes_options)
    
    # Apply filters
    if league_selected == 'All':
        league_filter = df['Comp'].notna()
    else:
        league_filter = df['Comp'] == league_selected
    
    if position_selected == 'All':
        position_filter = df['Pos'].notna()
    else:
        position_filter = df['Pos'].str.contains(position_selected, na=False)
    
    filtered_players = df[league_filter & position_filter & (df['Min'] > min_minutes)]
    
    # Team selection
    team_options = sorted(df[league_filter]['Squad'].unique())
    team_selected = st.sidebar.selectbox("Select Team", team_options)
    
    # Player selection
    player_options = sorted(filtered_players[filtered_players['Squad'] == team_selected]['Player'].unique())
    player_selected = st.sidebar.selectbox("Select Player", player_options)
    
    if st.sidebar.button("Generate Radar Chart"):
        squad_name = filtered_players.loc[filtered_players['Player'] == player_selected, 'Squad'].iloc[0]
        params = list(df.columns[5:])
        player_stats = df.loc[df['Player'] == player_selected].reset_index().loc[0, params].tolist()
        values = [percentile_rank(df[param].fillna(0).values, val) for param, val in zip(params, player_stats)]
        values = [99 if val == 100 else val for val in values]
        
        # Dynamic title
        title = f"{player_selected} - {squad_name}"
        subtitle = f"League: {league_selected} | Position: {position_selected} | Min Minutes: {min_minutes}"
        
        # Generate radar chart with logo in the center
        fig = generate_radar_chart(params, values, title, subtitle, "logo.png")
        
        # Display radar chart
        st.pyplot(fig)
        
        # Option to download the image
        file_name = f'{player_selected} - {squad_name}.png'
        plt.savefig(file_name, dpi=750, bbox_inches='tight', facecolor='#e5e5e5')
        with open(file_name, "rb") as img_file:
            st.download_button(label="Download Image", data=img_file, file_name=file_name, mime="image/png")
