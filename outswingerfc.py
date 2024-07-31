import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import PyPizza
import matplotlib.pyplot as plt

# Load your data
@st.cache_data
def load_data():
    df = pd.read_excel("T5 Women.xlsx")
    
    # Convert relevant columns to numeric
    numeric_columns = [
        'npxGPer90', 'xAGPer90', 'PassesCompletedPer90', 'PassesAttemptedPer90', 'ProgPassesPer90', 'ProgCarriesPer90',
        'SuccDrbPer90', 'Att3rdTouchPer90', 'ProgPassesRecPer90', 'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90',
        'AerialWinsPer90'
    ]
    
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid parsing will be set as NaN
    
    df['NpxG+xAG per 90'] = df['npxGPer90'] + df['xAGPer90']
    df["completion%"] = (df["PassesCompletedPer90"] / df["PassesAttemptedPer90"]) * 100
    df = df[['Player', 'Squad', 'Comp', 'Pos', 'Min', 'G+A', 'GoalsPer90', 'npxGPer90', 'Sh/90', 'AssistsPer90', 'xAGPer90', 
             'NpxG+xAG per 90', 'SCAPer90', 'PassesAttemptedPer90', 'completion%',
             'ProgPassesPer90', 'ProgCarriesPer90', 'SuccDrbPer90', 'Att3rdTouchPer90', 'ProgPassesRecPer90',
             'TklPer90', 'IntPer90', 'BlocksPer90', 'ClrPer90', 'AerialWinsPer90']]

    df['Goals'] = df['GoalsPer90']
    df['Non-penalty xG'] = df['npxGPer90']
    df['Shots'] = df['Sh/90']
    df['Assists'] = df['AssistsPer90']
    df['xG assisted'] = df['xAGPer90']
    df['NpxG+xAG'] = df['NpxG+xAG per 90']
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

# Define a function to calculate percentile ranks (without decimals)
def percentile_rank(data, score):
    data = np.array(data, dtype=float)  # Ensure data is numeric
    score = float(score)  # Ensure score is numeric
    count = len(data)
    below = np.sum(data < score)
    equal = np.sum(data == score)
    percentile = (below + 0.5 * equal) / count * 100
    return int(round(percentile))

# Define a function to generate the radar chart
def generate_radar_chart(params, values, title, subtitle):
    num_params = len(params)
    
    # Generate slice colors based on number of parameters
    slice_colors = ["#008000" if i < num_params // 3 else "#FF9300" if i < 2 * num_params // 3 else "#D70232" for i in range(num_params)]
    
    baker = PyPizza(
        params=params,
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-.",
        inner_circle_size=10  # Increase the size of the center circle
    )
    
    fig, ax = baker.make_pizza(
        values, figsize=(8, 8.5), param_location=110, color_blank_space="same",
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor="black", zorder=2, linewidth=1),
        kwargs_params=dict(color="black", fontsize=12, va="center", alpha=.5),
        kwargs_values=dict(color="black", fontsize=12, zorder=3,
                           bbox=dict(edgecolor="black", facecolor="#e5e5e5", boxstyle="round,pad=0.2", lw=1))
    )

    fig.text(0.515, 0.97, title, size=25, ha="center", color="black")
    fig.text(0.515, 0.932, subtitle, size=15, ha="center", color="black")
    fig.text(0.515, -0.05, "Marc Lamberts - Outswinger FC - @lambertsmarc", size=12, ha="center", color="black")
    return fig

# Streamlit App
st.title("Outswinger FC Analysis App")
st.sidebar.header("Navigation")

# Navigation
page = st.sidebar.radio("Go to", ("Welcome", "Player Analysis", "Team Analysis"))

if page == "Welcome":
    st.write("""
    # Welcome to the Outswinger FC Analysis App
    I made this app so that it will be more accessible for everyone interesting in data for women's football.\n
    In this app you will find the data for the Top-5 European leagues (England, Spain, Italy, Germany, France) as well as NWSL (US) and A-League (Australia).\n The app is divided into a player and team section.\n\nThis app was last updated 31-07-2024\n\nPlease credit my work when using this in public articles, podcast, videos or other forms of media.
    """)
    st.write("Choose an option from the sidebar to get started.")
    
elif page == "Player Analysis":
    st.header("Player Radar Chart")
    st.sidebar.header("Select Options")
    
    # Load data
    df = load_data()
    
    # League selection with 'All' option
    league_options = ['All'] + sorted(df['Comp'].unique())
    league_selected = st.sidebar.selectbox("Select League", league_options)
    
    # Position selection with 'All' option
    position_options = ['All'] + sorted(df['Pos'].unique())
    position_selected = st.sidebar.selectbox("Select Position", position_options)
    
    # Minimum minutes selection
    min_minutes_options = [450, 600, 750, 900]
    min_minutes = st.sidebar.selectbox("Select Minimum Minutes", min_minutes_options)
    
    # Apply filters
    if league_selected == 'All':
        league_filter = df['Comp'].notna()  # All rows if 'All' selected
    else:
        league_filter = df['Comp'] == league_selected
    
    if position_selected == 'All':
        position_filter = df['Pos'].notna()  # All rows if 'All' selected
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
        values = [99 if val == 100 else val for val in values]  # Cap at 99
        fig = generate_radar_chart(params, values, f"{player_selected} - {squad_name}", "Per 90 Percentile Rank")
        
        # Display radar chart
        st.pyplot(fig)
        
        # Option to download the image
        file_name = f'{player_selected} - {squad_name}.png'
        plt.savefig(file_name, dpi=750, bbox_inches='tight', facecolor='#e5e5e5')
        with open(file_name, "rb") as img_file:
            st.download_button(label="Download Image", data=img_file, file_name=file_name, mime="image/png")

elif page == "Team Analysis":
    st.header("Team Radar Chart")
    st.sidebar.header("Select Options")
    
    # Load data
    df = load_data()
    
    # League selection with 'All' option
    league_options = ['All'] + sorted(df['Comp'].unique())
    league_selected = st.sidebar.selectbox("Select League", league_options)
    
    # Position selection with 'All' option
    position_options = ['All'] + sorted(df['Pos'].unique())
    position_selected = st.sidebar.selectbox("Select Position", position_options)
    
    # Apply filters
    if league_selected == 'All':
        league_filter = df['Comp'].notna()  # All rows if 'All' selected
    else:
        league_filter = df['Comp'] == league_selected
    
    if position_selected == 'All':
        position_filter = df['Pos'].notna()  # All rows if 'All' selected
    else:
        position_filter = df['Pos'].str.contains(position_selected, na=False)
    
    filtered_players = df[league_filter & position_filter]
    
    # Team selection
    team_options = sorted(filtered_players['Squad'].unique())
    team_selected = st.sidebar.selectbox("Select Team", team_options)
    
    if st.sidebar.button("Generate Radar Chart"):
        if team_selected:
            # Aggregate data for the selected team
            team_data = filtered_players[filtered_players['Squad'] == team_selected].drop(columns=['Player', 'Squad', 'Comp', 'Pos', 'Min']).mean()
            params = list(filtered_players.columns[5:])
            
            # Ensure only numeric columns are included for aggregation
            numeric_params = [param for param in params if pd.api.types.is_numeric_dtype(filtered_players[param])]
            
            values = [percentile_rank(df[param].fillna(0).values, val) for param, val in zip(numeric_params, team_data)]
            values = [99 if val == 100 else val for val in values]  # Cap at 99
            
            fig = generate_radar_chart(numeric_params, values, f"{team_selected} Team", "Per 90 Percentile Rank")
            
            # Display radar chart
            st.pyplot(fig)
        
            # Option to download the image
            file_name = f'{team_selected} Team.png'
            plt.savefig(file_name, dpi=750, bbox_inches='tight', facecolor='#e5e5e5')
            with open(file_name, "rb") as img_file:
                st.download_button(label="Download Image", data=img_file, file_name=file_name, mime="image/png")
        else:
            st.error("Please select a team.")
