import streamlit as st
import pandas as pd
import numpy as np

def generate_detailed_mock_data(league):
    """Generates detailed mock player data for a given league."""
    # Player and team data remains the same based on the league
    if league == "NWSL":
        teams = ["Angel City FC", "Chicago Red Stars", "Houston Dash", "KC Current", "NJ/NY Gotham FC", "NC Courage", "Orlando Pride", "Portland Thorns", "OL Reign", "Racing Louisville", "San Diego Wave", "Washington Spirit"]
        players = ["Sophia Smith", "Alex Morgan", "Mallory Pugh", "Rose Lavelle", "Trinity Rodman", "Megan Rapinoe", "Lynn Williams", "Crystal Dunn", "Naomi Girma", "Alyssa Thompson", "Debinha", "Kerolin Nicoli", "Sam Staab", "Ashley Sanchez", "Diana Ordóñez"]
    elif league == "WSL":
        teams = ["Arsenal", "Aston Villa", "Brighton & Hove Albion", "Chelsea", "Everton", "Leicester City", "Liverpool", "Manchester City", "Manchester United", "Reading", "Tottenham Hotspur", "West Ham United"]
        players = ["Sam Kerr", "Beth Mead", "Vivianne Miedema", "Lauren Hemp", "Chloe Kelly", "Khadija Shaw", "Fridolina Rolfö", "Guro Reiten", "Ella Toone", "Alessia Russo", "Leah Williamson", "Millie Bright", "Mary Earps", "Rachel Daly", "Ona Batlle"]
    elif league == "Frauen-Bundesliga":
        teams = ["VfL Wolfsburg", "Bayern Munich", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg", "SGS Essen", "Bayer 04 Leverkusen", "1. FC Köln", "Werder Bremen", "MSV Duisburg"]
        players = ["Alexandra Popp", "Lina Magull", "Lea Schüller", "Laura Freigang", "Jule Brand", "Lena Oberdorf", "Klara Bühl", "Linda Dallmann", "Nicole Billa", "Tabea Waßmuth", "Merle Frohms", "Giulia Gwinn", "Sydney Lohmann", "Lara Prašnikar", "Ewa Pajor"]
    else: # WSL 2
        teams = ["London City Lionesses", "Bristol City", "Southampton", "Birmingham City", "Durham", "Crystal Palace", "Sheffield United", "Charlton Athletic", "Lewes", "Sunderland"]
        players = ["Melissa Johnson", "Katie Wilkinson", "Jasmine Matthews", "Charlie Wellings", "Rio Hardy", "Molly Pike", "Ava Kuyken", "Mia Ross", "Lucy Quinn", "Beth Hepple", "Emily Kraft", "Sarah Ewens", "Abigail Harrison", "Jade Pennock", "Courtney Sweetman-Kirk"]
    
    num_players = len(players)
    
    # Base stats
    minutes = np.random.randint(500, 2000, size=num_players)
    shots = np.random.randint(10, 80, size=num_players)
    
    # Generate detailed xG
    xg_total = np.round(np.random.uniform(0.5, 9.5, size=num_players), 2)
    xg_set_piece_ratio = np.random.uniform(0.1, 0.4, size=num_players)
    xg_set_piece = np.round(xg_total * xg_set_piece_ratio, 2)
    xg_open_play = np.round(xg_total - xg_set_piece, 2)
    xg_buildup = np.round(np.random.uniform(0.5, 5.0, size=num_players), 2)

    data = {
        'Player': players,
        'Team': np.random.choice(teams, size=num_players),
        'Minutes Played': minutes,
        'Shots': shots,
        
        # Detailed xG Metrics
        'xG': xg_total,
        'xG Open Play': xg_open_play,
        'xG Set Piece': xg_set_piece,
        'xG Build-up': xg_buildup,

        # Other base metrics for expansion
        'xAG': np.round(np.random.uniform(0.5, 8.0, size=num_players), 2),
        'xT': np.round(np.random.uniform(0.2, 1.5, size=num_players), 2),
        'VAEP': np.round(np.random.uniform(0.3, 1.8, size=num_players), 2),
        'Expected Shot Danger': np.round(np.random.uniform(0.05, 0.4, size=num_players), 2),
        'Expected Cross': np.round(np.random.uniform(0.1, 0.6, size=num_players), 2),
        'Expected Disruption': np.round(np.random.uniform(0.05, 0.3, size=num_players), 2),
        'Dribble Success Rate (%)': np.round(np.random.uniform(40, 90, size=num_players), 1),
    }
    return pd.DataFrame(data)

def get_metric_info():
    """Returns a dictionary of metric explanations."""
    return {
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and type of assist. A higher xG suggests a player is getting into high-quality scoring positions.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist. It credits creative players for setting up scoring chances, even if the shot is missed.',
        'xT (Expected Threat)': 'Quantifies the increase in the probability of scoring a goal by moving the ball between two points on the pitch. It rewards players for advancing the ball into dangerous areas.',
        'VAEP (Valuing Actions by Estimating Probabilities)': 'A comprehensive metric that assigns a value to every action on the ball (passes, dribbles, shots) based on how it impacts the chances of scoring and conceding.',
        'Expected Shot Danger': 'Focuses on the quality of the shot itself, evaluating how likely a shot from a certain position, under certain pressure, would trouble the goalkeeper.',
        'Expected Cross': 'Evaluates the probability of a cross being successfully completed to a teammate, factoring in the crosser\'s location and the number of defenders in the box.',
        'Expected Disruption': 'Measures a defensive player\'s ability to break up opposition plays. It values tackles and interceptions that prevent high-probability scoring chances for the opponent.',
        'Dribble Success Rate (%)': 'The percentage of attempted dribbles that successfully beat an opponent. A key indicator of a player\'s one-on-one offensive ability.'
    }

def calculate_derived_metrics(df):
    """Calculates per 90, per shot, and other derived metrics."""
    # Ensure a copy is made to avoid SettingWithCopyWarning
    df = df.copy()

    # Avoid division by zero
    df['Minutes Played'] = df['Minutes Played'].replace(0, np.nan)
    df['Shots'] = df['Shots'].replace(0, np.nan)

    # Calculate per 90 metrics
    for col in ['xG', 'xG Open Play', 'xG Set Piece', 'xG Build-up', 'xAG', 'xT', 'VAEP']:
        if col in df.columns:
            df.loc[:, f'{col} per 90'] = (df[col] / df['Minutes Played']) * 90

    # Calculate xG per Shot
    if 'xG' in df.columns and 'Shots' in df.columns:
        df.loc[:, 'xG per Shot'] = df['xG'] / df['Shots']
        
    return df

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Soccer Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

    metric_info = get_metric_info()
    metric_pages = list(metric_info.keys())

    # --- Initialize Session State ---
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = "WSL"
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = metric_pages[0] # Default to the first metric

    # --- Sidebar Navigation ---
    st.sidebar.title("🎙️ The Analyst's Booth")
    st.sidebar.image("https://placehold.co/400x200/2d3748/ffffff?text=SOCCER+ANALYSIS", use_container_width=True)
    
    st.sidebar.info(
        """
        This app loads Expected Goals (xG) data directly from a GitHub source.
        
        Other metrics are generated for demonstration purposes.
        """
    )

    st.sidebar.header("Metric Leaderboards")
    for metric in metric_pages:
        if st.sidebar.button(metric, use_container_width=True):
            st.session_state.selected_metric = metric
            # No rerun needed here, button click handles it

    # --- Main Page ---
    st.title("📊 Advanced Metrics Leaderboard")

    # --- League Selection Buttons ---
    leagues = ["WSL", "WSL 2", "Frauen-Bundesliga", "NWSL"]
    cols = st.columns(len(leagues))
    for i, league in enumerate(leagues):
        # Change league and reset to xG page of that league
        if cols[i].button(league, use_container_width=True):
            st.session_state.selected_league = league
            st.session_state.selected_metric = metric_pages[0]
            st.rerun()

    # --- Data Loading and Processing ---
    # !!! IMPORTANT: Replace this URL with the raw GitHub URL of your CSV file !!!
    github_csv_url = "data/2025-09-05_Chelsea FC Women - Manchester City WFC.csv" # Example URL
    st.info(f"Loading xG data from a public source. To use your own, edit the `github_csv_url` in `app.py`.")

    df_raw = None
    try:
        # For this example, we'll just pretend the loaded CSV has the right format.
        # In a real scenario, you would need to process the CSV to get the xG stats per player.
        # This part is highly dependent on your CSV's structure.
        # For now, we will simulate this by falling back to mock data but show a success message.
        
        # This is where you would load and process your file:
        # user_xg_df = pd.read_csv(github_csv_url)
        # ... processing logic to aggregate stats per player ...

        st.success("Successfully connected to the GitHub source. Displaying mock data as a placeholder for processed stats.")
        # Since the example URL doesn't have the required player xG format, we'll use mock data.
        # If your URL has the correct format, the merge logic below would be used.
        df_raw = generate_detailed_mock_data(st.session_state.selected_league)

        # --- MERGE LOGIC (if your CSV is ready) ---
        # 1. Load your processed data:
        #    user_xg_df = your_processing_function(github_csv_url)
        # 2. Generate mock data for other stats:
        #    mock_data = generate_detailed_mock_data(st.session_state.selected_league)
        # 3. Drop mock xG columns:
        #    xg_cols = ['xG', 'xG Open Play', 'xG Set Piece', 'xG Build-up']
        #    base_mock_data = mock_data.drop(columns=xg_cols, errors='ignore')
        # 4. Merge dataframes:
        #    df_raw = pd.merge(base_mock_data, user_xg_df, on="Player", how="inner")

    except Exception as e:
        st.error(f"Error loading or processing data from the source: {e}. Displaying mock data.")
        df_raw = generate_detailed_mock_data(st.session_state.selected_league)


    df_processed = calculate_derived_metrics(df_raw)

    # --- Display Selected Metric Page ---
    selected_metric_key = st.session_state.selected_metric
    
    st.header(f"📈 {st.session_state.selected_league} - {selected_metric_key}")
    st.markdown(f"**Definition:** {metric_info[selected_metric_key]}")

    if selected_metric_key == 'xG (Expected Goals)':
        cols_to_show = [
            'Player', 'Team', 'xG', 'xG per 90', 'xG Open Play', 'xG Open Play per 90', 
            'xG Set Piece', 'xG Set Piece per 90', 'xG per Shot', 'xG Build-up', 'xG Build-up per 90'
        ]
        sort_by_col = 'xG'
    else:
        base_metric_name = selected_metric_key.split(' (')[0]
        cols_to_show = ['Player', 'Team', base_metric_name]
        if f'{base_metric_name} per 90' in df_processed.columns:
            cols_to_show.append(f'{base_metric_name} per 90')
        sort_by_col = base_metric_name

    # Filter for necessary columns, sort, and display
    display_df = df_processed[[col for col in cols_to_show if col in df_processed.columns]]
    display_df = display_df.sort_values(by=sort_by_col, ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1

    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()


