import streamlit as st
import pandas as pd
import numpy as np

def generate_mock_data(league):
    """Generates mock player data for a given league."""
    if league == "NWSL":
        teams = ["Angel City FC", "Chicago Red Stars", "Houston Dash", "KC Current", "NJ/NY Gotham FC", 
                 "NC Courage", "Orlando Pride", "Portland Thorns", "OL Reign", "Racing Louisville", 
                 "San Diego Wave", "Washington Spirit"]
        players = [
            "Sophia Smith", "Alex Morgan", "Mallory Pugh", "Rose Lavelle", "Trinity Rodman", 
            "Megan Rapinoe", "Lynn Williams", "Crystal Dunn", "Naomi Girma", "Alyssa Thompson",
            "Debinha", "Kerolin Nicoli", "Sam Staab", "Ashley Sanchez", "Diana Ordóñez"
        ]
    elif league == "WSL":
        teams = ["Arsenal", "Aston Villa", "Brighton & Hove Albion", "Chelsea", "Everton", 
                 "Leicester City", "Liverpool", "Manchester City", "Manchester United", 
                 "Reading", "Tottenham Hotspur", "West Ham United"]
        players = [
            "Sam Kerr", "Beth Mead", "Vivianne Miedema", "Lauren Hemp", "Chloe Kelly",
            "Khadija Shaw", "Fridolina Rolfö", "Guro Reiten", "Ella Toone", "Alessia Russo",
            "Leah Williamson", "Millie Bright", "Mary Earps", "Rachel Daly", "Ona Batlle"
        ]
    elif league == "Frauen-Bundesliga":
        teams = ["VfL Wolfsburg", "Bayern Munich", "Eintracht Frankfurt", "TSG Hoffenheim", "SC Freiburg", 
                 "SGS Essen", "Bayer 04 Leverkusen", "1. FC Köln", "Werder Bremen", "MSV Duisburg"]
        players = [
            "Alexandra Popp", "Lina Magull", "Lea Schüller", "Laura Freigang", "Jule Brand",
            "Lena Oberdorf", "Klara Bühl", "Linda Dallmann", "Nicole Billa", "Tabea Waßmuth",
            "Merle Frohms", "Giulia Gwinn", "Sydney Lohmann", "Lara Prašnikar", "Ewa Pajor"
        ]
    else: # WSL 2 (FA Women's Championship)
        teams = ["London City Lionesses", "Bristol City", "Southampton", "Birmingham City", "Durham",
                 "Crystal Palace", "Sheffield United", "Charlton Athletic", "Lewes", "Sunderland"]
        players = [
            "Melissa Johnson", "Katie Wilkinson", "Jasmine Matthews", "Charlie Wellings", "Rio Hardy",
            "Molly Pike", "Ava Kuyken", "Mia Ross", "Lucy Quinn", "Beth Hepple",
            "Emily Kraft", "Sarah Ewens", "Abigail Harrison", "Jade Pennock", "Courtney Sweetman-Kirk"
        ]
    
    num_players = len(players)
    data = {
        'Player': players,
        'Team': np.random.choice(teams, size=num_players),
        'Minutes Played': np.random.randint(500, 2000, size=num_players),
        'xG': np.round(np.random.uniform(0.1, 0.9, size=num_players), 2),
        'xAG': np.round(np.random.uniform(0.1, 0.8, size=num_players), 2),
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

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Soccer Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

    # --- Sidebar ---
    st.sidebar.title("🎙️ The Analyst's Booth")
    st.sidebar.image("https://placehold.co/400x200/2d3748/ffffff?text=SOCCER+ANALYSIS", use_column_width=True)
    
    leagues = ["NWSL", "WSL", "Frauen-Bundesliga", "WSL 2"]
    selected_league = st.sidebar.selectbox("Select a League", leagues)
    
    st.sidebar.header("Metric Definitions")
    metric_info = get_metric_info()
    for metric, desc in metric_info.items():
        with st.sidebar.expander(metric):
            st.write(desc)

    # --- Main Page ---
    st.title(f"📊 {selected_league} - Advanced Metrics Leaderboard")
    st.markdown("### Deep Dive into Player Performance")
    st.markdown("Welcome to the film room! Here's the inside scoop on the league's top performers. We're breaking down the advanced numbers to see who's really making an impact on the field. All stats are normalized per 90 minutes.")

    # Generate data for the selected league
    df = generate_mock_data(selected_league)

    # --- Metrics Display ---
    metrics_to_display = {
        'Top Offensive Threats': ['xG', 'xAG', 'VAEP'],
        'Playmakers & Progression': ['xT', 'Expected Cross', 'Dribble Success Rate (%)'],
        'Defensive Impact': ['Expected Disruption', 'Expected Shot Danger']
    }

    for category, metrics in metrics_to_display.items():
        st.header(f"📈 {category}")
        
        # Create columns for each metric in the category
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                # Find the full metric name from the info dictionary for a better title
                display_name = [name for name in metric_info.keys() if name.startswith(metric.split(' ')[0])][0]
                
                st.subheader(display_name)
                
                # Sort dataframe by the current metric
                sorted_df = df[['Player', 'Team', metric]].sort_values(by=metric, ascending=False).reset_index(drop=True)
                sorted_df.index = sorted_df.index + 1 # Start index from 1 for ranking
                
                # Display the dataframe, highlighting the max value
                st.dataframe(
                    sorted_df.head(10),
                    use_container_width=True,
                    # Apply styling to highlight the top player in the table
                    column_config={
                        metric: st.column_config.NumberColumn(
                            format="%.2f",
                        )
                    }
                )

if __name__ == "__main__":
    main()
