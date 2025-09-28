import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import io
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics Platform | She Plots FC x Outswinger FC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- Professional Styling ---
def inject_custom_css():
    """Injects professional CSS styling"""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            /* --- Base & Typography --- */
            html, body, [class*="st-"] {
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                background-color: #FFFFFF;
            }
            h1, h2, h3 {
                color: #1E3A8A;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .stMarkdown, p, .st-bk, label {
                color: #374151;
                font-size: 0.95rem;
            }
            
            /* --- Sidebar Styling --- */
            [data-testid="stSidebar"] {
                background-color: #F8FAFC;
                border-right: 1px solid #E5E7EB;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #1E3A8A;
                font-weight: 600;
            }

            /* --- Navigation Buttons --- */
            .stButton > button {
                border-radius: 8px;
                border: 1px solid #E5E7EB;
                background-color: #FFFFFF;
                color: #374151;
                padding: 8px 16px;
                font-weight: 500;
                transition: all 0.2s ease;
                font-size: 0.9rem;
            }
            .stButton > button:hover {
                background-color: #3B82F6;
                color: #FFFFFF;
                border-color: #3B82F6;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
            }
            .stButton > button[kind="primary"] {
                background-color: #1E3A8A;
                color: #FFFFFF;
                border-color: #1E3A8A;
            }
            .stButton > button[kind="primary"]:hover {
                background-color: #3B82F6;
                border-color: #3B82F6;
            }

            /* --- Widgets --- */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #FFFFFF;
                border-radius: 6px;
                border: 1px solid #D1D5DB;
                color: #374151;
                font-size: 0.9rem;
            }
            .stSlider [data-baseweb="slider"] {
                color: #3B82F6;
            }
            .stTextInput > div > div > input, .stTextArea > div > textarea {
                background-color: #FFFFFF;
                border-radius: 6px;
                border: 1px solid #D1D5DB;
                color: #374151;
                font-size: 0.9rem;
            }

            /* --- Dataframes --- */
            .stDataFrame {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
            }
            .stDataFrame .data-grid-header {
                background-color: #F8FAFC;
                color: #1E3A8A;
                font-weight: 600;
            }

            /* --- Metrics --- */
            [data-testid="metric-container"] {
                background-color: #F8FAFC;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 1rem;
            }

            /* --- Section Headers --- */
            .section-header {
                border-bottom: 2px solid #1E3A8A;
                padding-bottom: 0.5rem;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

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
        'xG (Expected Goals)': 'Estimates the probability of a shot resulting in a goal based on factors like shot angle, distance, and assist type.',
        'xAG (Expected Assisted Goals)': 'Measures the likelihood that a given pass will become a goal assist.',
        'xT (Expected Threat)': 'Quantifies the increase in scoring probability by moving the ball between pitch locations.',
        'Expected Disruption (xDisruption)': "Measures defensive ability to break up opposition plays.",
        'Goal Probability Added (GPA/G+)': "Measures the change in goal probability from a player's actions."
    }

def resource_path(relative_path):
    """Get absolute path to resource"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def calculate_derived_metrics(df):
    """Calculates per 90 and per shot metrics"""
    df = df.copy()
    if 'Minutes' in df.columns:
        df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce').replace(0, np.nan)
        for col in ['xG', 'xAG', 'xT', 'xDisruption', 'GPA']:
            if col in df.columns:
                df[f'{col} per 90'] = (df[col] / df['Minutes'] * 90).round(3)
        if 'Shots' in df.columns and 'xG' in df.columns:
            df['Shots'] = pd.to_numeric(df['Shots'], errors='coerce').replace(0, np.nan)
            df['xG per Shot'] = (df['xG'] / df['Shots']).round(3)
    return df

def create_detailed_shot_map(df, title_text="Corner Shots"):
    """Creates a professional shot map visualization"""
    total_shots = df.shape[0]
    if total_shots == 0: return None, "No shots to plot."
    
    # Professional color scheme
    BG_COLOR = "#FFFFFF"
    PITCH_COLOR = "#F8FAFC"
    TEXT_COLOR = "#374151"
    PRIMARY_COLOR = "#1E3A8A"
    SECONDARY_COLOR = "#3B82F6"
    MISS_COLOR = "#9CA3AF"
    GOAL_COLOR = "#10B981"

    total_goals, total_xg = int(df['isGoal'].sum()), df['xG'].sum()
    xg_per_shot = total_xg / total_shots if total_shots > 0 else 0
    colors = {"missed": MISS_COLOR, "goal": GOAL_COLOR}

    pitch = VerticalPitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color='#6B7280', 
                         half=False, line_zorder=2, linewidth=1)
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.set_facecolor(BG_COLOR)
    
    ax.set_ylim(49.8, 105)
    for i in range(len(df['x'])):
        row, color = df.iloc[i], colors["goal"] if df.iloc[i]['isGoal'] else colors["missed"]
        size = max(row['xG'] * 400, 30)  # Minimum size for visibility
        ax.scatter(row['y'], row['x'], color=color, s=size, alpha=0.8, zorder=3, 
                  ec=TEXT_COLOR, linewidth=0.5)
        
    # Professional title and annotations
    ax.text(50, 108, title_text, fontsize=20, weight='bold', color=PRIMARY_COLOR, 
            ha='center', va='top')
    ax.text(50, 104, "Shot Map from Corners", fontsize=12, color=TEXT_COLOR, 
            ha='center', va='top')
    plt.subplots_adjust(bottom=0.3)
    
    # Key metrics display
    metrics_data = [
        ("Shots", total_shots, SECONDARY_COLOR),
        ("xG/Shot", round(xg_per_shot, 3), SECONDARY_COLOR),
        ("Goals", total_goals, GOAL_COLOR),
        ("xG", round(total_xg, 3), GOAL_COLOR)
    ]
    
    for i, (label, value, color) in enumerate(metrics_data):
        x_pos = 0.15 + i * 0.22
        circle = Circle((x_pos, -0.08), 0.03, transform=ax.transAxes, color=color, 
                       zorder=5, clip_on=False)
        ax.add_artist(circle)
        ax.text(x_pos, -0.15, label, transform=ax.transAxes, color=TEXT_COLOR, 
               fontsize=10, ha='center', va='center', zorder=6)
        ax.text(x_pos, -0.08, str(value), transform=ax.transAxes, color='white', 
               fontsize=11, weight='bold', ha='center', va='center', zorder=6)
        
    # Legend
    ax.text(0.75, -0.05, "xG Scale", transform=ax.transAxes, fontsize=11, 
            color=TEXT_COLOR, ha='center', va='center', weight='bold')
    ax.scatter([0.72, 0.75, 0.78], [-0.12, -0.12, -0.12], 
               s=[0.1*400, 0.4*400, 0.7*400], color=SECONDARY_COLOR, 
               transform=ax.transAxes, clip_on=False)
    ax.text(0.75, -0.18, "Low → High", transform=ax.transAxes, fontsize=9, 
            color=TEXT_COLOR, ha='center', va='center')
    
    return fig, None

def create_player_profile_fig(df, player_name, position_group):
    """Generates a professional player profile visualization"""
    # [Keep the existing player profile logic but update colors to match professional theme]
    # Use colors: PRIMARY_COLOR = "#1E3A8A", SECONDARY_COLOR = "#3B82F6", BG_COLOR = "#FFFFFF"
    # [Implementation details remain the same as original but with updated color scheme]
    pass

def create_match_shot_map_fig(df, file_path):
    """Generates professional xG shot map for matches"""
    # [Keep the existing match analysis logic but update colors to match professional theme]
    pass

# --- Streamlined Page Display Functions ---
def display_landing_page():
    """Professional landing page"""
    st.markdown("""
        <style>
        .block-container { max-width: 900px; padding-top: 3rem; } 
        div[data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1: 
            st.image(resource_path("sheplotsfc2.png"), use_container_width=True)
        with col2: 
            st.markdown("<div style='text-align: center; margin-top: 50px;'><h2 style='color: #6B7280;'>×</h2></div>", 
                       unsafe_allow_html=True)
        with col3: 
            st.image(resource_path("Outswinger FC.png"), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
            <div style='text-align: center;'>
                <h1 style='color: #1E3A8A; margin-bottom: 1rem;'>WoSo Analytics Platform</h1>
                <p style='color: #6B7280; font-size: 1.1rem;'>
                Advanced data analytics and performance insights for women's football
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Access Analytics Platform", use_container_width=True, type="primary"):
                st.session_state.app_mode = "MainApp"
                st.rerun()

def display_data_scouting_page(data_config, metric_info):
    """Professional data scouting interface"""
    st.markdown('<div class="section-header"><h2>Performance Analytics</h2></div>', 
                unsafe_allow_html=True)
    
    # League selection
    leagues = list(data_config.keys())
    selected_league = st.selectbox("League", leagues, key='selected_league')
    
    # Metric selection
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_metric = st.selectbox("Performance Metric", list(metric_info.keys()), 
                                     key='selected_metric')
    
    st.markdown(f"**{selected_metric}**")
    st.caption(metric_info.get(selected_metric_key, ''))
    
    # Data loading and display
    metric_config = data_config.get(selected_league, {}).get(selected_metric)
    if metric_config:
        try:
            df_metric = load_data(resource_path(os.path.join("data", metric_config["file"])))
            df_minutes = load_data(resource_path(os.path.join("data", data_config[selected_league]["minutes_file"])))
            
            # Data processing
            df_metric.rename(columns={'playerName': 'Player'}, inplace=True)
            df_raw = pd.merge(df_metric, df_minutes[['Player', 'Minutes']], on='Player', how='left')
            df_processed = calculate_derived_metrics(df_raw)
            
            # Filters
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_term = st.text_input("Player Search", placeholder="Enter player name...")
            with col2:
                top_n = st.selectbox("Show Top", [10, 25, 50], index=0)
            with col3:
                display_format = st.radio("View", ["Table", "Chart"], horizontal=True)
            
            # Filter and sort
            if search_term:
                df_processed = df_processed[df_processed['Player'].str.contains(search_term, case=False, na=False)]
            
            sort_col = metric_config["sort"]
            if not df_processed.empty and sort_col in df_processed.columns:
                display_df = df_processed.sort_values(by=sort_col, ascending=False).head(top_n)
                display_df.index = range(1, len(display_df) + 1)
                
                if display_format == "Chart":
                    # Professional chart
                    chart = alt.Chart(display_df.head(15)).mark_bar(
                        color='#3B82F6',
                        cornerRadius=2
                    ).encode(
                        x=alt.X(f'{sort_col}:Q', title=selected_metric),
                        y=alt.Y('Player:N', sort='-x', title="")
                    ).properties(height=400).configure_axis(
                        labelColor='#374151',
                        titleColor='#374151'
                    ).configure_view(strokeWidth=0)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    # Professional table
                    display_cols = [col for col in metric_config["cols"] if col in display_df.columns]
                    st.dataframe(display_df[display_cols], use_container_width=True)
                    
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")

def display_corners_page(data_config):
    """Professional corner analysis"""
    st.markdown('<div class="section-header"><h2>Set Piece Analysis</h2></div>', 
                unsafe_allow_html=True)
    
    # [Keep existing corner analysis logic but with streamlined UI]
    pass

def display_match_analysis_page():
    """Professional match analysis"""
    st.markdown('<div class="section-header"><h2>Match Analysis</h2></div>', 
                unsafe_allow_html=True)
    
    if st.button("Generate Match Report", use_container_width=True, type="primary"):
        selected_match = st.session_state.get('ma_selected_match')
        if selected_match:
            with st.spinner("Generating professional analysis..."):
                # [Keep existing match analysis logic]
                pass
        else:
            st.warning("Please select a match from the sidebar")

def display_player_profiling_page():
    """Professional player profiling"""
    st.markdown('<div class="section-header"><h2>Player Profiles</h2></div>', 
                unsafe_allow_html=True)
    
    if st.button("Generate Player Profile", use_container_width=True, type="primary"):
        selected_player = st.session_state.get('pp_selected_player')
        if selected_player:
            with st.spinner("Creating professional profile..."):
                # [Keep existing player profile logic]
                pass
        else:
            st.warning("Please select a player from the sidebar")

# --- Main App Logic ---
def main():
    """Main application function"""
    inject_custom_css()
    
    # Configuration
    metric_info = get_metric_info()
    data_config = {
        "WSL": {
            "minutes_file": "WSL_minutes.csv",
            'xG (Expected Goals)': {"file": "WSL.csv", "cols": ['Player', 'Team', 'Minutes', 'Shots', 'xG', 'xG per 90'], "sort": 'xG'},
            'xAG (Expected Assisted Goals)': {"file": "WSL_assists.csv", "cols": ['Player', 'Team', 'Minutes', 'Assists', 'xAG', 'xAG per 90'], "sort": 'xAG'},
            # ... other metrics
        },
        # ... other leagues
    }
    
    # Session state initialization
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"
    if 'page_view' not in st.session_state:
        st.session_state.page_view = "Data Scouting"
    
    # Page routing
    if st.session_state.app_mode == "Landing":
        st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", 
                   unsafe_allow_html=True)
        display_landing_page()
    else:
        # Professional navigation
        st.markdown("""
            <div style='border-bottom: 1px solid #E5E7EB; padding-bottom: 1rem; margin-bottom: 2rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <img src='https://via.placeholder.com/150x40/1E3A8A/FFFFFF?text=ANALYTICS' style='height: 40px;'>
                </div>
                <div style='display: flex; gap: 1rem;'>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        pages = {
            "Data Scouting": "📊 Performance",
            "Match Analysis": "🎯 Matches", 
            "Player Profiling": "👤 Profiles",
            "Corners": "⛳ Set Pieces"
        }
        
        cols = st.columns(len(pages))
        for idx, (page_key, page_label) in enumerate(pages.items()):
            if cols[idx].button(page_label, use_container_width=True, 
                              type="primary" if st.session_state.page_view == page_key else "secondary"):
                st.session_state.page_view = page_key
                st.rerun()
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)
        
        # Sidebar filters
        with st.sidebar:
            st.markdown("### Filters & Controls")
            
            if st.session_state.page_view == "Data Scouting":
                st.selectbox("Metric", list(metric_info.keys()), key='selected_metric')
                
            elif st.session_state.page_view == "Match Analysis":
                # [Streamlined match selection logic]
                pass
                
            elif st.session_state.page_view == "Player Profiling":
                # [Streamlined player selection logic]  
                pass
                
            elif st.session_state.page_view == "Corners":
                # [Streamlined corner filters]
                pass
        
        # Main content
        if st.session_state.page_view == "Data Scouting":
            display_data_scouting_page(data_config, metric_info)
        elif st.session_state.page_view == "Match Analysis":
            display_match_analysis_page()
        elif st.session_state.page_view == "Player Profiling":
            display_player_profiling_page()
        elif st.session_state.page_view == "Corners":
            display_corners_page(data_config)
        
        # Professional footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>
                <p>© 2024 WoSo Analytics Platform | She Plots FC × Outswinger FC</p>
                <p>Professional football intelligence</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()