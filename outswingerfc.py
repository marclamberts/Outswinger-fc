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
from matplotlib.patches import Circle, Rectangle
import io
from scipy.stats import zscore, norm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | StatsBomb Style",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- StatsBomb Inspired Styling ---
def inject_statsbomb_css():
    """Injects StatsBomb-inspired CSS styling"""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

            /* --- Base & Typography --- */
            html, body, [class*="st-"] {
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                background-color: #0C1A2A;
                color: #FFFFFF;
            }
            h1, h2, h3, h4 {
                font-family: 'Roboto Mono', monospace;
                color: #00FF88;
                font-weight: 600;
                letter-spacing: -0.5px;
            }
            .stMarkdown, p, .st-bk, label {
                color: #B0B7C3;
            }

            /* --- Sidebar Styling --- */
            [data-testid="stSidebar"] {
                background-color: #152642;
                border-right: 1px solid #1E3A5C;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #00FF88;
                font-family: 'Roboto Mono', monospace;
            }
            [data-testid="stSidebar"] .stMarkdown {
                color: #B0B7C3;
            }

            /* --- Navigation Header --- */
            .nav-header {
                background: linear-gradient(135deg, #152642 0%, #0C1A2A 100%);
                border-bottom: 2px solid #00FF88;
                padding: 1rem 0;
                margin-bottom: 2rem;
            }

            /* --- Navigation Buttons --- */
            .nav-button {
                background: #1E3A5C !important;
                border: 1px solid #2D4A76 !important;
                color: #FFFFFF !important;
                border-radius: 4px !important;
                font-family: 'Roboto Mono', monospace !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            .nav-button:hover {
                background: #00FF88 !important;
                color: #0C1A2A !important;
                border-color: #00FF88 !important;
                transform: translateY(-2px);
            }
            .nav-button.active {
                background: #00FF88 !important;
                color: #0C1A2A !important;
                border-color: #00FF88 !important;
            }

            /* --- Primary Action Buttons --- */
            .stButton > button[kind="primary"] {
                background: #00FF88 !important;
                color: #0C1A2A !important;
                border: none !important;
                border-radius: 4px !important;
                font-family: 'Roboto Mono', monospace !important;
                font-weight: 600 !important;
                padding: 0.5rem 1.5rem !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: #00CC6A !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
            }

            /* --- Secondary Buttons --- */
            .stButton > button:not([kind="primary"]) {
                background: transparent !important;
                border: 1px solid #2D4A76 !important;
                color: #B0B7C3 !important;
                border-radius: 4px !important;
                font-family: 'Inter', sans-serif !important;
            }
            .stButton > button:not([kind="primary"]):hover {
                border-color: #00FF88 !important;
                color: #00FF88 !important;
            }

            /* --- Widgets --- */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #152642 !important;
                border: 1px solid #2D4A76 !important;
                border-radius: 4px !important;
                color: #FFFFFF !important;
                font-family: 'Inter', sans-serif !important;
            }
            .stSlider [data-baseweb="slider"] {
                color: #00FF88 !important;
            }
            .stTextInput > div > div > input {
                background-color: #152642 !important;
                border: 1px solid #2D4A76 !important;
                border-radius: 4px !important;
                color: #FFFFFF !important;
            }

            /* --- Dataframes --- */
            .stDataFrame {
                background-color: #152642 !important;
                border: 1px solid #2D4A76 !important;
                border-radius: 4px !important;
            }
            .stDataFrame .data-grid-header {
                background-color: #1E3A5C !important;
                color: #00FF88 !important;
                font-family: 'Roboto Mono', monospace !important;
                font-weight: 600 !important;
            }

            /* --- Metrics --- */
            [data-testid="metric-container"] {
                background-color: #152642;
                border: 1px solid #2D4A76;
                border-radius: 8px;
                padding: 1rem;
            }
            [data-testid="metric-container"] label {
                color: #B0B7C3 !important;
                font-family: 'Roboto Mono', monospace !important;
            }
            [data-testid="metric-container"] div {
                color: #00FF88 !important;
                font-family: 'Roboto Mono', monospace !important;
                font-weight: 600 !important;
                font-size: 1.5rem !important;
            }

            /* --- Section Headers --- */
            .section-header {
                border-bottom: 2px solid #00FF88;
                padding-bottom: 0.5rem;
                margin-bottom: 1.5rem;
                font-family: 'Roboto Mono', monospace;
            }

            /* --- Cards --- */
            .statsbomb-card {
                background: #152642;
                border: 1px solid #2D4A76;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }

            /* --- Tabs --- */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #1E3A5C;
                border-radius: 4px 4px 0 0;
                padding: 0.5rem 1rem;
                font-family: 'Roboto Mono', monospace;
                color: #B0B7C3;
            }
            .stTabs [aria-selected="true"] {
                background-color: #00FF88 !important;
                color: #0C1A2A !important;
            }

            /* --- Radio Buttons --- */
            .stRadio > div {
                background-color: #152642;
                padding: 0.5rem;
                border-radius: 4px;
                border: 1px solid #2D4A76;
            }

            /* --- Progress Bar --- */
            .stProgress > div > div > div {
                background-color: #00FF88;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Caching ---
@st.cache_data(ttl=3600)
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data(ttl=3600)
def load_profile_data(file_path):
    return pd.read_excel(file_path)

# --- StatsBomb Style Helper Functions ---
def create_statsbomb_shot_map(df, title="SHOT MAP"):
    """Creates a StatsBomb-style shot map"""
    if df.empty:
        return None, "No data available"
    
    # StatsBomb color scheme
    BG_COLOR = "#0C1A2A"
    PITCH_COLOR = "#152642"
    LINE_COLOR = "#2D4A76"
    TEXT_COLOR = "#B0B7C3"
    ACCENT_COLOR = "#00FF88"
    GOAL_COLOR = "#00FF88"
    SHOT_COLOR = "#FF6B6B"
    
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_COLOR, 
                         line_color=LINE_COLOR, linewidth=1.5, half=False)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    
    # Plot shots
    for _, shot in df.iterrows():
        x, y = shot['x'], shot['y']
        xg = shot['xG']
        is_goal = shot['isGoal']
        
        color = GOAL_COLOR if is_goal else SHOT_COLOR
        size = max(xg * 600, 50)
        alpha = 0.8 if is_goal else 0.6
        
        ax.scatter(y, x, c=color, s=size, alpha=alpha, edgecolors='white', linewidth=1)
        
        # Add xG value for significant chances
        if xg > 0.1:
            ax.text(y, x + 1.5, f'{xg:.2f}', ha='center', va='bottom', 
                   color='white', fontsize=8, fontweight='bold')

    # StatsBomb-style title
    ax.text(0.5, 0.98, title, transform=ax.transAxes, ha='center', va='top',
           fontsize=16, fontweight='bold', color=ACCENT_COLOR, 
           fontfamily='Roboto Mono')
    
    # Key stats
    total_shots = len(df)
    total_xg = df['xG'].sum()
    goals = df['isGoal'].sum()
    
    stats_text = f"SHOTS: {total_shots} | xG: {total_xg:.2f} | GOALS: {goals}"
    ax.text(0.5, 0.92, stats_text, transform=ax.transAxes, ha='center', va='top',
           fontsize=12, color=TEXT_COLOR, fontfamily='Roboto Mono')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SHOT_COLOR, 
                  markersize=8, label='Shot', alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=GOAL_COLOR, 
                  markersize=8, label='Goal', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
             ncol=2, frameon=False, fontsize=10, labelcolor=TEXT_COLOR)
    
    return fig, None

def create_statsbomb_radar(player_data, categories, values, title="PLAYER PROFILE"):
    """Creates a StatsBomb-style radar chart"""
    # StatsBomb doesn't typically use radar charts, but we'll create a minimalist version
    # if needed. For now, we'll focus on their signature table and shot map styles.
    pass

def create_statsbomb_match_report(df, team1, team2):
    """Creates a StatsBomb-style match report"""
    # Implementation for match report in StatsBomb style
    pass

# --- StatsBomb Style Page Components ---
def create_navigation_header():
    """Creates StatsBomb-style navigation header"""
    st.markdown("""
        <div class="nav-header">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 2rem;">
                <div style="display: flex; align-items: center; gap: 2rem;">
                    <h2 style="margin: 0; color: #00FF88;">WOSO ANALYTICS</h2>
                    <div style="height: 30px; width: 2px; background: #2D4A76;"></div>
                    <p style="margin: 0; color: #B0B7C3; font-family: 'Roboto Mono';">STATSBOMB INSPIRED</p>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="nav-button" onclick="window.location.reload()">REFRESH</button>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_statsbomb_card(title, content, width=None):
    """Creates a StatsBomb-style card component"""
    col = st.columns([1])[0] if not width else st.columns(width)[0]
    with col:
        st.markdown(f"""
            <div class="statsbomb-card">
                <h4 style="color: #00FF88; margin-bottom: 1rem; font-family: 'Roboto Mono';">{title}</h4>
                {content}
            </div>
        """, unsafe_allow_html=True)

# --- Page Display Functions ---
def display_landing_page():
    """StatsBomb-style landing page"""
    st.markdown("""
        <style>
        .block-container { max-width: 1000px; padding-top: 4rem; } 
        div[data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 0;">
                <h1 style="font-size: 4rem; color: #00FF88; margin-bottom: 1rem; font-family: 'Roboto Mono';">
                    WOSO ANALYTICS
                </h1>
                <p style="color: #B0B7C3; font-size: 1.2rem; margin-bottom: 3rem;">
                    STATSBOMB-INSPIRED FOOTBALL INTELLIGENCE PLATFORM
                </p>
                <div style="display: flex; gap: 2rem; justify-content: center; margin-bottom: 4rem;">
                    <div style="text-align: center;">
                        <h3 style="color: #00FF88; font-family: 'Roboto Mono';">200K+</h3>
                        <p style="color: #B0B7C3;">EVENTS ANALYZED</p>
                    </div>
                    <div style="width: 2px; background: #2D4A76;"></div>
                    <div style="text-align: center;">
                        <h3 style="color: #00FF88; font-family: 'Roboto Mono';">15+</h3>
                        <p style="color: #B0B7C3;">LEAGUES</p>
                    </div>
                    <div style="width: 2px; background: #2D4A76;"></div>
                    <div style="text-align: center;">
                        <h3 style="color: #00FF88; font-family: 'Roboto Mono';">REALTIME</h3>
                        <p style="color: #B0B7C3;">DATA UPDATES</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Action button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ENTER ANALYTICS SUITE", use_container_width=True, type="primary"):
            st.session_state.app_mode = "MainApp"
            st.rerun()

def display_data_scouting_page(data_config, metric_info):
    """StatsBomb-style data scouting page"""
    st.markdown('<div class="section-header"><h2>PLAYER PERFORMANCE ANALYSIS</h2></div>', 
                unsafe_allow_html=True)
    
    # League and metric selection in cards
    col1, col2 = st.columns(2)
    with col1:
        create_statsbomb_card("COMPETITION", """
            <select style="width: 100%; padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                <option>WSL</option>
                <option>WSL 2</option>
                <option>FRAUEN-BUNDESLIGA</option>
                <option>LIGA F</option>
                <option>NWSL</option>
            </select>
        """)
    
    with col2:
        create_statsbomb_card("METRIC", """
            <select style="width: 100%; padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                <option>EXPECTED GOALS (xG)</option>
                <option>EXPECTED ASSISTS (xA)</option>
                <option>EXPECTED THREAT (xT)</option>
                <option>PRESSING INTENSITY</option>
                <option>PASSING PROGRESSION</option>
            </select>
        """)
    
    # Data display
    col1, col2 = st.columns([2, 1])
    with col1:
        create_statsbomb_card("PERFORMANCE LEADERBOARD", """
            <div style="background: #1E3A5C; padding: 1rem; border-radius: 4px;">
                <table style="width: 100%; color: #B0B7C3; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid #2D4A76;">
                            <th style="padding: 0.5rem; text-align: left; color: #00FF88;">PLAYER</th>
                            <th style="padding: 0.5rem; text-align: right; color: #00FF88;">TEAM</th>
                            <th style="padding: 0.5rem; text-align: right; color: #00FF88;">xG</th>
                            <th style="padding: 0.5rem; text-align: right; color: #00FF88;">PER 90</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #2D4A76;">
                            <td style="padding: 0.5rem;">SAM KERR</td>
                            <td style="padding: 0.5rem; text-align: right;">CHE</td>
                            <td style="padding: 0.5rem; text-align: right;">14.2</td>
                            <td style="padding: 0.5rem; text-align: right;">0.78</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2D4A76;">
                            <td style="padding: 0.5rem;">ALEXIA PUTELLAS</td>
                            <td style="padding: 0.5rem; text-align: right;">FCB</td>
                            <td style="padding: 0.5rem; text-align: right;">12.8</td>
                            <td style="padding: 0.5rem; text-align: right;">0.72</td>
                        </tr>
                        <!-- Add more rows as needed -->
                    </tbody>
                </table>
            </div>
        """)
    
    with col2:
        create_statsbomb_card("FILTERS", """
            <div style="margin-bottom: 1rem;">
                <label style="color: #B0B7C3; display: block; margin-bottom: 0.5rem;">MINUTES PLAYED</label>
                <input type="range" style="width: 100%;">
            </div>
            <div style="margin-bottom: 1rem;">
                <label style="color: #B0B7C3; display: block; margin-bottom: 0.5rem;">POSITION</label>
                <select style="width: 100%; padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                    <option>ALL POSITIONS</option>
                    <option>FORWARD</option>
                    <option>MIDFIELDER</option>
                    <option>DEFENDER</option>
                </select>
            </div>
        """)

def display_match_analysis_page():
    """StatsBomb-style match analysis page"""
    st.markdown('<div class="section-header"><h2>MATCH ANALYSIS CENTRE</h2></div>', 
                unsafe_allow_html=True)
    
    # Match selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        create_statsbomb_card("MATCH SELECTOR", """
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <select style="padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                    <option>SELECT COMPETITION</option>
                </select>
                <select style="padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                    <option>SELECT MATCH</option>
                </select>
            </div>
        """)
    
    # Shot map
    create_statsbomb_card("SHOT MAP", """
        <div style="text-align: center; padding: 2rem; background: #1E3A5C; border-radius: 4px;">
            <p style="color: #B0B7C3;">SHOT MAP VISUALIZATION WILL APPEAR HERE</p>
        </div>
    """)
    
    # Match stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("POSSESSION", "54%", "2%")
    with col2:
        st.metric("EXPECTED GOALS", "2.1", "0.3")
    with col3:
        st.metric("SHOTS", "14", "2")
    with col4:
        st.metric("PASS ACCURACY", "82%", "-1%")

def display_player_profiling_page():
    """StatsBomb-style player profiling page"""
    st.markdown('<div class="section-header"><h2>PLAYER PROFILES</h2></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        create_statsbomb_card("PLAYER SELECTION", """
            <div style="margin-bottom: 1rem;">
                <label style="color: #B0B7C3; display: block; margin-bottom: 0.5rem;">PLAYER</label>
                <select style="width: 100%; padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                    <option>SEARCH PLAYERS...</option>
                </select>
            </div>
            <div style="margin-bottom: 1rem;">
                <label style="color: #B0B7C3; display: block; margin-bottom: 0.5rem;">POSITION</label>
                <select style="width: 100%; padding: 0.5rem; background: #1E3A5C; border: 1px solid #2D4A76; color: white; border-radius: 4px;">
                    <option>ALL POSITIONS</option>
                </select>
            </div>
        """)
    
    with col2:
        create_statsbomb_card("PLAYER PROFILE", """
            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 2rem;">
                <div>
                    <div style="background: #1E3A5C; height: 120px; border-radius: 4px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                        <span style="color: #B0B7C3;">PLAYER IMAGE</span>
                    </div>
                    <div style="color: #B0B7C3;">
                        <p><strong>AGE:</strong> 26</p>
                        <p><strong>TEAM:</strong> CHELSEA</p>
                        <p><strong>MINUTES:</strong> 1,840</p>
                    </div>
                </div>
                <div>
                    <h4 style="color: #00FF88; margin-bottom: 1rem;">PERFORMANCE METRICS</h4>
                    <div style="background: #1E3A5C; padding: 1rem; border-radius: 4px;">
                        <p style="color: #B0B7C3; margin: 0.5rem 0;">xG: <strong style="color: #00FF88;">14.2</strong></p>
                        <p style="color: #B0B7C3; margin: 0.5rem 0;">ASSISTS: <strong style="color: #00FF88;">8</strong></p>
                        <p style="color: #B0B7C3; margin: 0.5rem 0;">PASS ACCURACY: <strong style="color: #00FF88;">84%</strong></p>
                    </div>
                </div>
            </div>
        """)

def display_corners_page():
    """StatsBomb-style corners analysis page"""
    st.markdown('<div class="section-header"><h2>SET PIECE ANALYSIS</h2></div>', 
                unsafe_allow_html=True)
    
    create_statsbomb_card("CORNER KICK ANALYSIS", """
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 2rem;">
            <div>
                <div style="background: #1E3A5C; height: 300px; border-radius: 4px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #B0B7C3;">CORNER KICK SHOT MAP VISUALIZATION</span>
                </div>
            </div>
            <div>
                <h4 style="color: #00FF88; margin-bottom: 1rem;">CORNER STATS</h4>
                <div style="background: #1E3A5C; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                    <p style="color: #B0B7C3; margin: 0.5rem 0;">TOTAL CORNERS: <strong style="color: #00FF88;">47</strong></p>
                    <p style="color: #B0B7C3; margin: 0.5rem 0;">SHOTS FROM CORNERS: <strong style="color: #00FF88;">12</strong></p>
                    <p style="color: #B0B7C3; margin: 0.5rem 0;">xG FROM CORNERS: <strong style="color: #00FF88;">1.8</strong></p>
                </div>
            </div>
        </div>
    """)

# --- Main App Logic ---
def main():
    """Main application function"""
    inject_statsbomb_css()
    
    # Configuration
    metric_info = {
        'xG (Expected Goals)': 'Probability of shot resulting in goal',
        'xAG (Expected Assisted Goals)': 'Likelihood of pass becoming assist',
        'xT (Expected Threat)': 'Threat created by ball progression',
        'Expected Disruption (xDisruption)': 'Defensive actions disrupting attacks',
        'Goal Probability Added (GPA/G+)': 'Impact on goal probability'
    }
    
    # Session state initialization
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Landing"
    if 'page_view' not in st.session_state:
        st.session_state.page_view = "Data Scouting"
    
    # Page routing
    if st.session_state.app_mode == "Landing":
        display_landing_page()
    else:
        # StatsBomb navigation
        create_navigation_header()
        
        # Navigation tabs
        tabs = st.tabs([
            "📊 PERFORMANCE", 
            "🎯 MATCHES", 
            "👤 PROFILES", 
            "⛳ SET PIECES"
        ])
        
        with tabs[0]:
            st.session_state.page_view = "Data Scouting"
            display_data_scouting_page({}, metric_info)
        with tabs[1]:
            st.session_state.page_view = "Match Analysis"
            display_match_analysis_page()
        with tabs[2]:
            st.session_state.page_view = "Player Profiling"
            display_player_profiling_page()
        with tabs[3]:
            st.session_state.page_view = "Corners"
            display_corners_page()
        
        # StatsBomb footer
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #2D4A76; font-size: 0.8rem; font-family: 'Roboto Mono';">
                <p>WOSO ANALYTICS PLATFORM | STATSBOMB INSPIRED | DATA PROVIDED BY OPTA</p>
                <p>© 2024 WOMEN'S FOOTBALL ANALYTICS</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()