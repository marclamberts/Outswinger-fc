import streamlit as st
import pandas as pd
import os
import glob

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | Advanced Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Neon/Modern Styling ---
def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; background-color: #0f1923; color: #fff; }
            h1, h2, h3, h4 { font-family: 'Roboto Mono', monospace; color: #00FFA3; }
            .stButton>button { background-color:#00FFA3; color:#0f1923; font-weight:600; border-radius:6px; padding:0.7rem 1.2rem; }
            .stButton>button:hover { background-color:#00CC7F; }
            .stSelectbox div[data-baseweb="select"] > div { background-color:#152642; color:#fff; border-radius:6px; }
            .stDataFrame { background-color:#152642; border-radius:6px; }
            .stDataFrame .data-grid-header { background-color:#1E3A5C; color:#00FFA3; font-weight:600; }
            .section-header { border-bottom:2px solid #00FFA3; padding-bottom:0.5rem; margin-bottom:1.5rem; font-family:'Roboto Mono'; }
        </style>
    """, unsafe_allow_html=True)

# --- Load Advanced Metrics ---
@st.cache_data(ttl=3600)
def load_all_metrics(base_path="data/advanced"):
    if not os.path.exists(base_path):
        return {}
    leagues = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,f))]
    all_metrics = {}
    for league in leagues:
        league_path = os.path.join(base_path, league)
        league_files = glob.glob(os.path.join(league_path,"*.csv"))
        league_dict = {}
        for f in league_files:
            filename = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f)
            league_dict[filename] = df
        all_metrics[league] = league_dict
    return all_metrics

# --- Advanced Metrics Page ---
def display_advanced_metrics(all_metrics):
    st.title("⚽ WoSo Advanced Metrics")
    st.markdown("<div class='section-header'>Player & Team Analytics</div>", unsafe_allow_html=True)

    if not all_metrics:
        st.warning("No metrics found.")
        return

    # Sidebar - league and metric selection
    league_selected = st.sidebar.selectbox("Select League", list(all_metrics.keys()))
    league_metrics = all_metrics.get(league_selected, {})

    if not league_metrics:
        st.warning(f"No metrics available for {league_selected}.")
        return

    metric_options = ["xG","xAG","xT","xDisruption","GPA"]
    metric_choice = st.sidebar.selectbox("Select Metric", metric_options)

    # Map metric choice to filenames
    filename_map = {
        "xG": f"{league_selected}.csv",
        "xAG": f"{league_selected}_assists.csv",
        "GPA": f"{league_selected}_gpa.csv",
        "xT": f"{league_selected}_xt.csv",
        "xDisruption": f"{league_selected}_disruption.csv"
    }

    expected_file = filename_map.get(metric_choice)
    df_metric = league_metrics.get(expected_file)

    if df_metric is not None:
        st.subheader(f"{metric_choice} Metrics - {league_selected}")
        st.dataframe(df_metric)
    else:
        st.warning(f"{metric_choice} data not found for {league_selected}.")

# --- Main ---
def main():
    inject_custom_css()
    all_metrics = load_all_metrics("data/advanced")
    display_advanced_metrics(all_metrics)

if __name__=="__main__":
    main()
