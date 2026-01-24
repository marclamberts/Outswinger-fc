import streamlit as st
import pandas as pd
from pathlib import Path

# --- OUTSWINGER COLOR PALETTE ---
OUTSWINGER = {
    "yellow":  "#F4C430",
    "black":   "#0B0B0B",
    "white":   "#F7F7F7",
    "grey":    "#9AA0A6",
    "emerald": "#2ECC71",
    "crimson": "#E63946",
    "blue":    "#4D96FF",
    "violet":  "#7B61FF",
}

# --- App Configuration ---
st.set_page_config(
    page_title="WoSo Analytics | WSL",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (uses OUTSWINGER palette) ---
def inject_custom_css():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
            html, body, [class*="st-"] {{
                font-family: 'Inter', sans-serif;
                background-color: {OUTSWINGER["black"]};
                color: {OUTSWINGER["white"]};
            }}
            h1, h2, h3, h4 {{
                font-family: 'Roboto Mono', monospace;
                color: {OUTSWINGER["emerald"]};
            }}
            .stButton>button {{
                background-color: {OUTSWINGER["emerald"]};
                color: {OUTSWINGER["black"]};
                font-weight: 700;
                border-radius: 10px;
                padding: 0.7rem 1.2rem;
                border: 0;
            }}
            .stButton>button:hover {{
                background-color: #22C55E;
            }}
            .stSelectbox div[data-baseweb="select"] > div {{
                background-color: #121212;
                color: {OUTSWINGER["white"]};
                border-radius: 10px;
                border: 1px solid #222;
            }}
            .stDataFrame {{
                background-color: #121212;
                border-radius: 12px;
                border: 1px solid #222;
            }}
            .section-header {{
                border-bottom: 2px solid {OUTSWINGER["emerald"]};
                padding-bottom: 0.5rem;
                margin-bottom: 1.2rem;
                font-family:'Roboto Mono';
                font-weight: 700;
            }}
            .kpi {{
                background: #121212;
                border: 1px solid #222;
                border-radius: 16px;
                padding: 14px 16px;
            }}
            .muted {{
                color: {OUTSWINGER["grey"]};
                font-size: 0.95rem;
            }}
        </style>
    """, unsafe_allow_html=True)

# --- Load WSL.csv ---
@st.cache_data(ttl=3600)
def load_wsl_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def safe_round(df: pd.DataFrame, cols, nd=2) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].round(nd)
    return df

def display_wsl_app(df: pd.DataFrame):
    st.title("⚽ WoSo Analytics — WSL")
    st.markdown("<div class='section-header'>WSL Player & Team Analytics</div>", unsafe_allow_html=True)

    if df.empty:
        st.warning("WSL.csv not found or empty. Check the file path in the sidebar.")
        return

    # --- Sidebar controls ---
    st.sidebar.header("Filters")
    search = st.sidebar.text_input("Search PlayerId", value="").strip().lower()

    # Try to infer common columns
    # If your WSL.csv is your aggregated output, you likely have:
    # PlayerId, Shots, Goals, Assists, xG, xG_Diff, xA, ShotConversion
    metric_cols = [c for c in ["Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"] if c in df.columns]

    sort_col = st.sidebar.selectbox(
        "Sort by",
        options=(metric_cols if metric_cols else list(df.columns)),
        index=(metric_cols.index("xG_Diff") if "xG_Diff" in metric_cols else 0)
    )
    ascending = st.sidebar.checkbox("Sort ascending", value=False)

    # Optional filters if columns exist
    min_shots = 0
    if "Shots" in df.columns:
        min_shots = st.sidebar.slider("Min shots", min_value=0, max_value=int(max(df["Shots"].max(), 0)), value=0)

    # --- Prepare dataframe ---
    df_view = df.copy()

    # Round numeric columns nicely (max 2 decimals)
    df_view = safe_round(df_view, ["xG", "xG_Diff", "xA", "ShotConversion"], nd=2)

    if search and "PlayerId" in df_view.columns:
        df_view = df_view[df_view["PlayerId"].astype(str).str.lower().str.contains(search, na=False)]

    if "Shots" in df_view.columns:
        df_view = df_view[df_view["Shots"].fillna(0) >= min_shots]

    if sort_col in df_view.columns:
        df_view = df_view.sort_values(sort_col, ascending=ascending)

    # --- KPIs (top row) ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi'>"
                    f"<div class='muted'>Players</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;'>{len(df_view):,}</div>"
                    "</div>", unsafe_allow_html=True)
    with c2:
        total_goals = int(pd.to_numeric(df_view["Goals"], errors="coerce").fillna(0).sum()) if "Goals" in df_view.columns else 0
        st.markdown("<div class='kpi'>"
                    f"<div class='muted'>Total Goals</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;'>{total_goals:,}</div>"
                    "</div>", unsafe_allow_html=True)
    with c3:
        total_xg = float(pd.to_numeric(df_view["xG"], errors="coerce").fillna(0).sum()) if "xG" in df_view.columns else 0.0
        st.markdown("<div class='kpi'>"
                    f"<div class='muted'>Total xG</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;'>{total_xg:.2f}</div>"
                    "</div>", unsafe_allow_html=True)
    with c4:
        total_xa = float(pd.to_numeric(df_view["xA"], errors="coerce").fillna(0).sum()) if "xA" in df_view.columns else 0.0
        st.markdown("<div class='kpi'>"
                    f"<div class='muted'>Total xA</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;'>{total_xa:.2f}</div>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("")  # spacing

    # --- Table ---
    st.subheader("WSL Table")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    # --- Quick highlights ---
    st.markdown("<div class='section-header'>Highlights</div>", unsafe_allow_html=True)
    colA, colB = st.columns(2)

    if "xG_Diff" in df_view.columns:
        top_over = df_view.nlargest(10, "xG_Diff")[["PlayerId", "xG_Diff", "Goals", "xG"]].copy()
        top_under = df_view.nsmallest(10, "xG_Diff")[["PlayerId", "xG_Diff", "Goals", "xG"]].copy()
        with colA:
            st.markdown("**Top overperformers (Goals − xG)**")
            st.dataframe(top_over, use_container_width=True, hide_index=True)
        with colB:
            st.markdown("**Top underperformers (Goals − xG)**")
            st.dataframe(top_under, use_container_width=True, hide_index=True)

def main():
    inject_custom_css()

    st.sidebar.title("Data Source")

    # Option 1: load from local path (default)
    default_path = "WSL.csv"  # put WSL.csv next to this app.py, or change this default
    csv_path = st.sidebar.text_input("Path to WSL.csv", value=default_path)

    # Option 2: upload a CSV (overrides path if provided)
    uploaded = st.sidebar.file_uploader("…or upload WSL.csv", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_wsl_csv(csv_path)

    display_wsl_app(df)

if __name__ == "__main__":
    main()
