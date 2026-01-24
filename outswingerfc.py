import streamlit as st
import pandas as pd
import numpy as np
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

# =========================
# CONFIG
# =========================
APP_TITLE = "WoSo Analytics | Advanced Metrics"
LEAGUES = {
    "WSL": "WSL.csv",
    "Liga F": "Liga F.csv",
    "Frauen-Bundesliga": "FBL.csv",
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLING: Understat (dark) + ASA (cards) + FBref (tables)
# =========================
def inject_custom_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Roboto+Mono:wght@400;700&family=Source+Serif+4:wght@400;600;700&display=swap');

      :root {{
        --bg: {OUTSWINGER["black"]};
        --panel: #111316;
        --panel2: #15181d;
        --border: #232832;
        --text: {OUTSWINGER["white"]};
        --muted: {OUTSWINGER["grey"]};
        --accent: {OUTSWINGER["emerald"]};
      }}

      html, body, [class*="st-"] {{
        font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 2rem;
      }}

      h1, h2, h3 {{
        font-family: "Source Serif 4", Georgia, serif !important;
        color: var(--text);
      }}
      h4, h5 {{
        font-family: "Roboto Mono", ui-monospace, monospace !important;
        color: var(--accent);
      }}

      section[data-testid="stSidebar"] {{
        background: #0E1013;
        border-right: 1px solid var(--border);
      }}

      .stTextInput input, .stNumberInput input {{
        background: #0F1115 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
      }}
      .stSelectbox div[data-baseweb="select"] > div {{
        background: #0F1115 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
      }}

      .section-header {{
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.45rem;
        margin: 0.3rem 0 1.1rem 0;
        font-family: "Roboto Mono", ui-monospace, monospace;
        color: var(--accent);
        letter-spacing: 0.3px;
        text-transform: uppercase;
        font-size: 0.95rem;
      }}

      /* Understat-ish chips */
      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #0F1115;
        color: var(--text);
        font-size: 0.85rem;
        margin-right: 8px;
        margin-top: 6px;
      }}
      .dot {{
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
        display: inline-block;
      }}

      /* KPI cards (ASA vibe) */
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 0.25rem 0 1rem 0;
      }}
      @media (max-width: 1100px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
      .kpi {{
        background: linear-gradient(180deg, var(--panel), var(--panel2));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
      }}
      .kpi .label {{
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 6px;
      }}
      .kpi .value {{
        font-size: 1.9rem;
        font-weight: 900;
        line-height: 1.05;
      }}
      .kpi .hint {{
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 6px;
      }}

      /* FBref-ish table container */
      div[data-testid="stDataFrame"] {{
        background: #0F1115 !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
      }}
      div[data-testid="stDataFrame"] * {{
        color: var(--text) !important;
      }}

      /* === TOP NAV (big outlined rectangles) === */
      div[data-testid="stRadio"] > label {{
        display: none;
      }}

      div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 10px 0 18px 0;
      }}

      div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        border: 2px solid var(--accent) !important;
        border-radius: 18px !important;
        padding: 16px 12px !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: all 0.15s ease;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 900;
        justify-content: center;
        align-items: center;
        display: flex !important;
      }}

      div[data-testid="stRadio"] div[role="radiogroup"] > label span {{
        color: var(--accent) !important;
        font-size: 1.05rem !important;
      }}

      div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        background: rgba(46, 204, 113, 0.12) !important;
      }}

      div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        background: var(--accent) !important;
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.35);
      }}
      div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {{
        color: #0B0B0B !important;
      }}

      div[data-testid="stRadio"] input[type="radio"] {{
        display: none !important;
      }}
    </style>
    """, unsafe_allow_html=True)

# =========================
# DATA HELPERS
# =========================
@st.cache_data(ttl=3600)
def load_league_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def round_cols(df: pd.DataFrame, cols: list[str], nd: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)
    return out

# =========================
# APP
# =========================
def app():
    inject_custom_css()

    # --- Sidebar league selection ---
    st.sidebar.title("League")
    league = st.sidebar.selectbox("Select League", list(LEAGUES.keys()), index=0)

    csv_path = Path(LEAGUES[league])
    df = load_league_csv(csv_path)

    st.title(f"⚽ WoSo Analytics — {league}")
    st.markdown("<div class='section-header'>Advanced Player Output</div>", unsafe_allow_html=True)

    if df.empty:
        st.error(
            f"Could not load **{csv_path.name}**.\n\n"
            "Put the CSV next to `app.py` or update the filename in `LEAGUES`."
        )
        return

    # Expected numeric columns (safe if some missing)
    numeric_candidates = ["Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"]
    df = coerce_numeric(df, [c for c in numeric_candidates if c in df.columns])

    if "PlayerId" in df.columns:
        df["PlayerId"] = df["PlayerId"].astype(str)

    # --- Sidebar filters ---
    st.sidebar.markdown("### Filters")
    search = st.sidebar.text_input("Search PlayerId", value="").strip().lower()

    min_shots = 0
    if "Shots" in df.columns:
        max_shots = int(np.nanmax(df["Shots"].values)) if len(df) else 0
        min_shots = st.sidebar.slider("Minimum shots", 0, max_shots if max_shots > 0 else 1, 0)

    # Sort controls
    sortable = [c for c in ["xG_Diff", "xG", "xA", "Goals", "Assists", "Shots", "ShotConversion"] if c in df.columns]
    if not sortable:
        sortable = list(df.columns)

    sort_col = st.sidebar.selectbox("Sort by", sortable, index=(0 if "xG_Diff" in sortable else 0))
    ascending = st.sidebar.checkbox("Ascending", value=False)

    # --- Build view (filters + sorting) ---
    view = df.copy()

    if search and "PlayerId" in view.columns:
        view = view[view["PlayerId"].str.lower().str.contains(search, na=False)]

    if "Shots" in view.columns:
        view = view[view["Shots"].fillna(0) >= min_shots]

    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)

    # Round display values to max 2 decimals
    view = round_cols(view, ["xG", "xG_Diff", "xA", "ShotConversion"], nd=2)

    # --- Top navigation (big outlined buttons) ---
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    page = st.radio("Navigation", ["Overview", "Table", "Profile"], horizontal=True, key="page")

    # --- Chips legend ---
    chips = []
    if "xG_Diff" in view.columns:
        chips.append("Goals − xG (over/under)")
    if "xA" in view.columns:
        chips.append("xA = assisted xG created")
    if "ShotConversion" in view.columns:
        chips.append("Goals / Shots")

    if chips:
        st.markdown("".join([f"<span class='chip'><span class='dot'></span>{c}</span>" for c in chips]),
                    unsafe_allow_html=True)

    # --- KPIs ---
    total_players = len(view)
    total_goals = int(np.nansum(view["Goals"])) if "Goals" in view.columns else 0
    total_xg = float(np.nansum(view["xG"])) if "xG" in view.columns else 0.0
    total_xa = float(np.nansum(view["xA"])) if "xA" in view.columns else 0.0

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="label">Players</div>
            <div class="value">{total_players:,}</div>
            <div class="hint">Filtered view</div>
          </div>
          <div class="kpi">
            <div class="label">Total Goals</div>
            <div class="value">{total_goals:,}</div>
            <div class="hint">Sum across filtered players</div>
          </div>
          <div class="kpi">
            <div class="label">Total xG</div>
            <div class="value">{total_xg:.2f}</div>
            <div class="hint">Shot quality generated</div>
          </div>
          <div class="kpi">
            <div class="label">Total xA</div>
            <div class="value">{total_xa:.2f}</div>
            <div class="hint">Assisted shot quality</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Pages ---
    if page == "Overview":
        st.markdown("<div class='section-header'>Highlights</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        if all(c in view.columns for c in ["PlayerId", "xG_Diff", "Goals", "xG"]):
            top_over = view.nlargest(10, "xG_Diff")[["PlayerId", "xG_Diff", "Goals", "xG"]].copy()
            top_under = view.nsmallest(10, "xG_Diff")[["PlayerId", "xG_Diff", "Goals", "xG"]].copy()

            with col1:
                st.subheader("Top overperformers")
                st.caption("Higher Goals − xG")
                st.dataframe(top_over, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Top underperformers")
                st.caption("Lower Goals − xG")
                st.dataframe(top_under, use_container_width=True, hide_index=True)
        else:
            st.info("Overview needs columns: PlayerId, xG_Diff, Goals, xG")

    elif page == "Table":
        st.markdown("<div class='section-header'>Full Player Table</div>", unsafe_allow_html=True)
        preferred = ["PlayerId", "Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"]
        show_cols = [c for c in preferred if c in view.columns]
        if not show_cols:
            show_cols = list(view.columns)
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
        st.caption("Tip: set a minimum shots threshold before judging xG_Diff.")

    else:  # Profile
        st.markdown("<div class='section-header'>Player Profile</div>", unsafe_allow_html=True)

        if "PlayerId" not in df.columns:
            st.warning("No PlayerId column found, so player profile is unavailable.")
            return

        player_list = view["PlayerId"].astype(str).tolist() if "PlayerId" in view.columns else df["PlayerId"].astype(str).tolist()
        player_list = sorted(list(dict.fromkeys(player_list)))

        if not player_list:
            st.info("No players in the current filtered view.")
            return

        picked = st.selectbox("Select Player", player_list, index=0)

        prow_df = df.loc[df["PlayerId"].astype(str) == str(picked)]
        if prow_df.empty:
            st.info("No rows found for that player.")
            return

        prow = prow_df.iloc[0]

        shots = int(prow["Shots"]) if "Shots" in prow and pd.notna(prow["Shots"]) else 0
        goals = int(prow["Goals"]) if "Goals" in prow and pd.notna(prow["Goals"]) else 0
        assists = int(prow["Assists"]) if "Assists" in prow and pd.notna(prow["Assists"]) else 0
        xg = round(float(prow["xG"]), 2) if "xG" in prow and pd.notna(prow["xG"]) else 0.0
        xg_diff = round(float(prow["xG_Diff"]), 2) if "xG_Diff" in prow and pd.notna(prow["xG_Diff"]) else 0.0
        xa = round(float(prow["xA"]), 2) if "xA" in prow and pd.notna(prow["xA"]) else 0.0
        conv = round(float(prow["ShotConversion"]), 2) if "ShotConversion" in prow and pd.notna(prow["ShotConversion"]) else 0.0

        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi"><div class="label">Shots</div><div class="value">{shots}</div><div class="hint">Total shots</div></div>
              <div class="kpi"><div class="label">Goals</div><div class="value">{goals}</div><div class="hint">Goals scored</div></div>
              <div class="kpi"><div class="label">xG (sum)</div><div class="value">{xg:.2f}</div><div class="hint">Shot quality</div></div>
              <div class="kpi"><div class="label">xG Diff</div><div class="value">{xg_diff:.2f}</div><div class="hint">Goals − xG</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Assists", assists)
        with c2:
            st.metric("xA (assisted xG)", xa)
        with c3:
            st.metric("Shot Conversion", conv)

    st.caption("WSL • Liga F • Frauen-Bundesliga | ASA × Understat × FBref inspired")

if __name__ == "__main__":
    inject_custom_css()
    app()
