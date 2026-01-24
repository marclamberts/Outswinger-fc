import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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

COL_PLAYER = "PlayerId"
NUMERIC_CANDIDATES = ["Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"]

COLUMN_GROUPS = {
    "Core": ["Shots", "Goals", "Assists"],
    "Finishing": ["xG", "xG_Diff", "ShotConversion"],
    "Creativity": ["xA"],
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
# STYLING
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
        --warn: {OUTSWINGER["yellow"]};
        --bad: {OUTSWINGER["crimson"]};
      }}

      html, body, [class*="st-"] {{
        font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2rem;
      }}

      h1, h2, h3 {{
        font-family: "Source Serif 4", Georgia, serif !important;
        color: var(--text);
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
        margin: 0.25rem 0 0.9rem 0;
      }}
      @media (max-width: 1100px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
      .kpi {{
        background: linear-gradient(180deg, var(--panel), var(--panel2));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
      }}
      .kpi .label {{
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 6px;
      }}
      .kpi .value {{
        font-size: 1.85rem;
        font-weight: 900;
        line-height: 1.05;
      }}
      .kpi .hint {{
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 6px;
      }}

      /* FBref-ish dataframe container */
      div[data-testid="stDataFrame"] {{
        background: #0F1115 !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
      }}
      div[data-testid="stDataFrame"] * {{
        color: var(--text) !important;
      }}

      /* Sticky top nav (big outlined rectangles) */
      .sticky-nav {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(11, 11, 11, 0.92);
        backdrop-filter: blur(10px);
        padding-top: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 10px;
      }}

      div[data-testid="stRadio"] > label {{ display:none; }}
      div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 0;
      }}
      div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        border: 2px solid var(--accent) !important;
        border-radius: 18px !important;
        padding: 14px 12px !important;
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
      div[data-testid="stRadio"] input[type="radio"] {{ display: none !important; }}

      /* Badges */
      .badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #0F1115;
        font-size: 0.85rem;
        font-weight: 800;
      }}
      .badge.good {{ border-color: rgba(46,204,113,0.55); }}
      .badge.bad {{ border-color: rgba(230,57,70,0.55); }}
      .badge .b-dot {{
        width: 8px; height: 8px; border-radius: 999px; display: inline-block;
      }}
      .badge.good .b-dot {{ background: var(--accent); }}
      .badge.bad .b-dot {{ background: var(--bad); }}

      /* Percentile bars */
      .pct-wrap {{
        background: #0F1115;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
      }}
      .pct-row {{
        display: grid;
        grid-template-columns: 170px 1fr 54px;
        gap: 10px;
        align-items: center;
        margin: 8px 0;
      }}
      .pct-name {{ color: var(--muted); font-size: 0.92rem; }}
      .pct-bar {{
        height: 10px;
        background: #141821;
        border: 1px solid var(--border);
        border-radius: 999px;
        overflow: hidden;
      }}
      .pct-fill {{ height: 100%; background: var(--accent); width: 50%; }}
      .pct-val {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        color: var(--text);
        font-weight: 800;
      }}

      /* Landing hero */
      .hero {{
        background: radial-gradient(900px 380px at 30% 10%, rgba(46,204,113,0.18), transparent 60%),
                    linear-gradient(180deg, rgba(17,19,22,1), rgba(10,10,10,1));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px 22px;
        box-shadow: 0 10px 34px rgba(0,0,0,0.28);
        margin: 10px 0 18px 0;
      }}
      .hero-kicker {{
        font-family: "Roboto Mono", ui-monospace, monospace;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--accent);
        font-size: 0.92rem;
        margin-bottom: 8px;
      }}
      .hero-title {{
        font-family: "Source Serif 4", Georgia, serif;
        font-weight: 800;
        font-size: 2.2rem;
        line-height: 1.1;
        margin: 0 0 8px 0;
      }}
      .hero-sub {{
        color: var(--muted);
        font-size: 1.02rem;
        margin-top: 4px;
        max-width: 70ch;
      }}
      .hero-row {{
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 16px;
        margin-top: 14px;
      }}
      @media (max-width: 1000px) {{
        .hero-row {{ grid-template-columns: 1fr; }}
      }}
      .hero-card {{
        background: #0F1115;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
      }}
      .hero-card h4 {{
        margin: 0 0 6px 0;
        font-family: "Roboto Mono", ui-monospace, monospace;
        color: var(--accent);
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
      }}
      .hero-card p {{
        margin: 0;
        color: var(--muted);
        font-size: 0.95rem;
      }}

      /* Make primary button feel more "product" */
      .stButton > button {{
        background: var(--accent) !important;
        color: #0B0B0B !important;
        font-weight: 900 !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.1rem !important;
        border: 0 !important;
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

def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0

def safe_float(x, nd=2) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(f"{float(x):.{nd}f}")
    except Exception:
        return 0.0

def percentile(series: pd.Series, value) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return 0
    return int(round((s < float(value)).mean() * 100.0))

def pct_block(df_all: pd.DataFrame, player_row: pd.Series, metrics: list[tuple[str, str]]):
    st.markdown("<div class='pct-wrap'>", unsafe_allow_html=True)
    for col, label in metrics:
        if col not in df_all.columns:
            continue
        val = player_row.get(col, np.nan)
        pct = percentile(df_all[col], val)
        fill = max(0, min(100, pct))
        st.markdown(
            f"""
            <div class="pct-row">
              <div class="pct-name">{label}</div>
              <div class="pct-bar"><div class="pct-fill" style="width:{fill}%"></div></div>
              <div class="pct-val">{pct}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

def file_meta(path: Path):
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return {"mtime": mtime}
    except Exception:
        return None

# =========================
# LANDING PAGE
# =========================
def landing_page():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">WoSo Analytics</div>
          <div class="hero-title">Advanced Metrics Dashboard</div>
          <div class="hero-sub">
            A fast, analyst-friendly view of finishing and chance-creation across WSL, Liga F, and Frauen-Bundesliga.
            Built with an ASA × Understat × FBref vibe: clean cards, dark UI, and dense tables that still read well.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 0.8], vertical_alignment="top")
    with c1:
        st.markdown("<div class='section-header'>How to use</div>", unsafe_allow_html=True)
        st.markdown(
            """
- Pick a **league** in the sidebar.
- Use **Minimum shots** (default 10) before judging finishing.
- Jump between **Overview / Table / Profile** with the top navigation.
- In **Profile**, check **percentiles** and optionally compare two players.
            """.strip()
        )

        st.markdown("<div class='section-header'>Click to learn what the metrics mean</div>", unsafe_allow_html=True)

        with st.expander("What is xG_Diff (Goals − xG)?", expanded=False):
            st.write(
                "xG_Diff is the simplest indicator of over/under-performance vs expected goals. "
                "Positive means scoring more than expected, negative means scoring less. "
                "Small samples can be noisy — use Min shots."
            )

        with st.expander("What is xA here?", expanded=False):
            st.write(
                "In your dataset, xA is calculated as the sum of the shot xG for shots where the player is the assister "
                "(relatedPlayerId). It represents the quality of chances created for teammates."
            )

        with st.expander("What is ShotConversion?", expanded=False):
            st.write(
                "ShotConversion = Goals / Shots. It’s a simple finishing rate. "
                "Use it with xG_Diff to separate shot volume from shot quality."
            )

        with st.expander("Why use Minimum shots?", expanded=False):
            st.write(
                "Finishing metrics fluctuate a lot in small samples. "
                "A minimum shots threshold helps avoid false conclusions from randomness."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Enter Dashboard →", use_container_width=True):
            st.session_state.show_dashboard = True
            st.rerun()

    with c2:
        st.markdown("<div class='section-header'>What’s inside</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hero-card">
              <h4>Overview</h4>
              <p>Leaders and laggards for xG_Diff plus distribution benchmarks.</p>
            </div>
            <div style="height:10px"></div>
            <div class="hero-card">
              <h4>Table</h4>
              <p>FBref-style browsing with column groups + custom selections.</p>
            </div>
            <div style="height:10px"></div>
            <div class="hero-card">
              <h4>Profile</h4>
              <p>Player KPIs + percentiles + optional head-to-head comparison.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption("Tip: To show this landing page again, use the sidebar toggle once you’re inside the dashboard.")

# =========================
# DASHBOARD
# =========================
def dashboard():
    # --- Sidebar: league + filters ---
    st.sidebar.title("League")
    league = st.sidebar.selectbox("Select League", list(LEAGUES.keys()), index=0)
    csv_path = Path(LEAGUES[league])

    # Option to return to landing
    st.sidebar.markdown("---")
    if st.sidebar.button("← Back to Landing", use_container_width=True):
        st.session_state.show_dashboard = False
        st.rerun()

    df = load_league_csv(csv_path)

    st.title(f"⚽ WoSo Analytics — {league}")
    st.markdown("<div class='section-header'>Advanced Player Output</div>", unsafe_allow_html=True)

    meta = file_meta(csv_path)
    if meta is None:
        st.error(
            f"Could not load **{csv_path.name}**.\n\n"
            "Put the CSV next to `app.py` (recommended) or update the filename in `LEAGUES`."
        )
        return
    if df.empty:
        st.warning(f"Loaded **{csv_path.name}** but it looks empty.")
        return

    # Coerce numeric columns (robust across leagues)
    df = coerce_numeric(df, [c for c in NUMERIC_CANDIDATES if c in df.columns])
    if COL_PLAYER in df.columns:
        df[COL_PLAYER] = df[COL_PLAYER].astype(str)

    # File status
    c_meta1, c_meta2, c_meta3 = st.columns(3)
    with c_meta1:
        st.caption("Data file")
        st.code(csv_path.name)
    with c_meta2:
        st.caption("Rows")
        st.code(f"{len(df):,}")
    with c_meta3:
        st.caption("Last modified")
        st.code(meta["mtime"].strftime("%Y-%m-%d %H:%M"))

    # Sidebar filters
    st.sidebar.markdown("### Filters")
    search = st.sidebar.text_input("Search PlayerId", value="").strip().lower()

    min_shots = 0
    if "Shots" in df.columns:
        max_shots = int(np.nanmax(df["Shots"].values)) if len(df) else 0
        default_min = 10 if max_shots >= 10 else 0
        min_shots = st.sidebar.slider("Minimum shots", 0, max_shots if max_shots > 0 else 1, default_min)

    sortable = [c for c in ["xG_Diff", "xG", "xA", "Goals", "Assists", "Shots", "ShotConversion"] if c in df.columns]
    if not sortable:
        sortable = [COL_PLAYER] if COL_PLAYER in df.columns else list(df.columns)

    sort_col = st.sidebar.selectbox("Sort by", sortable, index=(0 if "xG_Diff" in sortable else 0))
    ascending = st.sidebar.checkbox("Ascending", value=False)

    # Column groups + picker
    st.sidebar.markdown("### Columns")
    group_choice = st.sidebar.multiselect(
        "Column groups",
        options=list(COLUMN_GROUPS.keys()),
        default=["Core", "Finishing", "Creativity"],
    )

    grouped_cols = []
    for g in group_choice:
        grouped_cols.extend(COLUMN_GROUPS.get(g, []))
    grouped_cols = [c for c in grouped_cols if c in df.columns]

    extras = [c for c in NUMERIC_CANDIDATES if c in df.columns and c not in grouped_cols]
    include_extras = st.sidebar.checkbox("Include other numeric columns", value=False)

    show_cols_base = grouped_cols + (extras if include_extras else [])
    show_cols_base = list(dict.fromkeys(show_cols_base))

    if COL_PLAYER in df.columns and COL_PLAYER not in show_cols_base:
        show_cols_base = [COL_PLAYER] + show_cols_base
    elif COL_PLAYER in df.columns:
        show_cols_base = [COL_PLAYER] + [c for c in show_cols_base if c != COL_PLAYER]

    final_cols = st.sidebar.multiselect(
        "Select columns",
        options=([COL_PLAYER] if COL_PLAYER in df.columns else []) + [c for c in df.columns if c != COL_PLAYER],
        default=show_cols_base if show_cols_base else list(df.columns),
    )

    # Build view
    view = df.copy()
    if search and COL_PLAYER in view.columns:
        view = view[view[COL_PLAYER].str.lower().str.contains(search, na=False)]
    if "Shots" in view.columns:
        view = view[view["Shots"].fillna(0) >= min_shots]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending, na_position="last")

    view_display = round_cols(view, ["xG", "xG_Diff", "xA", "ShotConversion"], nd=2)

    # Chips
    chips = []
    if "xG_Diff" in df.columns:
        chips.append("Goals − xG (over/under)")
    if "xA" in df.columns:
        chips.append("xA = assisted xG created")
    if "ShotConversion" in df.columns:
        chips.append("Goals / Shots")
    if chips:
        st.markdown(
            "".join([f"<span class='chip'><span class='dot'></span>{c}</span>" for c in chips]),
            unsafe_allow_html=True
        )

    # KPIs
    total_players = len(view_display)
    total_goals = int(np.nansum(view_display["Goals"])) if "Goals" in view_display.columns else 0
    total_xg = float(np.nansum(view_display["xG"])) if "xG" in view_display.columns else 0.0
    total_xa = float(np.nansum(view_display["xA"])) if "xA" in view_display.columns else 0.0

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

    # Sticky nav
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    st.markdown("<div class='sticky-nav'>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview", "Table", "Profile"], horizontal=True, key="page")
    st.markdown("</div>", unsafe_allow_html=True)

    # Pages
    if page == "Overview":
        st.markdown("<div class='section-header'>Overview</div>", unsafe_allow_html=True)

        if "Shots" in df.columns and min_shots < 10:
            st.warning("You are viewing players with fewer than 10 shots. xG_Diff can be very noisy in small samples.")

        col1, col2 = st.columns(2)
        if all(c in view_display.columns for c in [COL_PLAYER, "xG_Diff", "Goals", "xG"]):
            with col1:
                st.subheader("Top overperformers")
                st.caption("Highest Goals − xG")
                st.dataframe(
                    view_display.nlargest(10, "xG_Diff")[[COL_PLAYER, "xG_Diff", "Goals", "xG"]],
                    use_container_width=True,
                    hide_index=True,
                )
            with col2:
                st.subheader("Top underperformers")
                st.caption("Lowest Goals − xG")
                st.dataframe(
                    view_display.nsmallest(10, "xG_Diff")[[COL_PLAYER, "xG_Diff", "Goals", "xG"]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Overview highlights need columns: PlayerId, xG_Diff, Goals, xG")

        st.markdown("<div class='section-header'>Benchmarks</div>", unsafe_allow_html=True)
        bench_cols = [c for c in ["Shots", "Goals", "Assists", "xG", "xA", "xG_Diff", "ShotConversion"] if c in view_display.columns]
        if bench_cols:
            desc = view_display[bench_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
            desc = desc.rename(columns={"50%": "median"})
            desc = round_cols(desc, list(desc.columns), nd=2)
            st.dataframe(desc, use_container_width=True)

    elif page == "Table":
        st.markdown("<div class='section-header'>Table</div>", unsafe_allow_html=True)

        if not final_cols:
            st.warning("No columns selected.")
            return

        if COL_PLAYER in view_display.columns and COL_PLAYER in final_cols:
            final_cols = [COL_PLAYER] + [c for c in final_cols if c != COL_PLAYER]

        show = [c for c in final_cols if c in view_display.columns]
        st.dataframe(view_display[show], use_container_width=True, hide_index=True)
        st.caption("Tip: use Min shots = 10+ before judging xG_Diff.")

    else:
        st.markdown("<div class='section-header'>Profile</div>", unsafe_allow_html=True)

        if COL_PLAYER not in df.columns:
            st.warning("No PlayerId column found, so profile is unavailable.")
            return

        player_list = (
            view_display[COL_PLAYER].astype(str).tolist()
            if COL_PLAYER in view_display.columns
            else df[COL_PLAYER].astype(str).tolist()
        )
        player_list = sorted(list(dict.fromkeys(player_list)))
        if not player_list:
            st.info("No players in the current filtered view.")
            return

        st.markdown("#### Player selection")
        picked = st.selectbox("Select Player", player_list, index=0)

        compare_mode = st.checkbox("Compare vs another player", value=False)
        picked_2 = None
        if compare_mode:
            others = [p for p in player_list if p != picked]
            if others:
                picked_2 = st.selectbox("Select comparison player", others, index=0)

        df_all = df.copy()
        df_all[COL_PLAYER] = df_all[COL_PLAYER].astype(str)

        prow_df = df_all.loc[df_all[COL_PLAYER] == str(picked)]
        if prow_df.empty:
            st.info("No row found for selected player.")
            return
        prow = prow_df.iloc[0]

        badge_html = ""
        if "xG_Diff" in df_all.columns:
            xgd = prow.get("xG_Diff", np.nan)
            if pd.notna(xgd) and float(xgd) >= 0:
                badge_html = "<span class='badge good'><span class='b-dot'></span>OVERPERFORMING xG</span>"
            elif pd.notna(xgd):
                badge_html = "<span class='badge bad'><span class='b-dot'></span>UNDERPERFORMING xG</span>"
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)

        shots = safe_int(prow.get("Shots", 0))
        goals = safe_int(prow.get("Goals", 0))
        assists = safe_int(prow.get("Assists", 0))
        xg = safe_float(prow.get("xG", 0.0), 2)
        xg_diff = safe_float(prow.get("xG_Diff", 0.0), 2)
        xa = safe_float(prow.get("xA", 0.0), 2)
        conv = safe_float(prow.get("ShotConversion", 0.0), 2)

        st.markdown(
            f"""
            <div class="kpi-grid">
              <div class="kpi"><div class="label">Shots</div><div class="value">{shots}</div><div class="hint">Volume</div></div>
              <div class="kpi"><div class="label">Goals</div><div class="value">{goals}</div><div class="hint">Output</div></div>
              <div class="kpi"><div class="label">xG (sum)</div><div class="value">{xg:.2f}</div><div class="hint">Chance quality</div></div>
              <div class="kpi"><div class="label">xG Diff</div><div class="value">{xg_diff:.2f}</div><div class="hint">Goals − xG</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Assists", assists if "Assists" in df_all.columns else 0)
        with c2:
            st.metric("xA (assisted xG)", f"{xa:.2f}" if "xA" in df_all.columns else "—")
        with c3:
            st.metric("Shot Conversion", f"{conv:.2f}" if "ShotConversion" in df_all.columns else "—")

        if "Shots" in df_all.columns and shots < 10:
            st.warning("This player has fewer than 10 shots. Treat xG_Diff as high-variance.")

        st.markdown("#### Percentile ranks (within this league dataset)")
        metrics = []
        for col, label in [
            ("xG", "xG"),
            ("xA", "xA"),
            ("Shots", "Shots"),
            ("Goals", "Goals"),
            ("Assists", "Assists"),
            ("xG_Diff", "Goals − xG"),
            ("ShotConversion", "Shot Conversion"),
        ]:
            if col in df_all.columns:
                metrics.append((col, label))
        pct_block(df_all, prow, metrics)

        if compare_mode and picked_2:
            prow2_df = df_all.loc[df_all[COL_PLAYER] == str(picked_2)]
            if not prow2_df.empty:
                prow2 = prow2_df.iloc[0]

                st.markdown("<div class='section-header'>Compare</div>", unsafe_allow_html=True)

                compare_cols = [COL_PLAYER] + [c for c in ["Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"] if c in df_all.columns]
                comp = pd.DataFrame([prow[compare_cols], prow2[compare_cols]]).copy()
                comp = comp.rename(columns={COL_PLAYER: "Player"})
                comp["Player"] = [picked, picked_2]
                for c in comp.columns:
                    if c != "Player":
                        comp[c] = pd.to_numeric(comp[c], errors="coerce")
                for c in ["xG", "xG_Diff", "xA", "ShotConversion"]:
                    if c in comp.columns:
                        comp[c] = comp[c].round(2)

                st.dataframe(comp, use_container_width=True, hide_index=True)
                st.markdown("#### Percentiles (comparison player)")
                pct_block(df_all, prow2, metrics)

        with st.expander("Show raw player row"):
            raw = prow.to_frame("value")
            for idx in raw.index:
                try:
                    raw.loc[idx, "value"] = round(float(raw.loc[idx, "value"]), 2)
                except Exception:
                    pass
            st.dataframe(raw, use_container_width=True)

    st.caption("WSL • Liga F • Frauen-Bundesliga | ASA × Understat × FBref inspired")

# =========================
# RUN
# =========================
def main():
    inject_custom_css()

    if "show_dashboard" not in st.session_state:
        st.session_state.show_dashboard = False

    if not st.session_state.show_dashboard:
        landing_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()
