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
DATA_PATH = Path("WSL.csv")   # put WSL.csv next to this app.py (or change to absolute path)
APP_TITLE = "WoSo Analytics | WSL"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLING: Hybrid ASA (clean cards) + Understat (dark) + FBref (table/typography)
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
        --good: {OUTSWINGER["emerald"]};
      }}

      html, body, [class*="st-"] {{
        font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      /* Tighten app padding a bit */
      .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2rem;
      }}

      /* Headers: a touch of FBref vibe */
      h1, h2, h3 {{
        font-family: "Source Serif 4", Georgia, serif !important;
        color: var(--text);
        letter-spacing: 0.2px;
      }}
      h4, h5 {{
        font-family: "Roboto Mono", ui-monospace, monospace !important;
        color: var(--accent);
      }}

      /* Sidebar polish */
      section[data-testid="stSidebar"] {{
        background: #0E1013;
        border-right: 1px solid var(--border);
      }}
      section[data-testid="stSidebar"] * {{
        color: var(--text);
      }}

      /* Inputs */
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

      /* Buttons (ASA-ish clean) */
      .stButton>button {{
        background: var(--accent);
        color: #0B0B0B;
        font-weight: 800;
        border-radius: 14px;
        border: 0;
        padding: 0.7rem 1.1rem;
      }}
      .stButton>button:hover {{
        filter: brightness(0.95);
      }}

      /* Section divider */
      .section-header {{
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.45rem;
        margin: 0.2rem 0 1.1rem 0;
        font-family: "Roboto Mono", ui-monospace, monospace;
        color: var(--accent);
        letter-spacing: 0.3px;
        text-transform: uppercase;
        font-size: 0.95rem;
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
      }}
      .dot {{
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
        display: inline-block;
      }}

      /* FBref-ish table feel */
      .stDataFrame, div[data-testid="stDataFrame"] {{
        background: #0F1115 !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        overflow: hidden;
      }}
      /* Remove some excessive white from grid */
      div[data-testid="stDataFrame"] * {{
        color: var(--text) !important;
      }}

      /* Percentile bars */
      .pct-wrap {{
        background: #0F1115;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
      }}
      .pct-row {{
        display: grid;
        grid-template-columns: 160px 1fr 54px;
        gap: 10px;
        align-items: center;
        margin: 8px 0;
      }}
      .pct-name {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .pct-bar {{
        height: 10px;
        background: #141821;
        border: 1px solid var(--border);
        border-radius: 999px;
        overflow: hidden;
      }}
      .pct-fill {{
        height: 100%;
        background: var(--accent);
        width: 50%;
      }}
      .pct-val {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        color: var(--text);
        font-weight: 700;
      }}

    </style>
    """, unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=3600)
def load_wsl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

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

def fmt_float(x, nd=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def percentile(series: pd.Series, value) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return 0.0
    return float((s < float(value)).mean() * 100.0)

def pct_block(df: pd.DataFrame, player_row: pd.Series, metrics: list[tuple[str, str]]):
    st.markdown("<div class='pct-wrap'>", unsafe_allow_html=True)
    for col, label in metrics:
        if col not in df.columns:
            continue
        val = player_row.get(col, np.nan)
        pct = percentile(df[col], val)
        pct_int = int(round(pct))
        fill = max(0, min(100, pct_int))
        st.markdown(
            f"""
            <div class="pct-row">
              <div class="pct-name">{label}</div>
              <div class="pct-bar"><div class="pct-fill" style="width:{fill}%"></div></div>
              <div class="pct-val">{pct_int}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# APP UI
# =========================
def app():
    inject_custom_css()

    df = load_wsl(DATA_PATH)

    st.title("⚽ WoSo Analytics — WSL")
    st.markdown("<div class='section-header'>Advanced Player Output</div>", unsafe_allow_html=True)

    if df.empty:
        st.error(
            f"Could not find **{DATA_PATH.name}**. Put `WSL.csv` next to `app.py` or change `DATA_PATH` at the top."
        )
        return

    # If your CSV is the aggregated output, these are common columns:
    numeric_candidates = ["Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"]
    df = coerce_numeric(df, [c for c in numeric_candidates if c in df.columns])

    # Sidebar controls (FBref-ish: lots of filters)
    st.sidebar.markdown("### Controls")
    search = st.sidebar.text_input("Search PlayerId", value="").strip().lower()

    min_shots = 0
    if "Shots" in df.columns:
        max_shots = int(np.nanmax(df["Shots"].values)) if len(df) else 0
        min_shots = st.sidebar.slider("Minimum shots", 0, max_shots if max_shots > 0 else 1, 0)

    # Sort controls
    sortable = [c for c in ["xG_Diff", "xG", "xA", "Goals", "Assists", "Shots", "ShotConversion"] if c in df.columns]
    if not sortable:
        sortable = list(df.columns)

    sort_col = st.sidebar.selectbox("Sort by", sortable, index=0 if "xG_Diff" in sortable else 0)
    ascending = st.sidebar.checkbox("Ascending", value=False)

    # Prepare view
    view = df.copy()
    if "PlayerId" in view.columns:
        view["PlayerId"] = view["PlayerId"].astype(str)

    if search and "PlayerId" in view.columns:
        view = view[view["PlayerId"].str.lower().str.contains(search, na=False)]

    if "Shots" in view.columns:
        view = view[view["Shots"].fillna(0) >= min_shots]

    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)

    # Round to max 2 decimals (as requested)
    view = round_cols(view, ["xG", "xG_Diff", "xA", "ShotConversion"], nd=2)

    # Top chips (Understat-ish quick legend)
    chips = []
    if "xG_Diff" in view.columns:
        chips.append(("xG_Diff", "Goals − xG (over/under)"))
    if "xA" in view.columns:
        chips.append(("xA", "Assisted xG created"))
    if "ShotConversion" in view.columns:
        chips.append(("ShotConversion", "Goals / Shots"))

    if chips:
        chip_html = ""
        for _, label in chips:
            chip_html += f"<span class='chip'><span class='dot'></span>{label}</span>"
        st.markdown(chip_html, unsafe_allow_html=True)

    # KPIs (ASA clean cards)
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

    tab_overview, tab_table, tab_profile = st.tabs(["Overview", "Table (FBref-style)", "Player Profile"])

    with tab_overview:
        st.markdown("<div class='section-header'>Highlights</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        if "xG_Diff" in view.columns and "PlayerId" in view.columns:
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

        # Simple distribution table cut (no forced colors)
        st.markdown("<div class='section-header'>Benchmarks</div>", unsafe_allow_html=True)
        bench_cols = [c for c in ["Shots", "Goals", "Assists", "xG", "xA", "xG_Diff", "ShotConversion"] if c in view.columns]
        if bench_cols:
            desc = view[bench_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
            desc = desc.rename(columns={"50%": "median"})
            desc = round_cols(desc, desc.columns.tolist(), nd=2)
            st.dataframe(desc, use_container_width=True)

    with tab_table:
        st.markdown("<div class='section-header'>Full Player Table</div>", unsafe_allow_html=True)

        # Choose columns to show (FBref: lots of columns, but tidy)
        preferred = ["PlayerId", "Shots", "Goals", "Assists", "xG", "xG_Diff", "xA", "ShotConversion"]
        show_cols = [c for c in preferred if c in view.columns]
        if not show_cols:
            show_cols = list(view.columns)

        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

        st.caption("Tip: use the sidebar filters to set a minimum shots threshold before judging xG_Diff.")

    with tab_profile:
        st.markdown("<div class='section-header'>Player Profile</div>", unsafe_allow_html=True)

        if "PlayerId" not in df.columns:
            st.warning("No PlayerId column found, so player profile is unavailable.")
        else:
            # Use unrounded df for percentiles, display rounded values in KPIs
            df_all = df.copy()
            df_all["PlayerId"] = df_all["PlayerId"].astype(str)

            # Player picker based on current filtered view
            player_list = view["PlayerId"].astype(str).tolist() if "PlayerId" in view.columns else df_all["PlayerId"].tolist()
            player_list = sorted(list(dict.fromkeys(player_list)))  # unique, stable
            picked = st.selectbox("Select Player", player_list, index=0 if player_list else 0)

            prow = df_all.loc[df_all["PlayerId"] == picked]
            if prow.empty:
                st.info("No rows found for that player.")
            else:
                prow = prow.iloc[0]

                # KPI row for player (ASA cards)
                shots = prow.get("Shots", np.nan)
                goals = prow.get("Goals", np.nan)
                assists = prow.get("Assists", np.nan)
                xg = prow.get("xG", np.nan)
                xg_diff = prow.get("xG_Diff", np.nan)
                xa = prow.get("xA", np.nan)
                conv = prow.get("ShotConversion", np.nan)

                # round displays to 2 decimals
                xg = float(xg) if pd.notna(xg) else np.nan
                xa = float(xa) if pd.notna(xa) else np.nan
                xg_diff = float(xg_diff) if pd.notna(xg_diff) else np.nan
                conv = float(conv) if pd.notna(conv) else np.nan

                st.markdown(
                    f"""
                    <div class="kpi-grid">
                      <div class="kpi">
                        <div class="label">Shots</div>
                        <div class="value">{int(shots) if pd.notna(shots) else 0}</div>
                        <div class="hint">Total shots</div>
                      </div>
                      <div class="kpi">
                        <div class="label">Goals</div>
                        <div class="value">{int(goals) if pd.notna(goals) else 0}</div>
                        <div class="hint">Goals scored</div>
                      </div>
                      <div class="kpi">
                        <div class="label">xG (sum)</div>
                        <div class="value">{fmt_float(xg, 2)}</div>
                        <div class="hint">Shot quality</div>
                      </div>
                      <div class="kpi">
                        <div class="label">xG Diff</div>
                        <div class="value">{fmt_float(xg_diff, 2)}</div>
                        <div class="hint">Goals − xG</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Secondary KPIs
                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Assists", int(assists) if pd.notna(assists) else 0)
                with cB:
                    st.metric("xA (assisted xG)", float(f"{xa:.2f}") if pd.notna(xa) else 0.0)
                with cC:
                    st.metric("Shot Conversion", float(f"{conv:.2f}") if pd.notna(conv) else 0.0)

                # Percentile bars (FBref vibe)
                st.subheader("Percentile ranks (vs dataset)")
                metrics = []
                if "xG" in df_all.columns: metrics.append(("xG", "xG"))
                if "xA" in df_all.columns: metrics.append(("xA", "xA"))
                if "Shots" in df_all.columns: metrics.append(("Shots", "Shots"))
                if "Goals" in df_all.columns: metrics.append(("Goals", "Goals"))
                if "Assists" in df_all.columns: metrics.append(("Assists", "Assists"))
                if "xG_Diff" in df_all.columns: metrics.append(("xG_Diff", "Goals − xG"))
                if "ShotConversion" in df_all.columns: metrics.append(("ShotConversion", "Shot Conversion"))

                pct_block(df_all, prow, metrics)

                # Raw row (debug / transparency)
                with st.expander("Show raw player row"):
                    show = prow.to_frame("value")
                    # Round numeric values to 2 decimals for readability
                    for idx in show.index:
                        try:
                            show.loc[idx, "value"] = round(float(show.loc[idx, "value"]), 2)
                        except Exception:
                            pass
                    st.dataframe(show, use_container_width=True)

    st.caption("Style blend: Understat (dark), ASA (clean KPI cards), FBref (dense tables + percentiles).")

if __name__ == "__main__":
    app()
