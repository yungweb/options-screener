"""
Options Screener — Streamlit Dashboard
Run: streamlit run dashboard.py

Requires: pip install streamlit plotly pandas numpy scipy requests
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import requests
from dataclasses import asdict

# ── Local modules ──────────────────────────────────
from pattern_detection import (
    detect_double_bottom,
    detect_break_and_retest,
    add_confluence_filters,
    find_swing_highs,
    find_swing_lows,
)
from options_chain import (
    get_options_recommendation,
    fetch_iv_rank,
    print_recommendation,
)
from backtester import run_backtest, BacktestReport


# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Screener",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
  }

  /* Dark terminal background */
  .stApp {
    background-color: #0a0e17;
    color: #c8d6e5;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2d44;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1b2e 0%, #111d30 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Share Tech Mono', monospace;
  }
  div[data-testid="metric-container"] label {
    color: #4a90d9 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4fd !important;
    font-size: 1.6rem !important;
  }

  /* Setup alert cards */
  .setup-card {
    background: linear-gradient(135deg, #0a1628 0%, #0d1e35 100%);
    border-left: 3px solid #00d4aa;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
  }
  .setup-card.bearish {
    border-left-color: #ff4d6d;
  }
  .setup-card h4 {
    color: #00d4aa;
    margin: 0 0 8px 0;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
  }
  .setup-card.bearish h4 { color: #ff4d6d; }
  .setup-card .label { color: #4a90d9; margin-right: 6px; }
  .setup-card .value { color: #e8f4fd; }

  /* Options recommendation box */
  .options-box {
    background: #080f1c;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.8;
  }
  .options-box .strategy-tag {
    display: inline-block;
    background: #0f3460;
    color: #4fc3f7;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
  }

  /* Section headers */
  .section-header {
    color: #4a90d9;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
  }

  /* Status indicator */
  .status-live {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00d4aa;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* IV bar */
  .iv-bar-bg {
    background: #1a2540;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1220;
    border-bottom: 1px solid #1e2d44;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    color: #4a6fa5;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 10px 20px;
    border: none;
    background: transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa;
    background: transparent !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0f3460, #1a4a80);
    color: #4fc3f7;
    border: 1px solid #1e5080;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1a4a80, #0f3460);
    border-color: #4fc3f7;
    color: #e8f4fd;
  }

  /* Selectbox */
  .stSelectbox label { color: #4a90d9; font-size: 0.75rem; letter-spacing: 0.1em; }
  div[data-baseweb="select"] > div {
    background: #0d1b2e;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    color: #c8d6e5;
    font-family: 'Share Tech Mono', monospace;
  }
  div[data-baseweb="popover"] { background: #0d1b2e !important; }
  li[role="option"] { color: #c8d6e5; font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0e17; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────

POLYGON_API_KEY = "UuSaC2Rxwihj2xpzVOWsbAbE3pcNPHVj"

WATCHLIST = [
    "PLTR", "NBIS", "VRT", "CRDO", "WDC", "GOOGL", "AAOI",
    "ONDS", "LMRI", "ZETA", "ASTS", "RDW", "ABCL",
    "SPY", "QQQ", "SOXX",
]

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17",
        font=dict(color="#c8d6e5", family="Share Tech Mono"),
        xaxis=dict(gridcolor="#111d30", linecolor="#1e2d44", zerolinecolor="#1e2d44"),
        yaxis=dict(gridcolor="#111d30", linecolor="#1e2d44", zerolinecolor="#1e2d44"),
    )
)


# ─────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_bars(ticker: str, timespan: str = "hour", days_back: int = 60) -> pd.DataFrame:
    if POLYGON_API_KEY == "YOUR_POLYGON_API_KEY":
        return _generate_mock_bars(ticker, days_back, timespan)

    end   = datetime.now()
    start = end - timedelta(days=days_back)
    url   = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data.get("results"):
            return _generate_mock_bars(ticker, days_back, timespan)
        df = pd.DataFrame(data["results"])
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","t":"timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df[["open","high","low","close","volume"]]
    except Exception:
        return _generate_mock_bars(ticker, days_back, timespan)


def _generate_mock_bars(ticker: str, days_back: int, timespan: str) -> pd.DataFrame:
    """Deterministic mock OHLCV for demo mode."""
    seed = sum(ord(c) for c in ticker)
    rng  = np.random.RandomState(seed)
    n    = days_back * (6 if timespan == "hour" else 1)
    n    = max(n, 80)

    base   = 20 + (seed % 150)
    trend  = rng.randn(n).cumsum() * 0.6 + base
    trend  = np.clip(trend, 5, 600)

    close  = trend
    high   = close + rng.rand(n) * close * 0.015
    low    = close - rng.rand(n) * close * 0.015
    open_  = close + rng.randn(n) * close * 0.005
    vol    = rng.randint(500_000, 3_000_000, n)

    idx = pd.date_range(end=datetime.now(), periods=n, freq="1h" if timespan=="hour" else "1D")
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=idx)


@st.cache_data(ttl=600)
def get_iv_rank_cached(ticker: str):
    try:
        return fetch_iv_rank(ticker)
    except Exception:
        return 45.0, 50.0, 0.32


# ─────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────

def build_candlestick_chart(df: pd.DataFrame, ticker: str, setups_db=None, setups_br=None) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # ── Candlesticks ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=ticker,
        increasing=dict(fillcolor="#00d4aa", line=dict(color="#00d4aa", width=1)),
        decreasing=dict(fillcolor="#ff4d6d", line=dict(color="#ff4d6d", width=1)),
        whiskerwidth=0.2,
    ), row=1, col=1)

    # ── 20 EMA ──
    ema20 = df["close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20, name="EMA 20",
        line=dict(color="#f39c12", width=1, dash="dot"),
    ), row=1, col=1)

    # ── VWAP ──
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap, name="VWAP",
        line=dict(color="#9b59b6", width=1, dash="dot"),
    ), row=1, col=1)

    # ── Swing highs / lows ──
    sh_idx = find_swing_highs(df["high"])
    sl_idx = find_swing_lows(df["low"])

    if len(sh_idx):
        fig.add_trace(go.Scatter(
            x=df.index[sh_idx], y=df["high"].iloc[sh_idx] * 1.005,
            mode="markers", name="Swing High",
            marker=dict(symbol="triangle-down", size=8, color="#ff4d6d"),
        ), row=1, col=1)

    if len(sl_idx):
        fig.add_trace(go.Scatter(
            x=df.index[sl_idx], y=df["low"].iloc[sl_idx] * 0.995,
            mode="markers", name="Swing Low",
            marker=dict(symbol="triangle-up", size=8, color="#00d4aa"),
        ), row=1, col=1)

    # ── Double Bottom annotations ──
    if setups_db:
        for s in setups_db:
            if not s.confirmed:
                continue
            # Neckline
            fig.add_hline(
                y=s.neckline, line_color="#f39c12", line_dash="dash", line_width=1,
                annotation_text=f"Neckline ${s.neckline:.2f}",
                annotation_font_color="#f39c12", annotation_font_size=10,
                row=1, col=1,
            )
            # Target
            fig.add_hline(
                y=s.target, line_color="#00d4aa", line_dash="dot", line_width=1,
                annotation_text=f"Target ${s.target:.2f}",
                annotation_font_color="#00d4aa", annotation_font_size=10,
                row=1, col=1,
            )
            # Stop
            fig.add_hline(
                y=s.stop_loss, line_color="#ff4d6d", line_dash="dot", line_width=1,
                annotation_text=f"Stop ${s.stop_loss:.2f}",
                annotation_font_color="#ff4d6d", annotation_font_size=10,
                row=1, col=1,
            )
            # Bottom markers
            b1 = df.index[s.bottom1_idx] if s.bottom1_idx < len(df) else df.index[-1]
            b2 = df.index[s.bottom2_idx] if s.bottom2_idx < len(df) else df.index[-1]
            fig.add_trace(go.Scatter(
                x=[b1, b2],
                y=[s.bottom1_price * 0.993, s.bottom2_price * 0.993],
                mode="markers+text",
                text=["W1", "W2"],
                textposition="bottom center",
                textfont=dict(color="#00d4aa", size=10),
                marker=dict(symbol="star", size=12, color="#00d4aa"),
                name="Double Bottom",
                showlegend=True,
            ), row=1, col=1)

    # ── Break & Retest annotations ──
    if setups_br:
        for s in setups_br:
            if not s.confirmed:
                continue
            color = "#00d4aa" if s.direction == "bullish" else "#ff4d6d"
            fig.add_hline(
                y=s.key_level, line_color=color, line_dash="longdash", line_width=1.5,
                annotation_text=f"B&R Level ${s.key_level:.2f}",
                annotation_font_color=color, annotation_font_size=10,
                row=1, col=1,
            )

    # ── Volume bars ──
    colors = ["#00d4aa" if c >= o else "#ff4d6d"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=10, color="#4a90d9"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor="#111d30", linecolor="#1e2d44"),
        yaxis2=dict(gridcolor="#111d30", linecolor="#1e2d44",
                    tickformat=".2s", title_text="Vol"),
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    return fig


def build_equity_curve(equity_curve: list, win_rate: float) -> go.Figure:
    fig = go.Figure()
    x   = list(range(len(equity_curve)))
    y   = equity_curve

    color_zones = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in y]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="#4a90d9", width=2),
        fill="tozeroy",
        fillcolor="rgba(74,144,217,0.08)",
        name="Equity (R)",
    ))

    # Zero line
    fig.add_hline(y=0, line_color="#1e3a5f", line_width=1)

    # Target line
    target_r = (win_rate / 100) * 2.0 * len(equity_curve) / 10
    fig.add_hline(
        y=max(y) * 0.85 if y else 1,
        line_color="#f39c12", line_dash="dot", line_width=1,
        annotation_text="85% Target Zone",
        annotation_font_color="#f39c12",
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        yaxis_title="Cumulative R",
        xaxis_title="Trade #",
    )
    return fig


def build_iv_gauge(iv_rank: float) -> go.Figure:
    color = "#ff4d6d" if iv_rank > 60 else ("#f39c12" if iv_rank > 30 else "#00d4aa")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=iv_rank,
        domain=dict(x=[0, 1], y=[0, 1]),
        number=dict(suffix="%", font=dict(color=color, size=28, family="Share Tech Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#4a90d9",
                      tickfont=dict(size=9, color="#4a90d9")),
            bar=dict(color=color),
            bgcolor="#0a0e17",
            borderwidth=1,
            bordercolor="#1e3a5f",
            steps=[
                dict(range=[0, 30],  color="#061020"),
                dict(range=[30, 60], color="#080f1c"),
                dict(range=[60, 100], color="#100810"),
            ],
            threshold=dict(
                line=dict(color="#fff", width=2),
                thickness=0.8,
                value=iv_rank,
            ),
        ),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def build_win_rate_donut(win_rate: float, wins: int, losses: int) -> go.Figure:
    color_win  = "#00d4aa"
    color_loss = "#ff4d6d"
    color_bg   = "#1a2540"

    fig = go.Figure(go.Pie(
        values=[wins, losses, max(0, 1 - wins - losses)],
        labels=["Wins", "Losses", ""],
        hole=0.68,
        marker=dict(colors=[color_win, color_loss, color_bg]),
        textinfo="none",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{win_rate:.0f}%</b>",
        x=0.5, y=0.55, showarrow=False,
        font=dict(size=22, color="#e8f4fd", family="Share Tech Mono"),
    )
    fig.add_annotation(
        text="WIN RATE",
        x=0.5, y=0.38, showarrow=False,
        font=dict(size=9, color="#4a90d9", family="Share Tech Mono"),
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px 0;">
      <div style="font-family:'Share Tech Mono',monospace; color:#00d4aa; font-size:1.1rem; letter-spacing:0.12em;">
        📡 OPTIONS<br>SCREENER
      </div>
      <div style="font-size:0.65rem; color:#4a6fa5; letter-spacing:0.2em; margin-top:4px;">
        PATTERN DETECTION ENGINE v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">TICKER</div>', unsafe_allow_html=True)
    selected_ticker = st.selectbox("", WATCHLIST, index=0, label_visibility="collapsed")

    st.markdown('<div class="section-header">TIMEFRAME</div>', unsafe_allow_html=True)
    timeframe = st.selectbox("", ["1H", "4H", "1D"], index=0, label_visibility="collapsed")
    tf_map = {"1H": ("hour", 60), "4H": ("hour", 120), "1D": ("day", 365)}
    timespan, days_back = tf_map[timeframe]

    st.markdown('<div class="section-header">PATTERN FILTERS</div>', unsafe_allow_html=True)
    show_db = st.toggle("Double Bottom", value=True)
    show_br = st.toggle("Break & Retest", value=True)

    st.markdown('<div class="section-header">THRESHOLDS</div>', unsafe_allow_html=True)
    min_rr        = st.slider("Min R:R", 1.0, 4.0, 2.0, 0.5)
    min_conf      = st.slider("Min Confluence", 0, 3, 2)
    db_tolerance  = st.slider("DB Tolerance %", 1, 6, 3) / 100

    st.markdown('<div class="section-header">ACCOUNT</div>', unsafe_allow_html=True)
    account_size  = st.number_input("Account Size ($)", value=10000, step=1000)
    risk_pct      = st.slider("Risk Per Trade %", 0.5, 3.0, 1.0, 0.5) / 100

    st.markdown("---")
    api_key_input = st.text_input("Polygon API Key", value="", type="password",
                                  placeholder="Paste key for live data")
    if api_key_input:
        POLYGON_API_KEY = api_key_input

    run_scan = st.button("⚡ RUN SCAN", use_container_width=True)

    st.markdown("""
    <div style="margin-top:24px; font-size:0.62rem; color:#2a4060; text-align:center; line-height:1.6;">
      DEMO MODE active<br>Using simulated OHLCV<br>Add API key for live data
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────

col_title, col_time = st.columns([4, 1])
with col_title:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; padding:8px 0 4px 0;">
      <span class="status-live"></span>
      <span style="font-family:'Share Tech Mono',monospace; font-size:1.3rem; color:#e8f4fd;">
        {selected_ticker}
      </span>
      <span style="font-family:'Share Tech Mono',monospace; color:#4a6fa5; font-size:0.75rem;">
        / {timeframe} / PATTERN SCANNER
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_time:
    st.markdown(f"""
    <div style="text-align:right; font-family:'Share Tech Mono',monospace;
                font-size:0.7rem; color:#4a6fa5; padding-top:16px;">
      {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# LOAD DATA + RUN DETECTION
# ─────────────────────────────────────────────────────

with st.spinner("Loading market data..."):
    df = fetch_bars(selected_ticker, timespan=timespan, days_back=days_back)

if df.empty:
    st.error("No data returned. Check your API key or ticker.")
    st.stop()

setups_db = detect_double_bottom(df, selected_ticker, tolerance=db_tolerance, rr_min=min_rr) if show_db else []
setups_br = detect_break_and_retest(df, selected_ticker, rr_min=min_rr) if show_br else []

confirmed_db = [s for s in setups_db if s.confirmed]
confirmed_br = [s for s in setups_br if s.confirmed]

# Confluence filter
def score(setup):
    c = add_confluence_filters(df, setup)
    return sum([c.get("rsi_bullish", False),
                c.get("price_above_vwap", False),
                c.get("price_above_ema20", False)])

confirmed_db = [s for s in confirmed_db if score(s) >= min_conf]
confirmed_br = [s for s in confirmed_br if score(s) >= min_conf]

all_confirmed = confirmed_db + confirmed_br

iv_rank, iv_pct, current_iv = get_iv_rank_cached(selected_ticker)
current_price = float(df["close"].iloc[-1])
price_change  = float(df["close"].iloc[-1] - df["close"].iloc[-2])
pct_change    = price_change / float(df["close"].iloc[-2]) * 100


# ─────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("PRICE", f"${current_price:.2f}", f"{pct_change:+.2f}%")
with m2:
    st.metric("SETUPS FOUND", len(all_confirmed),
              f"+{len(confirmed_db)} DB  +{len(confirmed_br)} B&R")
with m3:
    st.metric("IV RANK", f"{iv_rank:.0f}%",
              "HIGH IV" if iv_rank > 60 else ("LOW IV" if iv_rank < 30 else "NEUTRAL"))
with m4:
    vol_today = int(df["volume"].iloc[-1])
    vol_avg   = int(df["volume"].mean())
    st.metric("VOLUME", f"{vol_today/1e6:.1f}M", f"avg {vol_avg/1e6:.1f}M")
with m5:
    confluence_scores = [score(s) for s in all_confirmed]
    avg_conf = np.mean(confluence_scores) if confluence_scores else 0
    st.metric("AVG CONFLUENCE", f"{avg_conf:.1f}/3",
              "✅ STRONG" if avg_conf >= 2.5 else "⚠️ MODERATE" if avg_conf >= 1.5 else "WEAK")


# ─────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  CHART & SETUPS  ",
    "  🎯  OPTIONS CHAIN  ",
    "  📈  BACKTEST  ",
    "  🔍  WATCHLIST SCAN  ",
])


# ══════════════════════════════════════════════
# TAB 1 — CHART & SETUPS
# ══════════════════════════════════════════════
with tab1:
    left, right = st.columns([2.6, 1])

    with left:
        st.markdown('<div class="section-header">CANDLESTICK CHART</div>', unsafe_allow_html=True)
        fig = build_candlestick_chart(df, selected_ticker, confirmed_db, confirmed_br)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="section-header">IV ENVIRONMENT</div>', unsafe_allow_html=True)
        st.plotly_chart(build_iv_gauge(iv_rank),
                        use_container_width=True, config={"displayModeBar": False})

        regime = ("🔴 HIGH IV — Sell Premium" if iv_rank > 60
                  else "🟢 LOW IV — Buy Premium" if iv_rank < 30
                  else "🟡 NEUTRAL — Debit Spreads")
        st.markdown(f"""
        <div style="text-align:center; font-family:'Share Tech Mono',monospace;
                    font-size:0.7rem; color:#4a90d9; margin-top:-12px;">
          {regime}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">DETECTED SETUPS</div>', unsafe_allow_html=True)

        if not all_confirmed:
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        color:#2a4060; padding:20px 0; text-align:center;">
              No confirmed setups<br>at current thresholds
            </div>
            """, unsafe_allow_html=True)
        else:
            for s in confirmed_db:
                c = add_confluence_filters(df, s)
                conf = sum([c.get("rsi_bullish",False), c.get("price_above_vwap",False), c.get("price_above_ema20",False)])
                st.markdown(f"""
                <div class="setup-card">
                  <h4>⬆ DOUBLE BOTTOM — {s.ticker}</h4>
                  <div><span class="label">ENTRY</span><span class="value">${s.entry_price:.2f}</span></div>
                  <div><span class="label">STOP </span><span class="value">${s.stop_loss:.2f}</span></div>
                  <div><span class="label">TGRT </span><span class="value">${s.target:.2f}</span></div>
                  <div><span class="label">R:R  </span><span class="value">{s.rr_ratio}:1</span></div>
                  <div><span class="label">CONF </span><span class="value">{"★"*conf}{"☆"*(3-conf)}</span></div>
                  <div><span class="label">VOL  </span><span class="value">{"✅ Confirmed" if s.volume_confirmed else "⚠️ Weak"}</span></div>
                </div>
                """, unsafe_allow_html=True)

            for s in confirmed_br:
                c = add_confluence_filters(df, s)
                conf = sum([c.get("rsi_bullish",False), c.get("price_above_vwap",False), c.get("price_above_ema20",False)])
                arrow = "⬆" if s.direction == "bullish" else "⬇"
                bearish_cls = "" if s.direction == "bullish" else " bearish"
                st.markdown(f"""
                <div class="setup-card{bearish_cls}">
                  <h4>{arrow} B&R {s.direction.upper()} — {s.ticker}</h4>
                  <div><span class="label">LEVEL</span><span class="value">${s.key_level:.2f}</span></div>
                  <div><span class="label">ENTRY</span><span class="value">${s.entry_price:.2f}</span></div>
                  <div><span class="label">STOP </span><span class="value">${s.stop_loss:.2f}</span></div>
                  <div><span class="label">TGRT </span><span class="value">${s.target:.2f}</span></div>
                  <div><span class="label">R:R  </span><span class="value">{s.rr_ratio}:1</span></div>
                  <div><span class="label">CONF </span><span class="value">{"★"*conf}{"☆"*(3-conf)}</span></div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — OPTIONS CHAIN
# ══════════════════════════════════════════════
with tab2:
    if not all_confirmed:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; color:#2a4060;
                    text-align:center; padding:60px 0; font-size:0.85rem;">
          No confirmed setups to generate options recommendations.<br>
          Lower confluence threshold or adjust pattern filters.
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, setup in enumerate(all_confirmed[:4]):  # limit to 4 for perf
            pattern_name = "Double Bottom" if hasattr(setup, "bottom1_idx") else "Break & Retest"
            direction    = getattr(setup, "direction", "bullish")
            emoji        = "🟢" if direction == "bullish" else "🔴"

            with st.expander(f"{emoji}  {setup.ticker} — {pattern_name}  |  Entry ${setup.entry_price:.2f}  |  R:R {setup.rr_ratio}:1", expanded=(i==0)):
                col_opts, col_greeks = st.columns([1.6, 1])

                # Strategy selection logic (mirrors options_chain.py)
                if iv_rank > 60:
                    strategy = "Bull Put Spread (credit)" if direction == "bullish" else "Bear Call Spread (credit)"
                    strategy_note = f"IV Rank {iv_rank:.0f}% is HIGH — selling premium is favorable. Theta works for you."
                elif iv_rank < 30:
                    strategy = "Long Call" if direction == "bullish" else "Long Put"
                    strategy_note = f"IV Rank {iv_rank:.0f}% is LOW — buying premium is cheap. Full single-leg OK."
                else:
                    strategy = "Bull Call Spread" if direction == "bullish" else "Bear Put Spread"
                    strategy_note = f"IV Rank {iv_rank:.0f}% is NEUTRAL — debit vertical spread balances cost and risk."

                # Compute DTE options (21-45 day sweet spot)
                today       = date.today()
                exp_21      = (today + timedelta(days=21)).strftime("%Y-%m-%d")
                exp_30      = (today + timedelta(days=30)).strftime("%Y-%m-%d")
                exp_45      = (today + timedelta(days=45)).strftime("%Y-%m-%d")

                # Mock option prices (replace with live chain)
                price       = current_price
                long_strike = round(price * (0.98 if direction == "bullish" else 1.02) / 0.5) * 0.5
                short_strike= round(price * (1.05 if direction == "bullish" else 0.95) / 0.5) * 0.5
                long_mid    = round(price * 0.045, 2)
                short_mid   = round(long_mid * 0.45, 2)
                net_debit   = round(long_mid - short_mid, 2)
                spread_w    = abs(short_strike - long_strike)
                max_profit  = round((spread_w - net_debit) * 100, 2)
                max_loss    = round(net_debit * 100, 2)
                breakeven   = round(long_strike + net_debit if direction == "bullish" else long_strike - net_debit, 2)
                rr_spread   = round(max_profit / max_loss, 2) if max_loss else 0
                contracts   = max(1, int((account_size * risk_pct) / max_loss)) if max_loss else 1

                with col_opts:
                    st.markdown('<div class="section-header">RECOMMENDED STRUCTURE</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="options-box">
                      <div class="strategy-tag">{strategy.upper()}</div><br>
                      <span style="color:#4a90d9">EXPIRATION  </span><span style="color:#e8f4fd">{exp_30}  (30 DTE)</span><br>
                      <br>
                      <span style="color:#00d4aa">LONG  </span>
                      <span style="color:#4a90d9">{"CALL" if direction=="bullish" else "PUT"}  </span>
                      <span style="color:#e8f4fd">${long_strike:.2f}  strike</span>
                      <span style="color:#4a6fa5">  @  ${long_mid:.2f} mid</span><br>
                      <span style="color:#ff4d6d">SHORT </span>
                      <span style="color:#4a90d9">{"CALL" if direction=="bullish" else "PUT"}  </span>
                      <span style="color:#e8f4fd">${short_strike:.2f}  strike</span>
                      <span style="color:#4a6fa5">  @  ${short_mid:.2f} mid</span><br>
                      <br>
                      <span style="color:#4a90d9">NET DEBIT   </span><span style="color:#e8f4fd">${net_debit:.2f} / share</span>
                      <span style="color:#4a6fa5">  (${net_debit*100:.2f} / contract)</span><br>
                      <span style="color:#00d4aa">MAX PROFIT  </span><span style="color:#e8f4fd">${max_profit:.2f}</span><br>
                      <span style="color:#ff4d6d">MAX LOSS    </span><span style="color:#e8f4fd">${max_loss:.2f}</span><br>
                      <span style="color:#4a90d9">BREAKEVEN   </span><span style="color:#e8f4fd">${breakeven:.2f}</span><br>
                      <span style="color:#4a90d9">SPREAD R:R  </span><span style="color:#e8f4fd">{rr_spread}:1</span><br>
                      <br>
                      <span style="color:#4a90d9">CONTRACTS (1% risk) </span>
                      <span style="color:#e8f4fd">{contracts} contracts</span>
                      <span style="color:#4a6fa5"> on ${account_size:,} acct</span><br>
                      <span style="color:#4a90d9">TOTAL RISK  </span>
                      <span style="color:#e8f4fd">${contracts * max_loss:.2f}</span><br>
                      <br>
                      <span style="color:#f39c12">ℹ  {strategy_note}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_greeks:
                    st.markdown('<div class="section-header">GREEKS ESTIMATE</div>', unsafe_allow_html=True)

                    # Approximate Greeks for display
                    delta_long  = round(0.45 if direction == "bullish" else -0.45, 2)
                    delta_short = round(0.25 if direction == "bullish" else -0.25, 2)
                    net_delta   = round(delta_long - delta_short, 2)
                    theta_est   = round(-long_mid * 0.025, 3)
                    vega_est    = round(long_mid * 0.12, 3)
                    gamma_est   = round(0.015, 4)

                    greek_data = {
                        "": ["Long Leg", "Short Leg", "Net"],
                        "Delta": [delta_long, delta_short, net_delta],
                        "Theta/day": [theta_est, round(-theta_est*0.55,3), round(theta_est*0.45,3)],
                        "Vega": [vega_est, round(-vega_est*0.6,3), round(vega_est*0.4,3)],
                    }
                    greek_df = pd.DataFrame(greek_data)
                    st.dataframe(
                        greek_df.set_index(""),
                        use_container_width=True,
                        height=140,
                    )

                    st.markdown('<div class="section-header" style="margin-top:12px">EXPIRATION LADDER</div>', unsafe_allow_html=True)
                    ladder_data = {
                        "Expiry": [exp_21, exp_30, exp_45],
                        "DTE": [21, 30, 45],
                        "Est. Premium": [
                            f"${round(long_mid*0.82,2)}",
                            f"${round(long_mid,2)}",
                            f"${round(long_mid*1.22,2)}",
                        ],
                        "Theta/day": [
                            f"${round(abs(theta_est)*1.3,3)}",
                            f"${round(abs(theta_est),3)}",
                            f"${round(abs(theta_est)*0.7,3)}",
                        ],
                    }
                    st.dataframe(pd.DataFrame(ladder_data), use_container_width=True, height=160, hide_index=True)

                    st.markdown("""
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                                color:#2a4060; margin-top:10px; line-height:1.8;">
                      * Greeks estimated from delta model<br>
                      * Add API key for live chain data<br>
                      * Not financial advice
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — BACKTEST
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">WALK-FORWARD BACKTEST</div>', unsafe_allow_html=True)

    with st.spinner("Running backtest..."):
        report, eq_curve = run_backtest(
            df, selected_ticker,
            db_rr_min=min_rr,
            br_rr_min=min_rr,
            min_confluence_score=min_conf,
        )

    col_donut, col_metrics, col_curve = st.columns([1, 1.2, 2])

    with col_donut:
        st.plotly_chart(
            build_win_rate_donut(report.win_rate, report.wins, report.losses),
            use_container_width=True, config={"displayModeBar": False}
        )
        target_gap = 85 - report.win_rate
        if target_gap <= 0:
            st.markdown("""<div style="text-align:center; color:#00d4aa;
                font-family:'Share Tech Mono',monospace; font-size:0.7rem;">✅ TARGET MET</div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="text-align:center; color:#f39c12;
                font-family:'Share Tech Mono',monospace; font-size:0.7rem;">
                ⚠ {target_gap:.1f}% TO 85% TARGET</div>""",
                unsafe_allow_html=True)

    with col_metrics:
        st.markdown(f"""
        <div class="options-box" style="line-height:2.2;">
          <span style="color:#4a90d9">TOTAL TRADES  </span><span style="color:#e8f4fd">{report.total_trades}</span><br>
          <span style="color:#00d4aa">WINS          </span><span style="color:#e8f4fd">{report.wins}</span><br>
          <span style="color:#ff4d6d">LOSSES        </span><span style="color:#e8f4fd">{report.losses}</span><br>
          <span style="color:#4a90d9">AVG R:R       </span><span style="color:#e8f4fd">{report.avg_rr}</span><br>
          <span style="color:#4a90d9">AVG WIN       </span><span style="color:#00d4aa">{report.avg_win_pct}%</span><br>
          <span style="color:#4a90d9">AVG LOSS      </span><span style="color:#ff4d6d">{report.avg_loss_pct}%</span><br>
          <span style="color:#4a90d9">MAX DRAWDOWN  </span><span style="color:#ff4d6d">{report.max_drawdown}R</span><br>
          <span style="color:#4a90d9">EXPECTANCY    </span>
          <span style="color:{'#00d4aa' if report.expectancy > 0 else '#ff4d6d'}">{report.expectancy}R</span><br>
          <span style="color:#4a90d9">SHARPE        </span><span style="color:#e8f4fd">{report.sharpe}</span><br>
        </div>
        """, unsafe_allow_html=True)

        # Push toward 85% guidance
        st.markdown('<div class="section-header" style="margin-top:16px">OPTIMIZATION HINTS</div>', unsafe_allow_html=True)
        hints = []
        if report.win_rate < 85:
            if min_conf < 2:
                hints.append("↑ Raise confluence to 2+")
            if min_rr < 2.0:
                hints.append("↑ Raise min R:R to 2.0+")
            if report.total_trades > 30:
                hints.append("↓ Tighten DB tolerance")
            if report.avg_loss_pct < -5:
                hints.append("⚡ Add time-based exit rule")
        else:
            hints.append("✅ Parameters are solid")
            hints.append("✅ Scale with position sizing")

        for h in hints:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#4a90d9; padding:3px 0;">{h}</div>
            """, unsafe_allow_html=True)

    with col_curve:
        st.markdown('<div class="section-header">EQUITY CURVE (R)</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_equity_curve(eq_curve, report.win_rate),
            use_container_width=True, config={"displayModeBar": False}
        )

        if report.trades:
            st.markdown('<div class="section-header">TRADE LOG</div>', unsafe_allow_html=True)
            trade_rows = [{
                "Pattern":  t.pattern,
                "Entry":    f"${t.entry_price:.2f}",
                "Exit":     f"${t.exit_price:.2f}",
                "Outcome":  t.outcome.upper(),
                "P&L %":    f"{t.pnl_pct:+.2f}%",
                "R:R":      t.rr_ratio,
                "Conf":     t.confluence_score,
                "Bars":     t.bars_held,
            } for t in report.trades[-20:]]

            trade_df = pd.DataFrame(trade_rows)
            st.dataframe(trade_df, use_container_width=True, height=260, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — WATCHLIST SCAN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">FULL WATCHLIST SCAN</div>', unsafe_allow_html=True)

    scan_triggered = run_scan or st.button("🔄 SCAN ALL TICKERS", use_container_width=False)

    if scan_triggered:
        results = []
        prog = st.progress(0, text="Scanning watchlist...")

        for i, ticker in enumerate(WATCHLIST):
            prog.progress((i + 1) / len(WATCHLIST), text=f"Scanning {ticker}...")
            try:
                tdf = fetch_bars(ticker, timespan=timespan, days_back=days_back)
                if tdf.empty or len(tdf) < 30:
                    continue

                db  = [s for s in detect_double_bottom(tdf, ticker, rr_min=min_rr) if s.confirmed]
                br  = [s for s in detect_break_and_retest(tdf, ticker, rr_min=min_rr) if s.confirmed]
                all_s = db + br

                iv_r, _, _ = get_iv_rank_cached(ticker)
                price_now  = float(tdf["close"].iloc[-1])
                pct_chg    = float((tdf["close"].iloc[-1] - tdf["close"].iloc[-2]) / tdf["close"].iloc[-2] * 100)

                for s in all_s:
                    c = add_confluence_filters(tdf, s)
                    cs = sum([c.get("rsi_bullish",False), c.get("price_above_vwap",False), c.get("price_above_ema20",False)])
                    if cs < min_conf:
                        continue
                    direction = getattr(s, "direction", "bullish")
                    pattern   = "Double Bottom" if hasattr(s, "bottom1_idx") else "Break & Retest"
                    results.append({
                        "Ticker":    ticker,
                        "Pattern":   pattern,
                        "Direction": direction.upper(),
                        "Price":     f"${price_now:.2f}",
                        "Chg %":     f"{pct_chg:+.2f}%",
                        "Entry":     f"${s.entry_price:.2f}",
                        "Stop":      f"${s.stop_loss:.2f}",
                        "Target":    f"${s.target:.2f}",
                        "R:R":       s.rr_ratio,
                        "IV Rank":   f"{iv_r:.0f}%",
                        "Conf":      "★" * cs + "☆" * (3 - cs),
                    })

                if not all_s:
                    results.append({
                        "Ticker":    ticker,
                        "Pattern":   "—",
                        "Direction": "—",
                        "Price":     f"${price_now:.2f}",
                        "Chg %":     f"{pct_chg:+.2f}%",
                        "Entry":     "—",
                        "Stop":      "—",
                        "Target":    "—",
                        "R:R":       "—",
                        "IV Rank":   f"{iv_r:.0f}%",
                        "Conf":      "—",
                    })

            except Exception as e:
                results.append({"Ticker": ticker, "Pattern": f"ERROR: {e}", **{k: "—" for k in
                    ["Direction","Price","Chg %","Entry","Stop","Target","R:R","IV Rank","Conf"]}})

        prog.empty()

        if results:
            scan_df = pd.DataFrame(results)
            setup_rows = scan_df[scan_df["Pattern"] != "—"]
            empty_rows = scan_df[scan_df["Pattern"] == "—"]

            hits = len(setup_rows)
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                        color:#00d4aa; margin-bottom:12px;">
              ✅  Scan complete — {hits} setup{"s" if hits != 1 else ""} found
              across {len(WATCHLIST)} tickers
            </div>
            """, unsafe_allow_html=True)

            if not setup_rows.empty:
                st.markdown('<div class="section-header">🚨 ACTIVE SETUPS</div>', unsafe_allow_html=True)
                st.dataframe(setup_rows, use_container_width=True, hide_index=True, height=300)

            st.markdown('<div class="section-header">FULL WATCHLIST STATUS</div>', unsafe_allow_html=True)
            st.dataframe(scan_df, use_container_width=True, hide_index=True, height=400)

    else:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; color:#2a4060;
                    text-align:center; padding:60px 0; font-size:0.8rem; line-height:2;">
          Click  ⚡ RUN SCAN  or  🔄 SCAN ALL TICKERS<br>
          to screen all {n} watchlist tickers simultaneously
        </div>
        """.format(n=len(WATCHLIST)), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────

st.markdown("""
<div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#1e3a5f;
            text-align:center; padding:24px 0 8px 0; border-top:1px solid #0d1a2e; margin-top:24px;">
  OPTIONS SCREENER v1.0  ·  DEMO MODE  ·  NOT FINANCIAL ADVICE  ·  VERIFY ALL SETUPS BEFORE PLACING
</div>
""", unsafe_allow_html=True)
