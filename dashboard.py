
"""
Options Screener - Simplified Dashboard
Straight calls and puts only. No spreads.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests

from pattern_detection import (
    detect_double_bottom,
    detect_break_and_retest,
    add_confluence_filters,
)
from backtester import run_backtest

st.set_page_config(
    page_title="Options Screener",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');
* { font-family: 'Barlow', sans-serif; }
body, .stApp { background: #0a0e17; color: #e0e6f0; }
.stSidebar { background: #0d1219 !important; border-right: 1px solid #1e2d40; }
.metric-card { background: #111827; border: 1px solid #1e2d40; border-radius: 8px; padding: 16px; margin: 6px 0; }
.section-title { color: #00d4aa; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; letter-spacing: 2px; margin: 20px 0 8px; border-bottom: 1px solid #1e2d40; padding-bottom: 4px; }
.trade-box { background: #111827; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #00d4aa; }
.trade-box.bear { border-left-color: #ff4d6d; }
.big-price { font-size: 2rem; font-weight: 700; color: #e0e6f0; }
</style>
""", unsafe_allow_html=True)

POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", "")

WATCHLIST = ["PLTR", "NBIS", "VRT", "CRDO", "GOOGL", "AAOI", "ASTS", "ZETA", "SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

TIMEFRAMES = {
    "5 Min": ("minute", 5, 2),
    "15 Min": ("minute", 15, 5),
    "1 Hour": ("hour", 1, 14),
    "4 Hour": ("hour", 4, 30),
    "Daily": ("day", 1, 90),
}

@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, multiplier, timespan, days_back):
    if not POLYGON_API_KEY:
        return _demo_data(ticker)
    end = datetime.now()
    start = end - timedelta(days=days_back)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
           f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_API_KEY}")
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if not data.get("results"):
            return _demo_data(ticker)
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    except:
        return _demo_data(ticker)

def _demo_data(ticker, bars=200):
    np.random.seed(hash(ticker) % 999)
    prices = {"PLTR": 118, "NBIS": 45, "VRT": 92, "CRDO": 68, "GOOGL": 175,
              "AAOI": 22, "ASTS": 28, "ZETA": 19, "SPY": 570, "QQQ": 490,
              "NVDA": 138, "TSLA": 320, "AAPL": 228}.get(ticker, 100)
    dates = pd.date_range(end=datetime.now(), periods=bars, freq="1h")
    close = [prices]
    for _ in range(bars - 1):
        close.append(close[-1] * (1 + np.random.normal(0, 0.012)))
    close = np.array(close)
    hi = close * (1 + np.abs(np.random.normal(0, 0.008, bars)))
    lo = close * (1 - np.abs(np.random.normal(0, 0.008, bars)))
    op = lo + np.random.uniform(0, 1, bars) * (hi - lo)
    vol = np.random.randint(500000, 3000000, bars)
    return pd.DataFrame({"timestamp": dates, "open": op, "high": hi, "low": lo, "close": close, "volume": vol})

@st.cache_data(ttl=300)
def fetch_current_price(ticker):
    if not POLYGON_API_KEY:
        return {"PLTR": 118.42, "NBIS": 45.20, "VRT": 92.10, "CRDO": 68.50,
                "GOOGL": 175.30, "AAOI": 22.10, "ASTS": 28.40, "ZETA": 19.80,
                "SPY": 570.20, "QQQ": 490.50, "NVDA": 138.60, "TSLA": 320.10, "AAPL": 228.40}.get(ticker, 100.0)
    try:
        url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={POLYGON_API_KEY}"
        r = requests.get(url, timeout=5)
        return r.json().get("results", {}).get("p", 0)
    except:
        return 0

def calc_call(price, days_to_exp=30, iv=0.45, account=10000, risk_pct=0.01):
    strike = round(price * 1.05 / 0.5) * 0.5
    premium = round(price * iv * (days_to_exp / 365) ** 0.5 * 0.4, 2)
    breakeven = strike + premium
    max_loss = premium * 100
    contracts = max(1, int((account * risk_pct) / max_loss))
    target_price = price * 1.15
    profit_at_target = max(0, (target_price - strike - premium) * 100 * contracts)
    return {"strike": strike, "premium": premium, "breakeven": round(breakeven, 2),
            "max_loss_per_contract": max_loss, "contracts": contracts,
            "total_risk": round(max_loss * contracts, 2),
            "expiration": (date.today() + timedelta(days=days_to_exp)).strftime("%b %d, %Y"),
            "profit_at_target": round(profit_at_target, 2), "target_price": round(target_price, 2)}

def calc_put(price, days_to_exp=30, iv=0.45, account=10000, risk_pct=0.01):
    strike = round(price * 0.95 / 0.5) * 0.5
    premium = round(price * iv * (days_to_exp / 365) ** 0.5 * 0.4, 2)
    breakeven = strike - premium
    max_loss = premium * 100
    contracts = max(1, int((account * risk_pct) / max_loss))
    target_price = price * 0.85
    profit_at_target = max(0, (strike - target_price - premium) * 100 * contracts)
    return {"strike": strike, "premium": premium, "breakeven": round(breakeven, 2),
            "max_loss_per_contract": max_loss, "contracts": contracts,
            "total_risk": round(max_loss * contracts, 2),
            "expiration": (date.today() + timedelta(days=days_to_exp)).strftime("%b %d, %Y"),
            "profit_at_target": round(profit_at_target, 2), "target_price": round(target_price, 2)}

# Sidebar
with st.sidebar:
    st.markdown("## 📡 OPTIONS SCREENER")
    st.markdown("---")
    selected_ticker = st.selectbox("TICKER", WATCHLIST, index=0)
    custom = st.text_input("Or type any ticker", "").upper().strip()
    if custom:
        selected_ticker = custom
    selected_tf = st.selectbox("TIMEFRAME", list(TIMEFRAMES.keys()), index=2)
    st.markdown("---")
    st.markdown("**ACCOUNT SETTINGS**")
    account_size = st.number_input("Account Size ($)", value=10000, step=1000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    dte = st.selectbox("Days to Expiration", [14, 21, 30, 45, 60], index=2)
    st.markdown("---")
    if POLYGON_API_KEY:
        st.success("🟢 LIVE DATA")
    else:
        st.warning("🟡 DEMO MODE")

tf_mult, tf_span, tf_days = TIMEFRAMES[selected_tf]
df = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)
current_price = fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else current_price
pct_change = ((current_price - prev_close) / prev_close) * 100

# Header
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    color = "#00d4aa" if pct_change >= 0 else "#ff4d6d"
    arrow = "▲" if pct_change >= 0 else "▼"
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.8rem'>{selected_ticker} · {selected_tf}</div><div class='big-price'>${current_price:,.2f}</div><div style='color:{color}'>{arrow} {pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
with col2:
    ema20 = df["close"].ewm(span=20).mean().iloc[-1]
    above = current_price > ema20
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>TREND</div><div style='font-size:1.2rem;font-weight:700;color:{'#00d4aa' if above else '#ff4d6d'}'>{'BULLISH ▲' if above else 'BEARISH ▼'}</div></div>", unsafe_allow_html=True)
with col3:
    vol = df["volume"].iloc[-1]
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>VOLUME</div><div style='font-size:1.4rem;font-weight:700'>{vol/1e6:.1f}M</div></div>", unsafe_allow_html=True)
with col4:
    high = df["high"].max()
    low = df["low"].min()
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>RANGE</div><div style='color:#00d4aa'>H: ${high:.2f}</div><div style='color:#ff4d6d'>L: ${low:.2f}</div></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📈 CHART & SETUPS", "🎯 CALL / PUT BUILDER", "📊 BACKTEST", "🔍 WATCHLIST SCAN"])

with tab1:
    db_setups = [s for s in detect_double_bottom(df, selected_ticker, rr_min=2.0) if s.confirmed]
    br_setups = [s for s in detect_break_and_retest(df, selected_ticker, rr_min=2.0) if s.confirmed]
    all_setups = db_setups + br_setups

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=selected_ticker,
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00d4aa", decreasing_fillcolor="#ff4d6d"), row=1, col=1)
    ema = df["close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=ema, name="EMA 20",
        line=dict(color="#f0c040", width=1.5, dash="dot")), row=1, col=1)
    colors = ["#00d4aa" if c >= o else "#ff4d6d" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors, opacity=0.6, name="Volume"), row=2, col=1)
    for s in db_setups[:2]:
        fig.add_hline(y=s.neckline, line_dash="dash", line_color="#00d4aa", annotation_text="Neckline", row=1, col=1)
        fig.add_hline(y=s.stop_loss, line_dash="dot", line_color="#ff4d6d", annotation_text="Stop", row=1, col=1)
    fig.update_layout(paper_bgcolor="#0a0e17", plot_bgcolor="#0d1219", font=dict(color="#e0e6f0"),
        height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(gridcolor="#1e2d40")
    fig.update_yaxes(gridcolor="#1e2d40")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>DETECTED SETUPS</div>", unsafe_allow_html=True)
    if not all_setups:
        st.info("No confirmed setups right now. Try a different timeframe or ticker.")
    else:
        for s in all_setups[:5]:
            ptype = "Double Bottom" if hasattr(s, "bottom1_idx") else "Break & Retest"
            direction = getattr(s, "direction", "bullish")
            is_bull = direction == "bullish"
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown(f"<div class='trade-box {'bear' if not is_bull else ''}'><div style='color:#8899aa;font-size:0.75rem'>{ptype}</div><div style='font-size:1.1rem;font-weight:700;color:{'#00d4aa' if is_bull else '#ff4d6d'}'>{'📈 BUY CALL' if is_bull else '📉 BUY PUT'}</div></div>", unsafe_allow_html=True)
            with cb:
                st.metric("Entry", f"${s.entry_price:.2f}")
                st.metric("Stop Loss", f"${s.stop_loss:.2f}")
            with cc:
                st.metric("Target", f"${s.target:.2f}")
                st.metric("R:R", f"{s.rr_ratio:.1f}x")

with tab2:
    st.markdown("<div class='section-title'>CALL / PUT BUILDER</div>", unsafe_allow_html=True)
    call = calc_call(current_price, dte, 0.45, account_size, risk_pct)
    put = calc_put(current_price, dte, 0.45, account_size, risk_pct)
    cl, cr = st.columns(2)
    with cl:
        st.markdown(f"""<div class='trade-box'>
        <div style='font-size:1.3rem;font-weight:700;color:#00d4aa'>📈 CALL — Betting it goes UP</div>
        <br>
        <b>Strike:</b> ${call['strike']:.2f} &nbsp;|&nbsp; <b>Premium:</b> ${call['premium']:.2f}/share<br>
        <b>Breakeven:</b> ${call['breakeven']:.2f} &nbsp;|&nbsp; <b>Expiration:</b> {call['expiration']}<br><br>
        <b>Contracts to buy (1% risk):</b> <span style='color:#00d4aa;font-size:1.3rem'>{call['contracts']}</span><br>
        <b>Total you could lose:</b> <span style='color:#ff4d6d'>${call['total_risk']:.0f}</span><br><br>
        <div style='border-top:1px solid #1e2d40;padding-top:10px'>
        If {selected_ticker} hits ${call['target_price']:.2f} (+15%)<br>
        <span style='color:#00d4aa;font-size:1.3rem;font-weight:700'>Profit: ${call['profit_at_target']:,.0f}</span>
        </div></div>""", unsafe_allow_html=True)
    with cr:
        st.markdown(f"""<div class='trade-box bear'>
        <div style='font-size:1.3rem;font-weight:700;color:#ff4d6d'>📉 PUT — Betting it goes DOWN</div>
        <br>
        <b>Strike:</b> ${put['strike']:.2f} &nbsp;|&nbsp; <b>Premium:</b> ${put['premium']:.2f}/share<br>
        <b>Breakeven:</b> ${put['breakeven']:.2f} &nbsp;|&nbsp; <b>Expiration:</b> {put['expiration']}<br><br>
        <b>Contracts to buy (1% risk):</b> <span style='color:#ff4d6d;font-size:1.3rem'>{put['contracts']}</span><br>
        <b>Total you could lose:</b> <span style='color:#ff4d6d'>${put['total_risk']:.0f}</span><br><br>
        <div style='border-top:1px solid #1e2d40;padding-top:10px'>
        If {selected_ticker} hits ${put['target_price']:.2f} (-15%)<br>
        <span style='color:#00d4aa;font-size:1.3rem;font-weight:700'>Profit: ${put['profit_at_target']:,.0f}</span>
        </div></div>""", unsafe_allow_html=True)
    st.info("Use the Chart tab to find setups. Bullish setup = buy CALL. Bearish setup = buy PUT. Always verify premiums on your broker before trading.")

with tab3:
    st.markdown("<div class='section-title'>BACKTEST</div>", unsafe_allow_html=True)
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Analyzing..."):
            report, equity = run_backtest(df, selected_ticker)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{report.win_rate}%")
        c2.metric("Trades", report.total_trades)
        c3.metric("Avg R:R", f"{report.avg_rr}x")
        c4.metric("Expectancy", f"{report.expectancy}R")
        if len(equity) > 1:
            fig_eq = go.Figure(go.Scatter(y=equity, mode="lines+markers",
                line=dict(color="#00d4aa", width=2), fill="tozeroy", fillcolor="rgba(0,212,170,0.1)"))
            fig_eq.update_layout(paper_bgcolor="#0a0e17", plot_bgcolor="#0d1219",
                font=dict(color="#e0e6f0"), height=300, title="Equity Curve (R multiples)",
                margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_eq, use_container_width=True)
        if report.trades:
            rows = [{"Pattern": t.pattern, "Result": t.outcome.upper(),
                     "Entry": f"${t.entry_price:.2f}", "Exit": f"${t.exit_price:.2f}",
                     "P&L": f"{t.pnl_pct:+.1f}%"} for t in report.trades]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Click Run Backtest to analyze this ticker.")

with tab4:
    st.markdown("<div class='section-title'>WATCHLIST SCAN</div>", unsafe_allow_html=True)
    if st.button("🔍 SCAN ALL TICKERS", type="primary"):
        results = []
        prog = st.progress(0)
        status = st.empty()
        for i, ticker in enumerate(WATCHLIST):
            status.text(f"Scanning {ticker}...")
            prog.progress((i + 1) / len(WATCHLIST))
            try:
                tdf = fetch_ohlcv(ticker, tf_mult, tf_span, tf_days)
                price = fetch_current_price(ticker) or float(tdf["close"].iloc[-1])
                db = [s for s in detect_double_bottom(tdf, ticker, rr_min=2.0) if s.confirmed]
                br = [s for s in detect_break_and_retest(tdf, ticker, rr_min=2.0) if s.confirmed]
                all_s = db + br
                if all_s:
                    best = max(all_s, key=lambda x: x.rr_ratio)
                    direction = getattr(best, "direction", "bullish")
                    ptype = "Double Bottom" if hasattr(best, "bottom1_idx") else "Break & Retest"
                    results.append({"Ticker": ticker, "Price": f"${price:.2f}", "Setup": ptype,
                        "Action": "📈 BUY CALL" if direction == "bullish" else "📉 BUY PUT",
                        "Entry": f"${best.entry_price:.2f}", "Stop": f"${best.stop_loss:.2f}",
                        "Target": f"${best.target:.2f}", "R:R": f"{best.rr_ratio}x"})
            except:
                pass
        prog.empty()
        status.empty()
        if results:
            st.success(f"Found {len(results)} active setups!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No confirmed setups right now. Try again later.")

st.markdown("<div style='text-align:center;padding:20px;color:#8899aa;font-size:0.75rem;border-top:1px solid #1e2d40'>OPTIONS SCREENER v2.0 · NOT FINANCIAL ADVICE</div>", unsafe_allow_html=True)
