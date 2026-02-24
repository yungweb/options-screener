"""
Options Screener v3.0 - Traffic Light Signal System
5-factor confluence scoring with confidence percentage
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import os

from pattern_detection import detect_double_bottom, detect_break_and_retest
from backtester import run_backtest

st.set_page_config(page_title="Options Screener", page_icon="📡", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');
* { font-family: 'Barlow', sans-serif; }
body, .stApp { background: #0a0e17; color: #e0e6f0; }
.stSidebar { background: #0d1219 !important; border-right: 1px solid #1e2d40; }
.big-price { font-size: 2rem; font-weight: 700; }
.section-title { color: #00d4aa; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; letter-spacing: 2px; margin: 20px 0 8px; border-bottom: 1px solid #1e2d40; padding-bottom: 4px; }

/* Signal cards */
.signal-green { background: #0a1f1a; border: 2px solid #00d4aa; border-radius: 12px; padding: 20px; margin: 10px 0; }
.signal-yellow { background: #1a1a0a; border: 2px solid #f0c040; border-radius: 12px; padding: 20px; margin: 10px 0; }
.signal-red { background: #1a0a0a; border: 2px solid #ff4d6d; border-radius: 12px; padding: 20px; margin: 10px 0; }
.signal-gray { background: #111827; border: 2px solid #1e2d40; border-radius: 12px; padding: 20px; margin: 10px 0; }

.conf-number { font-size: 3rem; font-weight: 700; line-height: 1; }
.conf-green { color: #00d4aa; }
.conf-yellow { color: #f0c040; }
.conf-red { color: #ff4d6d; }
.conf-gray { color: #8899aa; }

.factor-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.9rem; }
.dot-green { width: 10px; height: 10px; background: #00d4aa; border-radius: 50%; display: inline-block; }
.dot-red { width: 10px; height: 10px; background: #ff4d6d; border-radius: 50%; display: inline-block; }
.dot-gray { width: 10px; height: 10px; background: #1e2d40; border-radius: 50%; display: inline-block; }

.trade-action { background: #0a1f1a; border-radius: 8px; padding: 16px; margin-top: 12px; }
.trade-action.bear { background: #1f0a10; }
.strike-price { font-size: 1.4rem; font-weight: 700; color: #00d4aa; }
.strike-price.bear { color: #ff4d6d; }

.metric-card { background: #111827; border: 1px solid #1e2d40; border-radius: 8px; padding: 14px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
WATCHLIST = ["PLTR", "NBIS", "VRT", "CRDO", "GOOGL", "AAOI", "ASTS", "ZETA", "SPY", "QQQ", "NVDA", "TSLA", "AAPL"]
TIMEFRAMES = {
    "5 Min":  ("minute", 5, 2),
    "15 Min": ("minute", 15, 5),
    "1 Hour": ("hour", 1, 14),
    "4 Hour": ("hour", 4, 30),
    "Daily":  ("day", 1, 90),
}

# ── Data ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, multiplier, timespan, days_back):
    try:
        import yfinance as yf
        intervals = {"minute": "5m", "hour": "1h", "day": "1d"}
        interval = intervals.get(timespan, "1h")
        period = f"{min(days_back, 59)}d" if timespan == "minute" else f"{days_back}d"
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return _demo_data(ticker)
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df.rename(columns={"datetime": "timestamp", "date": "timestamp"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    except:
        return _demo_data(ticker)

def _demo_data(ticker, bars=200):
    np.random.seed(hash(ticker) % 999)
    prices = {"PLTR": 118, "NBIS": 98, "VRT": 92, "CRDO": 68, "GOOGL": 175,
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

def fetch_current_price(ticker):
    try:
        import yfinance as yf
        price = yf.Ticker(ticker).fast_info["last_price"]
        return round(float(price), 2)
    except:
        return None

# ── Confluence scoring ────────────────────────────────
def score_confluence(df, setup):
    close = df["close"]
    price = float(close.iloc[-1])
    direction = getattr(setup, "direction", "bullish")
    is_bull = direction == "bullish"
    factors = {}

    # 1 — Pattern confirmed
    factors["pattern"] = {"pass": setup.confirmed, "label": "Pattern confirmed", "detail": "Double bottom or break & retest fully formed"}

    # 2 — RSI 40-60
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    rsi_pass = 35 < rsi < 65
    factors["rsi"] = {"pass": rsi_pass, "label": f"RSI in zone ({rsi:.0f})", "detail": "RSI between 35-65 — not overbought or oversold"}

    # 3 — VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = float((tp * df["volume"]).cumsum().iloc[-1] / df["volume"].cumsum().iloc[-1])
    vwap_pass = price > vwap if is_bull else price < vwap
    factors["vwap"] = {"pass": vwap_pass, "label": f"Price {'above' if is_bull else 'below'} VWAP (${vwap:.2f})", "detail": "Price on correct side of VWAP"}

    # 4 — Volume spike
    avg_vol = float(df["volume"].iloc[-20:].mean())
    cur_vol = float(df["volume"].iloc[-1])
    vol_pass = cur_vol > avg_vol * 1.2
    factors["volume"] = {"pass": vol_pass, "label": f"Volume spike ({cur_vol/1e6:.1f}M vs avg {avg_vol/1e6:.1f}M)", "detail": "Current volume 20%+ above average"}

    # 5 — EMA alignment
    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    ema_pass = price > ema20 if is_bull else price < ema20
    factors["ema"] = {"pass": ema_pass, "label": f"Price {'above' if is_bull else 'below'} EMA 20 (${ema20:.2f})", "detail": "Price on correct side of 20 EMA"}

    score = sum(1 for f in factors.values() if f["pass"])
    confidence = int((score / 5) * 100)
    return factors, score, confidence, rsi, vwap, ema20

def calc_option(price, direction, days_to_exp, iv=0.45, account=10000, risk_pct=0.01):
    is_call = direction == "bullish"
    strike = round(price * (1.05 if is_call else 0.95) / 0.5) * 0.5
    premium = round(price * iv * (days_to_exp / 365) ** 0.5 * 0.4, 2)
    breakeven = (strike + premium) if is_call else (strike - premium)
    max_loss = premium * 100
    contracts = max(1, int((account * risk_pct) / max_loss))
    target_price = price * (1.15 if is_call else 0.85)
    profit = max(0, (abs(target_price - strike) - premium) * 100 * contracts)
    return {"type": "CALL" if is_call else "PUT", "strike": strike, "premium": premium,
            "breakeven": round(breakeven, 2), "max_loss": round(max_loss * contracts, 2),
            "contracts": contracts, "target_price": round(target_price, 2),
            "profit_at_target": round(profit, 2),
            "expiration": (date.today() + timedelta(days=days_to_exp)).strftime("%b %d, %Y")}

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 OPTIONS SCREENER")
    st.markdown("---")
    selected_ticker = st.selectbox("TICKER", WATCHLIST)
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

# ── Load data ─────────────────────────────────────────
tf_mult, tf_span, tf_days = TIMEFRAMES[selected_tf]
df = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)
current_price = fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else current_price
pct_change = ((current_price - prev_close) / prev_close) * 100

# ── Header ────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    color = "#00d4aa" if pct_change >= 0 else "#ff4d6d"
    arrow = "▲" if pct_change >= 0 else "▼"
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.8rem'>{selected_ticker} · {selected_tf}</div><div class='big-price'>${current_price:,.2f}</div><div style='color:{color}'>{arrow} {pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
with col2:
    ema20 = float(df["close"].ewm(span=20).mean().iloc[-1])
    above = current_price > ema20
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>TREND</div><div style='font-weight:700;color:{'#00d4aa' if above else '#ff4d6d'}'>{'BULLISH ▲' if above else 'BEARISH ▼'}</div></div>", unsafe_allow_html=True)
with col3:
    vol = float(df["volume"].iloc[-1])
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>VOLUME</div><div style='font-weight:700'>{vol/1e6:.1f}M</div></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🚦 SIGNAL", "📈 CHART", "📊 BACKTEST", "🔍 SCAN"])

# ── TAB 1: Signal ─────────────────────────────────────
with tab1:
    db_setups = [s for s in detect_double_bottom(df, selected_ticker, rr_min=2.0) if s.confirmed]
    br_setups = [s for s in detect_break_and_retest(df, selected_ticker, rr_min=2.0) if s.confirmed]
    all_setups = db_setups + br_setups

    if not all_setups:
        st.markdown(f"""
        <div class='signal-gray'>
            <div class='conf-number conf-gray'>0%</div>
            <div style='font-size:1.1rem;font-weight:700;color:#8899aa;margin:8px 0'>NO SETUP DETECTED</div>
            <div style='color:#8899aa;font-size:0.9rem'>No double bottom or break & retest pattern found for {selected_ticker} on the {selected_tf} timeframe.</div>
            <div style='color:#8899aa;font-size:0.85rem;margin-top:12px'>Try switching to Daily or 4 Hour timeframe, or check a different ticker.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        best_setup = max(all_setups, key=lambda s: s.rr_ratio)
        factors, score, confidence, rsi, vwap, ema20_val = score_confluence(df, best_setup)
        direction = getattr(best_setup, "direction", "bullish")
        is_bull = direction == "bullish"
        ptype = "Double Bottom" if hasattr(best_setup, "bottom1_idx") else "Break & Retest"

        # Color based on confidence
        if confidence >= 80:
            card_class = "signal-green"
            conf_class = "conf-green"
            emoji = "🟢"
            headline = "STRONG SIGNAL — ACT NOW"
        elif confidence >= 60:
            card_class = "signal-yellow"
            conf_class = "conf-yellow"
            emoji = "🟡"
            headline = "WATCH CLOSELY — ALMOST READY"
        elif confidence >= 40:
            card_class = "signal-yellow"
            conf_class = "conf-yellow"
            emoji = "⚠️"
            headline = "PATTERN FORMING — NOT YET"
        else:
            card_class = "signal-red"
            conf_class = "conf-red"
            emoji = "🔴"
            headline = "STAY OUT — CONDITIONS NOT MET"

        # Missing factors
        missing = [f["label"] for f in factors.values() if not f["pass"]]
        passing = [f["label"] for f in factors.values() if f["pass"]]

        # Factor dots HTML
        dots_html = ""
        for key, f in factors.items():
            dot = "dot-green" if f["pass"] else "dot-red"
            dots_html += f"<div class='factor-row'><span class='{dot}'></span><span style='color:{'#e0e6f0' if f['pass'] else '#8899aa'}'>{f['label']}</span></div>"

        st.markdown(f"""
        <div class='{card_class}'>
            <div style='display:flex;align-items:center;gap:16px;margin-bottom:12px'>
                <div class='conf-number {conf_class}'>{confidence}%</div>
                <div>
                    <div style='font-size:1.1rem;font-weight:700'>{emoji} {headline}</div>
                    <div style='color:#8899aa;font-size:0.85rem'>{ptype} · {'📈 Bullish' if is_bull else '📉 Bearish'}</div>
                </div>
            </div>
            {dots_html}
        </div>
        """, unsafe_allow_html=True)

        # What's missing
        if missing and confidence < 100:
            st.markdown(f"<div style='background:#111827;border:1px solid #1e2d40;border-radius:8px;padding:12px;margin:8px 0;color:#f0c040;font-size:0.85rem'>⏳ <b>Waiting on:</b> {' · '.join(missing)}</div>", unsafe_allow_html=True)

        # Full signal action card
        if confidence >= 80:
            opt = calc_option(current_price, direction, dte, account=account_size, risk_pct=risk_pct)
            opt_color = "#00d4aa" if is_bull else "#ff4d6d"
            st.markdown(f"""
            <div class='trade-action {'bear' if not is_bull else ''}'>
                <div style='font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:8px'>YOUR TRADE</div>
                <div style='font-size:1.3rem;font-weight:700;color:{opt_color}'>
                    {'📈 BUY CALL' if is_bull else '📉 BUY PUT'} — {selected_ticker}
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px'>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>STRIKE</div>
                        <div class='strike-price {'bear' if not is_bull else ''}'>${opt['strike']:.2f}</div>
                    </div>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>PAY MAX</div>
                        <div style='font-size:1.3rem;font-weight:700'>${opt['premium']:.2f}/share</div>
                    </div>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>EXIT TARGET</div>
                        <div style='font-size:1.2rem;font-weight:700;color:#00d4aa'>${opt['target_price']:.2f}</div>
                    </div>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>STOP OUT IF BELOW</div>
                        <div style='font-size:1.2rem;font-weight:700;color:#ff4d6d'>${best_setup.stop_loss:.2f}</div>
                    </div>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>CONTRACTS</div>
                        <div style='font-size:1.3rem;font-weight:700;color:{opt_color}'>{opt['contracts']}</div>
                    </div>
                    <div>
                        <div style='color:#8899aa;font-size:0.75rem'>MAX LOSS</div>
                        <div style='font-size:1.2rem;font-weight:700;color:#ff4d6d'>${opt['max_loss']:.0f}</div>
                    </div>
                </div>
                <div style='margin-top:14px;padding-top:12px;border-top:1px solid #1e2d40'>
                    <div style='color:#8899aa;font-size:0.75rem'>EXPIRES</div>
                    <div style='font-weight:700'>{opt['expiration']}</div>
                </div>
                <div style='margin-top:12px;padding-top:12px;border-top:1px solid #1e2d40'>
                    <div style='color:#8899aa;font-size:0.75rem'>POTENTIAL PROFIT AT TARGET</div>
                    <div style='font-size:1.4rem;font-weight:700;color:#00d4aa'>${opt['profit_at_target']:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif confidence >= 60:
            opt = calc_option(current_price, direction, dte, account=account_size, risk_pct=risk_pct)
            st.markdown(f"""
            <div style='background:#1a1a0a;border:1px solid #f0c040;border-radius:8px;padding:16px;margin-top:10px'>
                <div style='color:#f0c040;font-weight:700;margin-bottom:8px'>👀 SET AN ALERT — ENTRY LIKELY NEAR</div>
                <div style='font-size:1.2rem;font-weight:700'>{'📈 CALL' if is_bull else '📉 PUT'} around <span style='color:{"#00d4aa" if is_bull else "#ff4d6d"}'>${opt['strike']:.2f} strike</span></div>
                <div style='color:#8899aa;margin-top:8px;font-size:0.85rem'>Check back in 1-2 bars. If {', '.join(missing[:2])} confirm, this becomes a full signal.</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:#1a0a0a;border:1px solid #ff4d6d;border-radius:8px;padding:16px;margin-top:10px'>
                <div style='color:#ff4d6d;font-weight:700'>🚫 DO NOT ENTER YET</div>
                <div style='color:#8899aa;margin-top:8px;font-size:0.85rem'>Too many signals missing. Wait for at least 4 out of 5 to confirm before considering a trade.</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Chart ──────────────────────────────────────
with tab2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=selected_ticker,
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00d4aa", decreasing_fillcolor="#ff4d6d"), row=1, col=1)
    ema = df["close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=ema, name="EMA 20",
        line=dict(color="#f0c040", width=1.5, dash="dot")), row=1, col=1)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap_line = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=vwap_line, name="VWAP",
        line=dict(color="#9966ff", width=1.5, dash="dash")), row=1, col=1)
    colors = ["#00d4aa" if c >= o else "#ff4d6d" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors, opacity=0.6, name="Volume"), row=2, col=1)
    fig.update_layout(paper_bgcolor="#0a0e17", plot_bgcolor="#0d1219", font=dict(color="#e0e6f0"),
        height=480, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1e2d40")
    fig.update_yaxes(gridcolor="#1e2d40")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Backtest ───────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>BACKTEST</div>", unsafe_allow_html=True)
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Analyzing historical patterns..."):
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
                font=dict(color="#e0e6f0"), height=280, title="Equity Curve",
                margin=dict(l=0, r=0, t=40, b=0))
            fig_eq.update_xaxes(gridcolor="#1e2d40")
            fig_eq.update_yaxes(gridcolor="#1e2d40")
            st.plotly_chart(fig_eq, use_container_width=True)
        if report.trades:
            rows = [{"Pattern": t.pattern, "Result": t.outcome.upper(),
                     "Entry": f"${t.entry_price:.2f}", "Exit": f"${t.exit_price:.2f}",
                     "P&L": f"{t.pnl_pct:+.1f}%"} for t in report.trades]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Click Run Backtest to see how this pattern performed historically on this ticker.")

# ── TAB 4: Scan ───────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>WATCHLIST SCAN</div>", unsafe_allow_html=True)
    st.markdown("Scans all tickers and ranks by confidence score.")
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
                    _, score, confidence, _, _, _ = score_confluence(tdf, best)
                    direction = getattr(best, "direction", "bullish")
                    ptype = "Double Bottom" if hasattr(best, "bottom1_idx") else "Break & Retest"
                    if confidence >= 80:
                        status_label = "🟢 STRONG"
                    elif confidence >= 60:
                        status_label = "🟡 WATCH"
                    else:
                        status_label = "🔴 WEAK"
                    results.append({"Ticker": ticker, "Price": f"${price:.2f}",
                        "Confidence": f"{confidence}%", "Status": status_label,
                        "Action": "📈 CALL" if direction == "bullish" else "📉 PUT",
                        "Setup": ptype, "R:R": f"{best.rr_ratio}x"})
            except:
                pass
        prog.empty()
        status.empty()
        if results:
            results.sort(key=lambda x: int(x["Confidence"].replace("%", "")), reverse=True)
            st.success(f"Found {len(results)} setups — sorted by confidence")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No confirmed setups right now. Try again later or switch timeframe.")

st.markdown("<div style='text-align:center;padding:20px;color:#8899aa;font-size:0.75rem;border-top:1px solid #1e2d40;margin-top:20px'>OPTIONS SCREENER v3.0 · NOT FINANCIAL ADVICE</div>", unsafe_allow_html=True)
