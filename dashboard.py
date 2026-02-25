# Options Screener v5.0
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import pytz

from pattern_detection import detect_double_bottom, detect_double_top, detect_break_and_retest
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
.metric-card { background: #111827; border: 1px solid #1e2d40; border-radius: 8px; padding: 14px; margin: 4px 0; }
.rank-best   { background: #061a10; border: 2px solid #00d4aa; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-better { background: #0a1a0a; border: 2px solid #40c070; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-good   { background: #141a0a; border: 2px solid #f0c040; border-radius: 12px; padding: 16px; margin: 6px 0; }
.rank-badge  { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; padding: 3px 10px; border-radius: 20px; display: inline-block; margin-bottom: 8px; }
.badge-best   { background: #00d4aa22; color: #00d4aa; }
.badge-better { background: #40c07022; color: #40c070; }
.badge-good   { background: #f0c04022; color: #f0c040; }
.conf-num-best   { font-size: 2.2rem; font-weight: 700; color: #00d4aa; }
.conf-num-better { font-size: 2.2rem; font-weight: 700; color: #40c070; }
.conf-num-good   { font-size: 2.2rem; font-weight: 700; color: #f0c040; }
.factor-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.82rem; }
.dot-green { width: 8px; height: 8px; background: #00d4aa; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-red   { width: 8px; height: 8px; background: #ff4d6d; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.trade-box { background: #111827; border-radius: 8px; padding: 14px; margin-top: 10px; border-left: 3px solid #00d4aa; }
.trade-box.bear { border-left-color: #ff4d6d; }
.conflict-warn { background: #1a150a; border: 1px solid #f0c040; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #f0c040; }
.market-open   { background: #061a10; border: 1px solid #00d4aa; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #00d4aa; font-size: 0.85rem; }
.market-closed { background: #1a1010; border: 1px solid #ff4d6d; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #ff4d6d; font-size: 0.85rem; }
.market-pre    { background: #1a150a; border: 1px solid #f0c040; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; color: #f0c040; font-size: 0.85rem; }
.divergence-bull { background: #061a10; border: 1px solid #00d4aa; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #00d4aa; }
.divergence-bear { background: #1a0610; border: 1px solid #ff4d6d; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.83rem; color: #ff4d6d; }
</style>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
WATCHLIST = ["PLTR","NBIS","VRT","CRDO","GOOGL","AAOI","ASTS","ZETA","SPY","QQQ","NVDA","TSLA","AAPL"]
TIMEFRAMES = {
    "5 Min":  ("minute", 5,  2),
    "15 Min": ("minute", 15, 5),
    "1 Hour": ("hour",   1,  14),
    "4 Hour": ("hour",   4,  30),
    "Daily":  ("day",    1,  90),
}

# -- Market hours --------------------------------------
def get_market_status():
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    wd = now.weekday()  # 0=Mon, 6=Sun
    t = now.time()
    from datetime import time as dtime
    if wd >= 5:
        return "closed", "Market Closed - Weekend"
    pre_start  = dtime(4, 0)
    open_start = dtime(9, 30)
    close_time = dtime(16, 0)
    after_end  = dtime(20, 0)
    if t < pre_start:
        return "closed", "Market Closed - Opens at 4:00 AM ET Pre-Market"
    elif t < open_start:
        return "pre", f"⏰ Pre-Market Hours - Regular session opens at 9:30 AM ET"
    elif t < close_time:
        return "open", f"🟢 Market Open - Regular Session Until 4:00 PM ET"
    elif t < after_end:
        return "after", f"🌙 After-Hours Trading - Until 8:00 PM ET"
    else:
        return "closed", "Market Closed - Pre-market opens 4:00 AM ET"

# -- Data ---------------------------------------------
@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, multiplier, timespan, days_back):
    try:
        import yfinance as yf
        intervals = {"minute":"5m","hour":"1h","day":"1d"}
        interval = intervals.get(timespan,"1h")
        period = f"{min(days_back,59)}d" if timespan=="minute" else f"{days_back}d"
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return _demo_data(ticker)
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
        df = df.rename(columns={"datetime":"timestamp","date":"timestamp"})
        return df[["timestamp","open","high","low","close","volume"]].dropna().reset_index(drop=True)
    except:
        return _demo_data(ticker)

def _demo_data(ticker, bars=200):
    np.random.seed(hash(ticker)%999)
    prices = {"PLTR":118,"NBIS":98,"VRT":92,"CRDO":68,"GOOGL":175,
              "AAOI":22,"ASTS":28,"ZETA":19,"SPY":570,"QQQ":490,
              "NVDA":138,"TSLA":320,"AAPL":228}.get(ticker,100)
    dates = pd.date_range(end=datetime.now(), periods=bars, freq="1h")
    close = [prices]
    for _ in range(bars-1):
        close.append(close[-1]*(1+np.random.normal(0,0.012)))
    close = np.array(close)
    hi = close*(1+np.abs(np.random.normal(0,0.008,bars)))
    lo = close*(1-np.abs(np.random.normal(0,0.008,bars)))
    op = lo+np.random.uniform(0,1,bars)*(hi-lo)
    vol = np.random.randint(500000,3000000,bars)
    return pd.DataFrame({"timestamp":dates,"open":op,"high":hi,"low":lo,"close":close,"volume":vol})

def fetch_current_price(ticker):
    try:
        import yfinance as yf
        return round(float(yf.Ticker(ticker).fast_info["last_price"]),2)
    except:
        return None

@st.cache_data(ttl=3600)
def check_earnings(ticker):
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return None
        days_away = (pd.Timestamp(cal.iloc[0,0]).date()-date.today()).days
        return days_away if 0<=days_away<=7 else None
    except:
        return None

# -- RSI divergence ------------------------------------
def detect_rsi_divergence(df):
    """
    Bullish divergence: price makes lower low but RSI makes higher low - reversal up coming
    Bearish divergence: price makes higher high but RSI makes lower high - reversal down coming
    """
    if len(df) < 30:
        return None
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100-(100/(1+(gain/loss)))

    # Look at last 20 bars
    recent_close = close.iloc[-20:].values
    recent_rsi   = rsi.iloc[-20:].values

    # Find two recent lows in price
    price_lows = []
    rsi_lows   = []
    for i in range(2, len(recent_close)-2):
        if recent_close[i] < recent_close[i-1] and recent_close[i] < recent_close[i+1]:
            price_lows.append((i, recent_close[i]))
            rsi_lows.append((i, recent_rsi[i]))

    # Find two recent highs in price
    price_highs = []
    rsi_highs   = []
    for i in range(2, len(recent_close)-2):
        if recent_close[i] > recent_close[i-1] and recent_close[i] > recent_close[i+1]:
            price_highs.append((i, recent_close[i]))
            rsi_highs.append((i, recent_rsi[i]))

    # Bullish divergence - price lower low, RSI higher low
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p1, p2 = price_lows[-2][1], price_lows[-1][1]
        r1, r2 = rsi_lows[-2][1],   rsi_lows[-1][1]
        if p2 < p1 and r2 > r1:
            return {
                "type": "bullish",
                "label": "📈 Bullish RSI Divergence Detected",
                "detail": f"Price made a lower low (${p2:.2f} < ${p1:.2f}) but RSI made a higher low ({r2:.0f} > {r1:.0f}). This often signals a reversal UP is coming before the pattern fully forms.",
            }

    # Bearish divergence - price higher high, RSI lower high
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = price_highs[-2][1], price_highs[-1][1]
        r1, r2 = rsi_highs[-2][1],   rsi_highs[-1][1]
        if p2 > p1 and r2 < r1:
            return {
                "type": "bearish",
                "label": "📉 Bearish RSI Divergence Detected",
                "detail": f"Price made a higher high (${p2:.2f} > ${p1:.2f}) but RSI made a lower high ({r2:.0f} < {r1:.0f}). This often signals a reversal DOWN is coming.",
            }

    return None

# -- Signal history ------------------------------------
def load_signal_log():
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    return st.session_state.signal_log

def log_signal(ticker, direction, strike, target, stop, confidence, pattern):
    log = load_signal_log()
    log.append({
        "Date":       datetime.now().strftime("%m/%d %H:%M"),
        "Ticker":     ticker,
        "Action":     "📈 CALL" if direction=="bullish" else "📉 PUT",
        "Pattern":    pattern,
        "Strike":     f"${strike:.2f}",
        "Target":     f"${target:.2f}",
        "Stop":       f"${stop:.2f}",
        "Confidence": f"{confidence}%",
        "Result":     "⏳ Open",
    })
    st.session_state.signal_log = log[-100:]

def get_ticker_signal_stats():
    """Count strong signals per ticker from the log."""
    log = load_signal_log()
    if not log:
        return {}
    stats = {}
    for entry in log:
        t = entry["Ticker"]
        if t not in stats:
            stats[t] = {"total":0,"calls":0,"puts":0,"wins":0}
        stats[t]["total"] += 1
        if "CALL" in entry["Action"]:
            stats[t]["calls"] += 1
        else:
            stats[t]["puts"] += 1
        if "WIN" in entry.get("Result",""):
            stats[t]["wins"] += 1
    return stats

# -- Trend analysis ------------------------------------
def get_trend(df):
    close=df["close"]; high=df["high"]; low=df["low"]
    price=float(close.iloc[-1])
    ema20=float(close.ewm(span=20).mean().iloc[-1])
    tp=(high+low+close)/3
    vwap=float((tp*df["volume"]).cumsum().iloc[-1]/df["volume"].cumsum().iloc[-1])
    rsi=calc_rsi(close)
    recent=df.tail(10)
    up_vol  =float(recent[recent["close"]>=recent["open"]]["volume"].mean() or 0)
    down_vol=float(recent[recent["close"]< recent["open"]]["volume"].mean() or 0)
    hl=[float(high.iloc[i]) for i in range(-10,0)]
    ll=[float(low.iloc[i])  for i in range(-10,0)]
    lower_highs=len(hl)>=9 and hl[-1]<hl[-5]<hl[-9]
    higher_lows=len(ll)>=9 and ll[-1]>ll[-5]>ll[-9]
    bear={"below_ema":{"pass":price<ema20,"label":f"Price below EMA 20 (${ema20:.2f})"},
          "below_vwap":{"pass":price<vwap,"label":f"Price below VWAP (${vwap:.2f})"},
          "rsi_high":{"pass":rsi>55,"label":f"RSI elevated ({rsi:.0f})"},
          "down_vol":{"pass":down_vol>up_vol,"label":"Heavier volume on down bars"},
          "lower_highs":{"pass":lower_highs,"label":"Lower highs forming"}}
    bull={"above_ema":{"pass":price>ema20,"label":f"Price above EMA 20 (${ema20:.2f})"},
          "above_vwap":{"pass":price>vwap,"label":f"Price above VWAP (${vwap:.2f})"},
          "rsi_low":{"pass":rsi<45,"label":f"RSI low ({rsi:.0f})"},
          "up_vol":{"pass":up_vol>down_vol,"label":"Heavier volume on up bars"},
          "higher_lows":{"pass":higher_lows,"label":"Higher lows forming"}}
    bear_score=sum(1 for f in bear.values() if f["pass"])
    bull_score=sum(1 for f in bull.values() if f["pass"])
    if bear_score>=bull_score:
        return "bearish",bear_score,bear,ema20,vwap,rsi
    return "bullish",bull_score,bull,ema20,vwap,rsi

def score_setup(df, setup):
    close=df["close"]; high=df["high"]; low=df["low"]
    price=float(close.iloc[-1])
    is_bull=setup.direction=="bullish"
    ema20=float(close.ewm(span=20).mean().iloc[-1])
    tp=(high+low+close)/3
    vwap=float((tp*df["volume"]).cumsum().iloc[-1]/df["volume"].cumsum().iloc[-1])
    rsi=calc_rsi(close)
    avg_vol=float(df["volume"].iloc[-20:].mean())
    cur_vol=float(df["volume"].iloc[-1])
    factors={
        "pattern":{"pass":True,"label":"Pattern confirmed"},
        "rsi":    {"pass":35<rsi<65,"label":f"RSI in zone ({rsi:.0f})"},
        "vwap":   {"pass":(price>vwap if is_bull else price<vwap),"label":f"Price {'above' if is_bull else 'below'} VWAP (${vwap:.2f})"},
        "volume": {"pass":cur_vol>avg_vol*1.2,"label":f"Volume spike ({cur_vol/1e6:.1f}M vs avg {avg_vol/1e6:.1f}M)"},
        "ema":    {"pass":(price>ema20 if is_bull else price<ema20),"label":f"Price {'above' if is_bull else 'below'} EMA 20 (${ema20:.2f})"},
    }
    score=sum(1 for f in factors.values() if f["pass"])
    return factors,score,int(score/5*100),rsi,vwap,ema20

def calc_rsi(close, period=14):
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def estimate_delta(price, strike, dte, iv=0.45, is_call=True):
    import math
    T = max(dte / 365, 0.001)
    try:
        d1 = (math.log(price / strike) + (0.05 + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        nd1 = 1 / (1 + math.exp(-1.7 * d1))
        return nd1 if is_call else nd1 - 1
    except:
        return 0.5

def calc_trade(entry, stop, target, direction, days_to_exp, account, risk_pct, current_price, iv=0.45):
    is_call = direction == "bullish"
    strike = round(entry / 0.5) * 0.5
    delta = estimate_delta(current_price, strike, days_to_exp, iv, is_call)
    abs_delta = abs(delta)
    premium = round(current_price * iv * (days_to_exp / 365) ** 0.5 * 0.4, 2)
    premium = max(premium, 0.10)
    breakeven = (strike + premium) if is_call else (strike - premium)
    max_loss_per = premium * 100
    contracts = max(1, int((account * risk_pct) / max_loss_per)) if max_loss_per > 0 else 1
    if is_call:
        profit_per = max(0, (target - strike - premium) * 100)
    else:
        profit_per = max(0, (strike - target - premium) * 100)
    total_profit = profit_per * contracts
    rr = round(abs(target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0
    return {
        "type": "CALL" if is_call else "PUT",
        "strike": strike, "premium": premium,
        "breakeven": round(breakeven, 2),
        "max_loss": round(max_loss_per * contracts, 2),
        "contracts": contracts,
        "profit_at_target": round(total_profit, 2),
        "target": round(target, 2), "stop": round(stop, 2), "entry": round(entry, 2),
        "rr": rr, "delta": round(abs_delta, 2),
        "delta_ok": 0.35 <= abs_delta <= 0.85,
        "expiration": (date.today() + timedelta(days=days_to_exp)).strftime("%b %d, %Y"),
    }

def build_candidates(df, ticker, toggles, account, risk_pct, dte):
    trend_dir,trend_score,trend_factors,t_ema,t_vwap,t_rsi = get_trend(df)
    price=float(df["close"].iloc[-1])
    atr=float((df["high"]-df["low"]).tail(14).mean())
    candidates=[]

    raw=[]
    if toggles["db"]:
        raw+=[s for s in detect_double_bottom(df,ticker,rr_min=2.0) if s.confirmed]
    if toggles["dt"]:
        raw+=[s for s in detect_double_top(df,ticker,rr_min=2.0) if s.confirmed]
    if toggles["br"]:
        raw+=[s for s in detect_break_and_retest(df,ticker,rr_min=2.0) if s.confirmed]

    for setup in raw:
        # Staleness filter - skip if entry is more than 5% away from current price
        if abs(setup.entry_price - price) / price > 0.05:
            continue
        factors,score,confidence,rsi,vwap,ema20=score_setup(df,setup)
        conflict=setup.direction!=trend_dir and trend_score>=3
        if conflict:
            t_entry=round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
            t_stop =round(price+atr*1.5,2) if trend_dir=="bearish" else round(price-atr*1.5,2)
            t_target=round(price-atr*3.0,2) if trend_dir=="bearish" else round(price+atr*3.0,2)
            candidates.append({
                "source":"trend_override","direction":trend_dir,
                "confidence":int(trend_score/5*100),"score":trend_score,
                "factors":trend_factors,"conflict":True,
                "conflict_pattern":setup.pattern,
                "entry":t_entry,"stop":t_stop,"target":t_target,
                "pattern_label":"Trend Override",
                "rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
            })
        else:
            bonus=10 if setup.direction==trend_dir else 0
            candidates.append({
                "source":"pattern","direction":setup.direction,
                "confidence":min(100,confidence+bonus),"score":score,
                "factors":factors,"conflict":False,
                "entry":setup.entry_price,"stop":setup.stop_loss,"target":setup.target,
                "pattern_label":setup.pattern.replace("Double","Double ").replace("BreakRetest","Break & Retest"),
                "rsi":rsi,"vwap":vwap,"ema20":ema20,"rr":setup.rr_ratio,
            })

    # Trend signal even without patterns
    if trend_score>=3:
        t_entry=round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
        t_stop =round(price+atr*1.5,2) if trend_dir=="bearish" else round(price-atr*1.5,2)
        t_target=round(price-atr*3.0,2) if trend_dir=="bearish" else round(price+atr*3.0,2)
        candidates.append({
            "source":"trend","direction":trend_dir,
            "confidence":int(trend_score/5*100),"score":trend_score,
            "factors":trend_factors,"conflict":False,
            "entry":t_entry,"stop":t_stop,"target":t_target,
            "pattern_label":f"{'Bearish' if trend_dir=='bearish' else 'Bullish'} Trend",
            "rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
        })

    # Deduplicate - keep highest confidence per unique pattern+direction
    seen={}
    for c in sorted(candidates,key=lambda x:x["confidence"],reverse=True):
        key=f"{c['direction']}_{c['pattern_label']}"
        if key not in seen:
            seen[key]=c

    return sorted(seen.values(),key=lambda x:x["confidence"],reverse=True)[:3]

# -- Share text ----------------------------------------
def build_share_text(ticker, sig, opt, market_status):
    direction = "📈 CALL" if sig["direction"]=="bullish" else "📉 PUT"
    return (f"OPTIONS SCREENER SIGNAL\n"
            f"{'='*30}\n"
            f"{ticker} - {direction}\n"
            f"Pattern: {sig['pattern_label']}\n"
            f"Confidence: {sig['confidence']}%\n"
            f"{'='*30}\n"
            f"Strike:   ${opt['strike']:.2f}\n"
            f"Pay max:  ${opt['premium']:.2f}/share\n"
            f"Entry:    ${opt['entry']:.2f}\n"
            f"Target:   ${opt['target']:.2f}\n"
            f"Stop out: ${opt['stop']:.2f}\n"
            f"R:R:      {opt['rr']}x\n"
            f"Contracts:{opt['contracts']}\n"
            f"Max loss: ${opt['max_loss']:.0f}\n"
            f"Expires:  {opt['expiration']}\n"
            f"Profit at target: ${opt['profit_at_target']:,.0f}\n"
            f"{'='*30}\n"
            f"Market: {market_status}\n"
            f"Time: {datetime.now().strftime('%m/%d/%Y %H:%M')}\n"
            f"NOT FINANCIAL ADVICE")

# -- Sidebar -------------------------------------------
with st.sidebar:
    st.markdown("## 📡 OPTIONS SCREENER")
    st.markdown("---")
    selected_ticker=st.selectbox("TICKER",WATCHLIST)
    custom=st.text_input("Or type any ticker","").upper().strip()
    if custom:
        selected_ticker=custom
    selected_tf=st.selectbox("TIMEFRAME",list(TIMEFRAMES.keys()),index=2)
    st.markdown("---")
    st.markdown("**PATTERNS TO SCAN**")
    tog_db   =st.toggle("📈 Double Bottom (calls)",value=True)
    tog_br_up=st.toggle("📈 Break & Retest Up (calls)",value=True)
    tog_dt   =st.toggle("📉 Double Top (puts)",value=True)
    tog_br_dn=st.toggle("📉 Break & Retest Down (puts)",value=True)
    toggles={"db":tog_db,"dt":tog_dt,"br":tog_br_up or tog_br_dn}
    st.markdown("---")
    st.markdown("**ACCOUNT SETTINGS**")
    account_size=st.number_input("Account Size ($)",value=10000,step=1000)
    risk_pct=st.slider("Risk per Trade (%)",0.5,5.0,1.0,0.5)/100
    dte=st.selectbox("Days to Expiration",[14,21,30,45,60],index=2)
    st.markdown("---")
    st.markdown("**AUTO REFRESH**")
    refresh_on=st.toggle("Live refresh",value=False)
    refresh_interval=st.selectbox("Interval",["1 min","5 min","15 min"],index=1) if refresh_on else None
    st.markdown("---")
    if POLYGON_API_KEY:
        st.success("🟢 LIVE DATA")
    else:
        st.warning("🟡 DEMO MODE")

# -- Auto refresh --------------------------------------
if refresh_on and AUTOREFRESH_AVAILABLE and refresh_interval:
    ms={"1 min":60000,"5 min":300000,"15 min":900000}.get(refresh_interval,300000)
    st_autorefresh(interval=ms, key="autorefresh")

# -- Load data -----------------------------------------
tf_mult,tf_span,tf_days=TIMEFRAMES[selected_tf]
df=fetch_ohlcv(selected_ticker,tf_mult,tf_span,tf_days)
current_price=fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
prev_close=float(df["close"].iloc[-2]) if len(df)>1 else current_price
pct_change=((current_price-prev_close)/prev_close)*100

# -- Market status banner ------------------------------
mstatus, mtext = get_market_status()
css_class = {"open":"market-open","pre":"market-pre","after":"market-pre","closed":"market-closed"}.get(mstatus,"market-closed")
st.markdown(f"<div class='{css_class}'>{mtext}</div>", unsafe_allow_html=True)

# -- Earnings warning ----------------------------------
ed=check_earnings(selected_ticker)
if ed is not None:
    if ed<=1:
        st.error(f"🚨 {selected_ticker} reports earnings {'today' if ed==0 else 'tomorrow'} - Avoid new options positions.")
    else:
        st.warning(f"📅 {selected_ticker} earns in {ed} days - Options premiums may be inflated.")

# -- Header --------------------------------------------
c1,c2,c3=st.columns([2,1,1])
with c1:
    color="#00d4aa" if pct_change>=0 else "#ff4d6d"
    arrow="UP" if pct_change>=0 else "DN"
    prepost = "" if mstatus=="open" else " <span style='color:#f0c040;font-size:0.75rem'>(delayed)</span>"
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.8rem'>{selected_ticker} . {selected_tf}</div><div class='big-price'>${current_price:,.2f}{prepost}</div><div style='color:{color}'>{arrow} {pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
with c2:
    ema20v=float(df["close"].ewm(span=20).mean().iloc[-1])
    above=current_price>ema20v
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>TREND</div><div style='font-weight:700;color:{'#00d4aa' if above else '#ff4d6d'}'>{'BULLISH UP' if above else 'BEARISH DN'}</div></div>", unsafe_allow_html=True)
with c3:
    vol=float(df["volume"].iloc[-1])
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>VOLUME</div><div style='font-weight:700'>{vol/1e6:.1f}M</div></div>", unsafe_allow_html=True)

# -- RSI Divergence alert ------------------------------
div=detect_rsi_divergence(df)
if div:
    css=f"divergence-{'bull' if div['type']=='bullish' else 'bear'}"
    st.markdown(f"<div class='{css}'><b>{div['label']}</b><br>{div['detail']}</div>", unsafe_allow_html=True)

# -- Tabs ----------------------------------------------
tab1,tab2,tab3,tab4,tab5=st.tabs(["🚦 SIGNALS","📈 CHART","📊 BACKTEST","🔍 SCAN","📋 SIGNAL LOG"])

# -- TAB 1: Signals ------------------------------------
with tab1:
    candidates=build_candidates(df,selected_ticker,toggles,account_size,risk_pct,dte)

    if not candidates:
        st.markdown("""<div style='background:#111827;border:2px solid #1e2d40;border-radius:12px;padding:24px;text-align:center;color:#8899aa'>
            <div style='font-size:2rem'>&mdash;</div>
            <div style='font-size:1rem;font-weight:700;margin:8px 0'>NO SIGNALS FOUND</div>
            <div style='font-size:0.85rem'>Try Daily or 4 Hour timeframe, enable more patterns, or check a different ticker.</div>
        </div>""", unsafe_allow_html=True)
    else:
        rank_labels  =["🥇 BEST","🥈 BETTER","🥉 GOOD"]
        rank_classes =["rank-best","rank-better","rank-good"]
        badge_classes=["badge-best","badge-better","badge-good"]
        conf_classes =["conf-num-best","conf-num-better","conf-num-good"]

        for i,sig in enumerate(candidates):
            rl=rank_labels[i]  if i<3 else f"#{i+1}"
            rc=rank_classes[i] if i<3 else "rank-good"
            bc=badge_classes[i]if i<3 else "badge-good"
            cc=conf_classes[i] if i<3 else "conf-num-good"

            is_bull=sig["direction"]=="bullish"
            dir_color="#00d4aa" if is_bull else "#ff4d6d"
            dir_label="📈 BUY CALL" if is_bull else "📉 BUY PUT"

            conflict_html=""
            if sig.get("conflict"):
                pname=sig.get("conflict_pattern","pattern")
                conflict_html=f"<div class='conflict-warn'>⚠️ {pname} pattern found but trend is {'bearish' if not is_bull else 'bullish'} - trend wins. Showing {'PUT' if not is_bull else 'CALL'} instead.</div>"

            dots_html=""
            for f in sig["factors"].values():
                dot="dot-green" if f["pass"] else "dot-red"
                dots_html+=f"<div class='factor-row'><span class='{dot}'></span><span style='color:{'#e0e6f0' if f['pass'] else '#8899aa'}'>{f['label']}</span></div>"

            st.markdown(f"""
            {conflict_html}
            <div class='{rc}'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <div>
                        <span class='rank-badge {bc}'>{rl}</span>
                        <div style='font-size:1.1rem;font-weight:700;color:{dir_color}'>{dir_label} - {selected_ticker}</div>
                        <div style='color:#8899aa;font-size:0.82rem;margin-top:2px'>{sig['pattern_label']}</div>
                    </div>
                    <div class='{cc}'>{sig['confidence']}%</div>
                </div>
                <div style='margin-top:10px'>{dots_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if sig["confidence"]>=60:
                opt=calc_trade(sig["entry"],sig["stop"],sig["target"],sig["direction"],dte,account_size,risk_pct,current_price)
                if not opt["delta_ok"]:
                    st.markdown(f"<div style='background:#1a1010;border:1px solid #ff4d6d;border-radius:8px;padding:10px;margin-top:6px;color:#ff4d6d;font-size:0.83rem'>Delta {opt['delta']:.2f} is outside 0.35-0.85 range - not worth trading.</div>", unsafe_allow_html=True)
                elif opt["profit_at_target"] < 50:
                    st.markdown(f"<div style='background:#1a150a;border:1px solid #f0c040;border-radius:8px;padding:10px;margin-top:6px;color:#f0c040;font-size:0.83rem'>Profit at target too low (${opt['profit_at_target']:.0f}) - target too close to entry. Wait for a wider setup.</div>", unsafe_allow_html=True)
                else:
                    delta_color = "#00d4aa"
                    st.markdown(f"""
                    <div class='trade-box {"" if is_bull else "bear"}'>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.88rem'>
                            <div><div style='color:#8899aa;font-size:0.72rem'>STRIKE</div><div style='font-size:1.2rem;font-weight:700;color:{dir_color}'>${opt['strike']:.2f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>PAY MAX</div><div style='font-weight:700'>${opt['premium']:.2f}/sh</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>ENTRY</div><div style='font-weight:700'>${opt['entry']:.2f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>DELTA</div><div style='font-weight:700;color:{delta_color}'>{opt['delta']:.2f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>EXIT TARGET</div><div style='font-weight:700;color:#00d4aa'>${opt['target']:.2f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>STOP OUT</div><div style='font-weight:700;color:#ff4d6d'>${opt['stop']:.2f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>R:R RATIO</div><div style='font-weight:700;color:#00d4aa'>{opt['rr']}x</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>MAX LOSS</div><div style='font-weight:700;color:#ff4d6d'>${opt['max_loss']:.0f}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>CONTRACTS</div><div style='font-size:1.2rem;font-weight:700;color:{dir_color}'>{opt['contracts']}</div></div>
                            <div><div style='color:#8899aa;font-size:0.72rem'>PROFIT AT TARGET</div><div style='font-size:1.2rem;font-weight:700;color:#00d4aa'>${opt['profit_at_target']:,.0f}</div></div>
                        </div>
                        <div style='margin-top:10px;padding-top:8px;border-top:1px solid #1e2d40'>
                            <div style='color:#8899aa;font-size:0.72rem'>EXPIRES</div>
                            <div style='font-weight:700'>{opt['expiration']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button(f"📋 Log Signal #{i+1}", key=f"log_{i}"):
                            log_signal(selected_ticker, sig["direction"], opt["strike"], opt["target"], opt["stop"], sig["confidence"], sig["pattern_label"])
                            st.success("Logged!")
                    with bcol2:
                        share_text = build_share_text(selected_ticker, sig, opt, mtext)
                        st.download_button(f"📤 Share #{i+1}", data=share_text,
                            file_name=f"{selected_ticker}_signal_{datetime.now().strftime('%m%d_%H%M')}.txt",
                            mime="text/plain", key=f"share_{i}")

            if i<len(candidates)-1:
                st.markdown("<hr style='border-color:#1e2d40;margin:12px 0'>", unsafe_allow_html=True)

# -- TAB 2: Chart --------------------------------------
with tab2:
    chart_db=[s for s in detect_double_bottom(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_dt=[s for s in detect_double_top(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_br=[s for s in detect_break_and_retest(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_setups=chart_db+chart_dt+chart_br

    tf_formats={"5 Min":"%H:%M","15 Min":"%H:%M","1 Hour":"%b %d %H:%M","4 Hour":"%b %d","Daily":"%b %d '%y"}
    tick_format=tf_formats.get(selected_tf,"%b %d")
    price_min=float(df["low"].min())*0.995
    price_max=float(df["high"].max())*1.005
    price_range=price_max-price_min
    raw_tick=price_range/8
    magnitude=10**(len(str(int(max(raw_tick,1))))-1)
    tick_interval=max(round(raw_tick/magnitude)*magnitude,0.01)

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df["timestamp"],open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name=selected_ticker,
        increasing_line_color="#00d4aa",decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00d4aa",decreasing_fillcolor="#ff4d6d",line_width=1),row=1,col=1)
    ema_line=df["close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df["timestamp"],y=ema_line,name="EMA 20",
        line=dict(color="#f0c040",width=1.5,dash="dot"),
        hovertemplate="EMA 20: $%{y:.2f}<extra></extra>"),row=1,col=1)
    tp=(df["high"]+df["low"]+df["close"])/3
    vwap_line=(tp*df["volume"]).cumsum()/df["volume"].cumsum()
    fig.add_trace(go.Scatter(x=df["timestamp"],y=vwap_line,name="VWAP",
        line=dict(color="#9966ff",width=1.5,dash="dash"),
        hovertemplate="VWAP: $%{y:.2f}<extra></extra>"),row=1,col=1)
    for s in chart_setups[:3]:
        lc="#00d4aa" if s.direction=="bullish" else "#ff4d6d"
        fig.add_hline(y=s.entry_price,line_dash="solid",line_color=lc,line_width=1.5,opacity=0.8,
            annotation_text=f"  Entry ${s.entry_price:.2f}",annotation_font_color=lc,annotation_font_size=11,row=1,col=1)
        fig.add_hline(y=s.target,line_dash="dash",line_color="#00d4aa",line_width=1,opacity=0.6,
            annotation_text=f"  Target ${s.target:.2f}",annotation_font_color="#00d4aa",annotation_font_size=11,row=1,col=1)
        fig.add_hline(y=s.stop_loss,line_dash="dot",line_color="#ff4d6d",line_width=1,opacity=0.6,
            annotation_text=f"  Stop ${s.stop_loss:.2f}",annotation_font_color="#ff4d6d",annotation_font_size=11,row=1,col=1)
        fig.add_hrect(y0=min(s.entry_price,s.target),y1=max(s.entry_price,s.target),
            fillcolor="rgba(0,212,170,0.05)",line_width=0,row=1,col=1)
    vol_colors=["#00d4aa" if c>=o else "#ff4d6d" for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"],y=df["volume"],marker_color=vol_colors,opacity=0.5,name="Volume",
        hovertemplate="%{x}<br>Vol: %{y:,.0f}<extra></extra>"),row=2,col=1)
    fig.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0d1219",font=dict(color="#e0e6f0",size=12),
        height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=80,t=10,b=10),
        legend=dict(bgcolor="rgba(13,18,25,0.8)",bordercolor="#1e2d40",borderwidth=1,x=0.01,y=0.99),
        hovermode="x unified",
        modebar_remove=["pan","lasso2d","select2d","autoScale2d","hoverCompareCartesian",
                        "hoverClosestCartesian","toggleSpikelines","zoomIn2d","zoomOut2d"])
    fig.update_xaxes(gridcolor="#1e2d40",tickformat=tick_format,nticks=8,
        showspikes=True,spikecolor="#1e2d40",spikedash="solid",spikethickness=1)
    fig.update_yaxes(gridcolor="#1e2d40",tickformat="$.2f",dtick=tick_interval,
        range=[price_min,price_max],showspikes=True,spikecolor="#1e2d40",row=1,col=1)
    fig.update_yaxes(gridcolor="#1e2d40",tickformat=".2s",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    if chart_setups:
        st.markdown("""<div style='display:flex;gap:20px;flex-wrap:wrap;padding:8px 4px;font-size:0.8rem;color:#8899aa'>
            <span><span style='color:#00d4aa'>-----</span> Entry</span>
            <span><span style='color:#00d4aa'>- - -</span> Target</span>
            <span><span style='color:#ff4d6d'>.....</span> Stop Loss</span>
            <span><span style='color:#f0c040'>.....</span> EMA 20</span>
            <span><span style='color:#9966ff'>- - -</span> VWAP</span>
        </div>""", unsafe_allow_html=True)

# -- TAB 3: Backtest -----------------------------------
with tab3:
    st.markdown("<div class='section-title'>BACKTEST</div>", unsafe_allow_html=True)
    if st.button("Run Backtest",type="primary"):
        with st.spinner("Analyzing..."):
            report,equity=run_backtest(df,selected_ticker)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Win Rate",f"{report.win_rate}%")
        c2.metric("Trades",report.total_trades)
        c3.metric("Avg R:R",f"{report.avg_rr}x")
        c4.metric("Expectancy",f"{report.expectancy}R")
        if len(equity)>1:
            fig_eq=go.Figure(go.Scatter(y=equity,mode="lines+markers",
                line=dict(color="#00d4aa",width=2),fill="tozeroy",fillcolor="rgba(0,212,170,0.1)"))
            fig_eq.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0d1219",
                font=dict(color="#e0e6f0"),height=280,title="Equity Curve",margin=dict(l=0,r=0,t=40,b=0))
            fig_eq.update_xaxes(gridcolor="#1e2d40")
            fig_eq.update_yaxes(gridcolor="#1e2d40")
            st.plotly_chart(fig_eq,use_container_width=True)
        if report.trades:
            st.dataframe(pd.DataFrame([{"Pattern":t.pattern,"Result":t.outcome.upper(),
                "Entry":f"${t.entry_price:.2f}","Exit":f"${t.exit_price:.2f}",
                "P&L":f"{t.pnl_pct:+.1f}%"} for t in report.trades]),use_container_width=True)
    else:
        st.info("Click Run Backtest to analyze this ticker.")

# -- TAB 4: Scan ---------------------------------------
with tab4:
    st.markdown("<div class='section-title'>WATCHLIST SCAN</div>", unsafe_allow_html=True)
    if st.button("🔍 SCAN ALL TICKERS",type="primary"):
        results=[]
        prog=st.progress(0); status=st.empty()
        for i,ticker in enumerate(WATCHLIST):
            status.text(f"Scanning {ticker}...")
            prog.progress((i+1)/len(WATCHLIST))
            try:
                tdf=fetch_ohlcv(ticker,tf_mult,tf_span,tf_days)
                price=fetch_current_price(ticker) or float(tdf["close"].iloc[-1])
                cands=build_candidates(tdf,ticker,toggles,account_size,risk_pct,dte)
                if cands:
                    best=cands[0]
                    conf=best["confidence"]
                    results.append({"Ticker":ticker,"Price":f"${price:.2f}",
                        "Confidence":f"{conf}%",
                        "Status":"🟢 STRONG" if conf>=80 else "🟡 WATCH" if conf>=60 else "🔴 WEAK",
                        "Action":"📈 CALL" if best["direction"]=="bullish" else "📉 PUT",
                        "Setup":best["pattern_label"]})
            except:
                pass
        prog.empty(); status.empty()
        if results:
            results.sort(key=lambda x:int(x["Confidence"].replace("%","")),reverse=True)
            st.success(f"Found {len(results)} signals")
            st.dataframe(pd.DataFrame(results),use_container_width=True)
        else:
            st.info("No signals right now. Try again later or switch timeframe.")

# -- TAB 5: Signal Log ---------------------------------
with tab5:
    st.markdown("<div class='section-title'>SIGNAL LOG & TICKER STATS</div>", unsafe_allow_html=True)

    # Ticker signal stats
    stats=get_ticker_signal_stats()
    if stats:
        st.markdown("**Signal Frequency by Ticker** - live tracked since you started logging")
        stat_rows=[{"Ticker":t,"Total Signals":v["total"],"Calls":v["calls"],
                    "Puts":v["puts"],"Wins":v["wins"]} for t,v in
                   sorted(stats.items(),key=lambda x:x[1]["total"],reverse=True)]
        st.dataframe(pd.DataFrame(stat_rows),use_container_width=True)
        st.markdown("<hr style='border-color:#1e2d40;margin:16px 0'>", unsafe_allow_html=True)

    log=load_signal_log()
    if not log:
        st.info("No signals logged yet. Click 'Log Signal' on any 60%+ signal to start tracking.")
    else:
        st.markdown("**Full Signal History**")
        df_log=pd.DataFrame(log[::-1])
        st.dataframe(df_log,use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear Log"):
                st.session_state.signal_log=[]
                st.rerun()
        with col_b:
            csv=df_log.to_csv(index=False)
            st.download_button("📥 Export Log as CSV", data=csv,
                file_name=f"signal_log_{datetime.now().strftime('%m%d%Y')}.csv",
                mime="text/csv")

st.markdown("<div style='text-align:center;padding:20px;color:#8899aa;font-size:0.75rem;border-top:1px solid #1e2d40;margin-top:20px'>OPTIONS SCREENER v5.0 . NOT FINANCIAL ADVICE</div>", unsafe_allow_html=True)
