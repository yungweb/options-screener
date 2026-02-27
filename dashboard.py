# Options Screener v6.0
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import pytz
import math

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
.dot-green  { width: 8px; height: 8px; background: #00d4aa; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-red    { width: 8px; height: 8px; background: #ff4d6d; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-yellow { width: 8px; height: 8px; background: #f0c040; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.trade-box  { background: #111827; border-radius: 8px; padding: 14px; margin-top: 10px; border-left: 3px solid #00d4aa; }
.trade-box.bear { border-left-color: #ff4d6d; }
.exit-rules { background: #0d1525; border: 1px solid #1e2d40; border-radius: 8px; padding: 12px 14px; margin-top: 10px; font-size: 0.83rem; }
.gate-box   { background: #0d1219; border: 1px solid #1e2d40; border-radius: 8px; padding: 12px 14px; margin-top: 8px; }
.ai-placeholder { background: #0d1219; border: 1px dashed #1e2d40; border-radius: 8px; padding: 14px; margin-top: 10px; color: #8899aa; font-size: 0.83rem; text-align: center; }
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

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
POLYGON_API_KEY   = os.environ.get("POLYGON_API_KEY", "")
WATCHLIST = ["PLTR","NBIS","VRT","CRDO","GOOGL","AAOI","ASTS","ZETA","SPY","QQQ","NVDA","TSLA","AAPL"]
TIMEFRAMES = {
    "5 Min":  ("minute", 5,  2),
    "15 Min": ("minute", 15, 5),
    "1 Hour": ("hour",   1,  14),
    "4 Hour": ("hour",   4,  30),
    "Daily":  ("day",    1,  90),
}

def get_market_status():
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    wd  = now.weekday()
    t   = now.time()
    from datetime import time as dtime
    if wd >= 5: return "closed", "Market Closed - Weekend"
    if   t < dtime(4,  0): return "closed", "Market Closed - Opens 4:00 AM ET"
    elif t < dtime(9, 30): return "pre",    "Pre-Market Hours - Regular session opens 9:30 AM ET"
    elif t < dtime(16, 0): return "open",   "Market Open - Regular Session Until 4:00 PM ET"
    elif t < dtime(20, 0): return "after",  "After-Hours Trading - Until 8:00 PM ET"
    else:                  return "closed", "Market Closed - Pre-market opens 4:00 AM ET"

@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, multiplier, timespan, days_back):
    try:
        import yfinance as yf
        intervals = {"minute":"5m","hour":"1h","day":"1d"}
        interval  = intervals.get(timespan,"1h")
        period    = f"{min(days_back,59)}d" if timespan=="minute" else f"{days_back}d"
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return _demo_data(ticker)
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
    hi  = close*(1+np.abs(np.random.normal(0,0.008,bars)))
    lo  = close*(1-np.abs(np.random.normal(0,0.008,bars)))
    op  = lo+np.random.uniform(0,1,bars)*(hi-lo)
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
        if cal is None or cal.empty: return None
        days_away = (pd.Timestamp(cal.iloc[0,0]).date()-date.today()).days
        return days_away if 0<=days_away<=14 else None
    except:
        return None

@st.cache_data(ttl=300)
def fetch_iv_rank(ticker):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty or len(hist) < 30: return None, None
        log_ret    = np.log(hist["Close"]/hist["Close"].shift(1)).dropna()
        rolling_hv = log_ret.rolling(20).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()
        current_hv = float(rolling_hv.iloc[-1])
        hv_low     = float(rolling_hv.min())
        hv_high    = float(rolling_hv.max())
        if hv_high == hv_low: return 50, current_hv
        iv_rank = int((current_hv - hv_low) / (hv_high - hv_low) * 100)
        return iv_rank, current_hv
    except:
        return None, None

def calc_rsi(close, period=14):
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def estimate_delta(price, strike, dte, iv=0.45, is_call=True):
    T = max(dte/365, 0.001)
    try:
        d1  = (math.log(price/strike) + (0.05 + 0.5*iv**2)*T) / (iv*math.sqrt(T))
        nd1 = 1 / (1 + math.exp(-1.7*d1))
        return nd1 if is_call else nd1 - 1
    except:
        return 0.5

def get_expiration_date(dte_target):
    today   = date.today()
    d       = today
    fridays = []
    while len(fridays) < 16:
        d += timedelta(days=1)
        if d.weekday() == 4:
            fridays.append(d)
    valid = [f for f in fridays if (f-today).days >= 5]
    return min(valid, key=lambda f: abs((f-(today+timedelta(days=dte_target))).days))

def estimate_move_timeframe(pattern_label):
    if "Double" in pattern_label:  est_days = 21
    elif "Break" in pattern_label: est_days = 14
    else:                          est_days = 10
    return est_days, int(est_days * 1.5)

def calc_trade(entry, stop, target, direction, days_to_exp, account, risk_pct, current_price, iv=0.45, atr=None):
    is_call    = direction == "bullish"
    exp_date   = get_expiration_date(days_to_exp)
    actual_dte = max((exp_date - date.today()).days, 1)

    raw_strike = current_price * 1.02 if is_call else current_price * 0.98
    strike     = round(raw_strike / 0.5) * 0.5
    premium    = round(current_price * iv * (actual_dte/365)**0.5 * 0.4, 2)
    premium    = max(premium, 0.10)
    breakeven  = (strike + premium) if is_call else (strike - premium)

    # ── Target sanity check ───────────────────────────────────────────────────
    # Use the pattern's measured move as-is. Only apply a sanity cap so we never
    # show a target that requires an unrealistic price move.
    # Cap: target cannot be more than 20% away from current price for stocks,
    # or more than 4x ATR away. Whichever is less restrictive.
    max_move_pct = 0.20  # 20% max move
    if atr and atr > 0:
        # Allow up to 6x ATR as the measured move (generous for double bottoms)
        atr_cap_pct = (atr * 6) / current_price
        max_move_pct = max(max_move_pct, min(atr_cap_pct, 0.35))

    if is_call:
        max_target = round(current_price * (1 + max_move_pct), 2)
        stock_target = min(target, max_target)
    else:
        min_target = round(current_price * (1 - max_move_pct), 2)
        stock_target = max(target, min_target)

    # ATR-based move probability
    move_needed = abs(stock_target - current_price)
    atr_multiples = round(move_needed / atr, 1) if atr and atr > 0 else None
    if atr_multiples is not None:
        if atr_multiples <= 2.0:   target_realistic = "Likely"
        elif atr_multiples <= 4.0: target_realistic = "Possible"
        else:                       target_realistic = "Ambitious"
    else:
        target_realistic = "Unknown"

    # Move pct for display
    move_pct = round((move_needed / current_price) * 100, 1)

    delta     = estimate_delta(current_price, strike, actual_dte, iv, is_call)
    abs_delta = abs(delta)
    max_loss_per     = premium * 100
    contracts        = max(1, int((account * risk_pct) / max_loss_per)) if max_loss_per > 0 else 1
    position_dollars = round(max_loss_per * contracts, 2)
    pct_of_account   = round((position_dollars / account) * 100, 1) if account > 0 else 0

    # Option profit at stock target using intrinsic value estimate
    if is_call: profit_per = max(0, (stock_target - strike - premium) * 100)
    else:       profit_per = max(0, (strike - stock_target - premium) * 100)
    total_profit = round(profit_per * contracts, 2)

    # R:R on the stock move (pattern level)
    rr_stock = round(abs(stock_target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0
    # R:R on the option (dollar gain vs dollar loss)
    rr_option = round(total_profit / position_dollars, 2) if position_dollars > 0 and total_profit > 0 else 0

    return {
        "type": "CALL" if is_call else "PUT",
        "strike": strike, "premium": premium, "breakeven": round(breakeven, 2),
        "max_loss": position_dollars, "contracts": contracts,
        "position_dollars": position_dollars, "pct_of_account": pct_of_account,
        "profit_at_target": total_profit,
        "target": round(stock_target, 2), "stop": round(stop, 2), "entry": round(entry, 2),
        "rr": rr_stock, "rr_option": rr_option,
        "delta": round(abs_delta, 2), "delta_ok": 0.35 <= abs_delta <= 0.85,
        "expiration": exp_date.strftime("%b %d, %Y"), "actual_dte": actual_dte,
        "exit_take_half": round(premium * 2.0, 2),
        "exit_stop_stock": round(stop, 2),
        "move_pct": move_pct,
        "atr_multiples": atr_multiples,
        "target_realistic": target_realistic,
    }

def detect_rsi_divergence(df):
    if len(df) < 30: return None
    close = df["close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100/(1+(gain/loss)))
    rc = close.iloc[-20:].values
    rr = rsi.iloc[-20:].values
    plows, rlows, phighs, rhighs = [], [], [], []
    for i in range(2, len(rc)-2):
        if rc[i] < rc[i-1] and rc[i] < rc[i+1]: plows.append((i,rc[i]));  rlows.append((i,rr[i]))
        if rc[i] > rc[i-1] and rc[i] > rc[i+1]: phighs.append((i,rc[i])); rhighs.append((i,rr[i]))
    if len(plows)>=2 and len(rlows)>=2:
        p1,p2 = plows[-2][1],plows[-1][1]; r1,r2 = rlows[-2][1],rlows[-1][1]
        if p2<p1 and r2>r1: return {"type":"bullish","label":"Bullish RSI Divergence","detail":f"Price lower low (${p2:.2f}) but RSI higher low ({r2:.0f})"}
    if len(phighs)>=2 and len(rhighs)>=2:
        p1,p2 = phighs[-2][1],phighs[-1][1]; r1,r2 = rhighs[-2][1],rhighs[-1][1]
        if p2>p1 and r2<r1: return {"type":"bearish","label":"Bearish RSI Divergence","detail":f"Price higher high (${p2:.2f}) but RSI lower high ({r2:.0f})"}
    return None

def run_seven_point_gate(df, sig, opt, iv_rank, earnings_days, dte_used):
    is_bull = sig["direction"] == "bullish"
    price   = float(df["close"].iloc[-1])
    _, dte_rec = estimate_move_timeframe(sig["pattern_label"])

    iv_ok     = iv_rank is not None and iv_rank < 60
    iv_label  = f"IV Rank {iv_rank}% - {'options cheap, good to buy' if iv_ok else 'elevated, options expensive'}" if iv_rank is not None else "IV Rank unavailable"

    avg_vol   = float(df["volume"].iloc[-20:].mean())
    cur_vol   = float(df["volume"].iloc[-3:].mean())
    vol_ok    = cur_vol > avg_vol * 1.1
    vol_label = f"Volume {'expanding' if vol_ok else 'contracting'} ({cur_vol/1e6:.1f}M vs avg {avg_vol/1e6:.1f}M)"

    div       = detect_rsi_divergence(df)
    div_ok    = div is not None and div["type"] == ("bullish" if is_bull else "bearish")
    div_label = f"RSI divergence {'confirmed' if div_ok else 'not detected'}"

    entry_dist = abs(opt["entry"] - price) / price * 100
    neck_ok    = entry_dist < 3.0
    neck_label = f"Entry {entry_dist:.1f}% from current price ({'in range' if neck_ok else 'too far - stale'})"

    rr_ok    = opt["rr"] >= 2.0
    rr_label = f"Option R:R {opt['rr']}x ({'meets 2:1 min' if rr_ok else 'below 2:1 - skip'})"

    dte_ok    = dte_used >= dte_rec
    dte_label = f"DTE {dte_used} days vs recommended {dte_rec}+ ({'ok' if dte_ok else 'too short, use longer expiry'})"

    earn_ok    = earnings_days is None or earnings_days > 7
    earn_label = f"{'No earnings within 7 days' if earn_ok else f'EARNINGS IN {earnings_days} DAYS - blocked'}"

    gates = {
        "IV Rank":         {"pass": iv_ok,   "label": iv_label,   "critical": False},
        "Volume":          {"pass": vol_ok,  "label": vol_label,  "critical": False},
        "RSI Divergence":  {"pass": div_ok,  "label": div_label,  "critical": False},
        "Entry Proximity": {"pass": neck_ok, "label": neck_label, "critical": True},
        "R:R Ratio":       {"pass": rr_ok,   "label": rr_label,   "critical": True},
        "DTE Adequacy":    {"pass": dte_ok,  "label": dte_label,  "critical": False},
        "Earnings Clear":  {"pass": earn_ok, "label": earn_label, "critical": True},
    }
    passed          = sum(1 for g in gates.values() if g["pass"])
    critical_pass   = all(g["pass"] for g in gates.values() if g["critical"])
    non_crit_pass   = sum(1 for k,g in gates.items() if not g["critical"] and g["pass"])
    elevate         = critical_pass and non_crit_pass >= 2
    return gates, passed, elevate

# ── Entry confirmation candles ────────────────────────────────────────────────
def check_entry_confirmation(df, direction):
    """
    Checks last candles to see if price is moving in signal direction.
    Calls: need 2 consecutive green candles, each close higher than previous.
    Puts:  need 2 consecutive red candles, each close lower than previous.
    Returns status: CONFIRMED / WAITING / AGAINST
    """
    if len(df) < 4:
        return {"confirmed": False, "status": "WAITING", "candles": [], "message": "Not enough data"}

    recent = df.tail(5)
    is_bull = direction == "bullish"

    candle_dirs = []
    for _, row in recent.iterrows():
        if float(row["close"]) > float(row["open"]):   candle_dirs.append("green")
        elif float(row["close"]) < float(row["open"]): candle_dirs.append("red")
        else:                                            candle_dirs.append("doji")

    c1 = recent.iloc[-2]
    c2 = recent.iloc[-1]

    if is_bull:
        both_green    = float(c1["close"]) > float(c1["open"]) and float(c2["close"]) > float(c2["open"])
        higher_closes = float(c2["close"]) > float(c1["close"])
        confirmed     = both_green and higher_closes
        last_green    = candle_dirs[-1] == "green"
        if confirmed:
            status  = "CONFIRMED"
            message = f"2 bullish candles confirmed - buyers in control. Entry window open near ${float(c2['close']):.2f}"
        elif last_green:
            status  = "WAITING"
            message = "1 of 2 bullish candles printed. Need 1 more green candle closing higher."
        else:
            status  = "AGAINST"
            message = "Price still dropping. Signal valid but entry is early - wait for 2 consecutive green candles."
    else:
        both_red     = float(c1["close"]) < float(c1["open"]) and float(c2["close"]) < float(c2["open"])
        lower_closes = float(c2["close"]) < float(c1["close"])
        confirmed    = both_red and lower_closes
        last_red     = candle_dirs[-1] == "red"
        if confirmed:
            status  = "CONFIRMED"
            message = f"2 bearish candles confirmed - sellers in control. Entry window open near ${float(c2['close']):.2f}"
        elif last_red:
            status  = "WAITING"
            message = "1 of 2 bearish candles printed. Need 1 more red candle closing lower."
        else:
            status  = "AGAINST"
            message = "Price still climbing. Signal valid but entry is early - wait for 2 consecutive red candles."

    return {"confirmed": confirmed, "status": status, "candles": candle_dirs, "message": message}

# ── Watch queue ───────────────────────────────────────────────────────────────
WATCH_TIMEOUT_MINS = 30

def init_watch_queue():
    if "watch_queue" not in st.session_state:
        st.session_state.watch_queue = {}

def add_to_watch_queue(ticker, direction, sig, opt):
    init_watch_queue()
    key = f"{ticker}_{direction}"
    if key not in st.session_state.watch_queue:
        st.session_state.watch_queue[key] = {
            "ticker":    ticker,
            "direction": direction,
            "action":    "CALL" if direction == "bullish" else "PUT",
            "strike":    opt["strike"],
            "entry":     opt["entry"],
            "target":    opt["target"],
            "stop":      opt["stop"],
            "pattern":   sig["pattern_label"],
            "confidence":sig["confidence"],
            "added_at":  datetime.now(),
            "last_checked": None,
            "status":    "WAITING",
            "message":   "Watching for 2 confirmation candles...",
            "alerted":   False,
        }

def remove_from_watch_queue(key):
    init_watch_queue()
    if key in st.session_state.watch_queue:
        del st.session_state.watch_queue[key]

def run_background_watch_checks(tf_mult, tf_span, tf_days):
    """
    Runs on EVERY app refresh regardless of selected ticker.
    Fetches fresh candle data for every watched ticker and rechecks confirmation.
    Returns True if any ticker just flipped to CONFIRMED for the first time.
    """
    init_watch_queue()
    queue    = st.session_state.watch_queue
    any_new_confirm = False
    to_remove = []

    for key, item in queue.items():
        # Timeout after 30 minutes
        elapsed = (datetime.now() - item["added_at"]).total_seconds() / 60
        if elapsed > WATCH_TIMEOUT_MINS:
            to_remove.append(key)
            continue

        try:
            fresh_df = fetch_ohlcv(item["ticker"], tf_mult, tf_span, tf_days)
            conf     = check_entry_confirmation(fresh_df, item["direction"])
            was_confirmed_before = item["status"] == "CONFIRMED"

            item["status"]       = conf["status"]
            item["message"]      = conf["message"]
            item["candles"]      = conf.get("candles", [])
            item["last_checked"] = datetime.now()

            # First time flipping to confirmed - trigger alert
            if conf["confirmed"] and not was_confirmed_before and not item["alerted"]:
                item["alerted"]    = True
                any_new_confirm    = True
        except:
            item["message"] = "Data fetch failed - retrying..."

    for key in to_remove:
        del queue[key]

    st.session_state.watch_queue = queue
    return any_new_confirm

def get_trend(df):
    close=df["close"]; high=df["high"]; low=df["low"]
    price=float(close.iloc[-1])
    ema20=float(close.ewm(span=20).mean().iloc[-1])
    tp   =(high+low+close)/3
    vwap =float((tp*df["volume"]).cumsum().iloc[-1]/df["volume"].cumsum().iloc[-1])
    rsi  =calc_rsi(close)
    recent   = df.tail(10)
    up_vol   = float(recent[recent["close"]>=recent["open"]]["volume"].mean() or 0)
    down_vol = float(recent[recent["close"]< recent["open"]]["volume"].mean() or 0)
    hl = [float(high.iloc[i]) for i in range(-10,0)]
    ll = [float(low.iloc[i])  for i in range(-10,0)]
    lower_highs = len(hl)>=9 and hl[-1]<hl[-5]<hl[-9]
    higher_lows = len(ll)>=9 and ll[-1]>ll[-5]>ll[-9]
    bear={"below_ema":{"pass":price<ema20,"label":f"Price below EMA 20 (${ema20:.2f})"},
          "below_vwap":{"pass":price<vwap,"label":f"Price below VWAP (${vwap:.2f})"},
          "rsi_high":  {"pass":rsi>55,    "label":f"RSI elevated ({rsi:.0f})"},
          "down_vol":  {"pass":down_vol>up_vol,"label":"Heavier volume on down bars"},
          "lower_highs":{"pass":lower_highs,"label":"Lower highs forming"}}
    bull={"above_ema": {"pass":price>ema20,"label":f"Price above EMA 20 (${ema20:.2f})"},
          "above_vwap":{"pass":price>vwap, "label":f"Price above VWAP (${vwap:.2f})"},
          "rsi_low":   {"pass":rsi<45,     "label":f"RSI low ({rsi:.0f})"},
          "up_vol":    {"pass":up_vol>down_vol,"label":"Heavier volume on up bars"},
          "higher_lows":{"pass":higher_lows,"label":"Higher lows forming"}}
    bear_score = sum(1 for f in bear.values() if f["pass"])
    bull_score = sum(1 for f in bull.values() if f["pass"])
    if bear_score >= bull_score: return "bearish",bear_score,bear,ema20,vwap,rsi
    return "bullish",bull_score,bull,ema20,vwap,rsi

def detect_market_regime(df):
    """
    Determines if market is TRENDING or CHOPPY using ATR expansion and directional consistency.
    Trending = ATR expanding + price making consistent directional moves.
    Choppy   = ATR contracting or price reversing frequently.
    Returns: regime ("trending"/"choppy"), strength (0-100)
    """
    if len(df) < 30: return "unknown", 50
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    # ATR trend: compare recent 7-bar ATR to prior 14-bar ATR
    tr = pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    atr_recent = float(tr.iloc[-7:].mean())
    atr_prior  = float(tr.iloc[-21:-7].mean())
    atr_expanding = atr_recent > atr_prior * 1.1
    # Directional consistency: how many of last 10 bars close in same direction
    last10 = df.tail(10)
    bull_bars = int((last10["close"] > last10["open"]).sum())
    bear_bars = 10 - bull_bars
    directional = max(bull_bars, bear_bars)  # 5=choppy, 10=strong trend
    consistency_score = int((directional - 5) / 5 * 100)  # 0-100
    if atr_expanding and directional >= 7:
        regime = "trending"
        strength = min(100, int(consistency_score * 1.2))
    elif not atr_expanding and directional <= 6:
        regime = "choppy"
        strength = max(0, 100 - consistency_score)
    else:
        regime = "trending" if directional >= 7 else "choppy"
        strength = consistency_score
    return regime, strength

@st.cache_data(ttl=300)
def check_liquidity(ticker):
    """
    Checks options liquidity via yfinance.
    Returns: liquid (bool), avg_volume, avg_oi, message
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps: return False, 0, 0, "No options data"
        # Use nearest expiration
        chain = tk.option_chain(exps[0])
        calls = chain.calls
        if calls.empty: return False, 0, 0, "No calls data"
        avg_vol = float(calls["volume"].fillna(0).mean())
        avg_oi  = float(calls["openInterest"].fillna(0).mean())
        liquid  = avg_vol >= 50 and avg_oi >= 100
        msg = f"Avg vol {avg_vol:.0f} | OI {avg_oi:.0f}"
        return liquid, avg_vol, avg_oi, msg
    except:
        return True, 0, 0, "Liquidity check unavailable"

def score_setup(df, setup):
    """
    Weighted confidence scoring.
    Weights (must sum to 100):
      RSI divergence  25  - strongest signal, price/momentum divergence
      Volume          25  - smart money confirmation
      Pattern confirm 20  - structural setup
      EMA alignment   15  - trend filter
      VWAP            15  - intraday anchor
    """
    close  = df["close"]; high = df["high"]; low = df["low"]
    price  = float(close.iloc[-1])
    is_bull = setup.direction == "bullish"
    ema20  = float(close.ewm(span=20).mean().iloc[-1])
    tp     = (high + low + close) / 3
    vwap   = float((tp * df["volume"]).cumsum().iloc[-1] / df["volume"].cumsum().iloc[-1])
    rsi    = calc_rsi(close)
    avg_vol = float(df["volume"].iloc[-20:].mean())
    cur_vol = float(df["volume"].iloc[-1])

    # RSI divergence check (weighted 25 pts)
    rsi_div = detect_rsi_divergence(df)
    rsi_div_match = rsi_div is not None and (
        (is_bull and rsi_div.get("type") == "bullish") or
        (not is_bull and rsi_div.get("type") == "bearish")
    )
    # RSI in zone (momentum not exhausted)
    rsi_in_zone = 35 < rsi < 65

    # Volume (weighted 25 pts) - expanding volume on signal bar
    vol_expanding = cur_vol > avg_vol * 1.3  # 30% above average = strong confirmation
    vol_present   = cur_vol > avg_vol * 1.1  # 10% above = moderate

    factors = {
        "Pattern":{"pass":True,       "label":"Pattern confirmed",                                   "weight":20},
        "RSI Div":{"pass":rsi_div_match,"label":f"RSI divergence {'confirmed' if rsi_div_match else 'not detected'}","weight":25},
        "Volume": {"pass":vol_expanding,"label":f"Volume {'spike' if vol_expanding else 'expanding' if vol_present else 'weak'} ({cur_vol/1e6:.1f}M vs avg {avg_vol/1e6:.1f}M)","weight":25},
        "EMA":    {"pass":(price>ema20 if is_bull else price<ema20),"label":f"Price {'above' if is_bull else 'below'} EMA 20 (${ema20:.2f})","weight":15},
        "VWAP":   {"pass":(price>vwap  if is_bull else price<vwap), "label":f"Price {'above' if is_bull else 'below'} VWAP (${vwap:.2f})",  "weight":15},
    }

    # Weighted confidence score
    weighted_score = sum(f["weight"] for f in factors.values() if f["pass"])
    raw_score      = sum(1 for f in factors.values() if f["pass"])

    return factors, raw_score, weighted_score, rsi, vwap, ema20

def calc_quick_levels(price, direction, atr):
    """
    For QUICK trades (weekly/0DTE): tight ATR-based levels.
    Target = 0.75x ATR, Stop = 0.4x ATR. Realistic intraday moves.
    """
    if not atr or atr <= 0: atr = price * 0.01  # fallback 1%
    is_bull = direction == "bullish"
    entry  = round(price, 2)
    target = round(price + atr * 0.75, 2) if is_bull else round(price - atr * 0.75, 2)
    stop   = round(price - atr * 0.4,  2) if is_bull else round(price + atr * 0.4,  2)
    return entry, target, stop

def build_candidates(df, ticker, toggles, account, risk_pct, dte, trade_style="swing", atr=None):
    trend_dir,trend_score,trend_factors,t_ema,t_vwap,t_rsi = get_trend(df)
    price   = float(df["close"].iloc[-1])
    regime, regime_strength = detect_market_regime(df)
    is_quick = trade_style == "quick"
    candidates = []
    raw = []
    if toggles["db"]: raw += [s for s in detect_double_bottom(df,ticker,rr_min=2.0) if s.confirmed]
    if toggles["dt"]: raw += [s for s in detect_double_top(df,ticker,rr_min=2.0)    if s.confirmed]
    if toggles["br"]: raw += [s for s in detect_break_and_retest(df,ticker,rr_min=2.0) if s.confirmed]

    for setup in raw:
        if abs(setup.entry_price - price) / price > 0.05: continue
        factors, raw_score, weighted_conf, rsi, vwap, ema20 = score_setup(df, setup)
        conflict = setup.direction != trend_dir and trend_score >= 3

        # Regime bonus: trending market boosts quick trades, both modes get swing bonus
        regime_bonus = 5 if regime == "trending" else -5
        # HTF alignment bonus (already fetched globally but we add 10 if direction matches trend)
        htf_bonus = 10 if setup.direction == trend_dir else 0

        final_conf = min(100, weighted_conf + htf_bonus + regime_bonus)

        if is_quick:
            q_entry, q_target, q_stop = calc_quick_levels(price, setup.direction, atr)
        else:
            q_entry, q_target, q_stop = setup.entry_price, setup.target, setup.stop_loss

        if conflict:
            t_entry  = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
            t_stop   = round(price*1.02,2) if trend_dir=="bearish" else round(price*0.98,2)
            if is_quick:
                _, t_target, t_stop = calc_quick_levels(price, trend_dir, atr)
                t_entry = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
            else:
                t_target = round(price*0.96,2) if trend_dir=="bearish" else round(price*1.04,2)
            conf_val = min(100, int(trend_score/5*100) + regime_bonus)
            candidates.append({"source":"trend_override","direction":trend_dir,
                "confidence":conf_val,"score":trend_score,"factors":trend_factors,
                "conflict":True,"conflict_pattern":setup.pattern,
                "entry":t_entry,"stop":t_stop,"target":t_target,
                "pattern_label":"Trend Override","rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
                "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})
        else:
            candidates.append({"source":"pattern","direction":setup.direction,
                "confidence":final_conf,"score":raw_score,"factors":factors,"conflict":False,
                "entry":q_entry,"stop":q_stop,"target":q_target,
                "pattern_label":setup.pattern.replace("Double","Double ").replace("BreakRetest","Break & Retest"),
                "rsi":rsi,"vwap":vwap,"ema20":ema20,"rr":setup.rr_ratio,
                "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})

    if trend_score >= 3:
        t_entry  = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
        if is_quick:
            _, t_target, t_stop = calc_quick_levels(price, trend_dir, atr)
            t_entry = round(price*(0.998 if trend_dir=="bearish" else 1.002),2)
        else:
            t_stop   = round(price*1.02,2) if trend_dir=="bearish" else round(price*0.98,2)
            t_target = round(price*0.96,2) if trend_dir=="bearish" else round(price*1.04,2)
        trend_conf = min(100, int(trend_score/5*100) + regime_bonus)
        candidates.append({"source":"trend","direction":trend_dir,
            "confidence":trend_conf,"score":trend_score,"factors":trend_factors,"conflict":False,
            "entry":t_entry,"stop":t_stop,"target":t_target,
            "pattern_label":f"{'Bearish' if trend_dir=='bearish' else 'Bullish'} Trend",
            "rsi":t_rsi,"vwap":t_vwap,"ema20":t_ema,
            "regime":regime,"regime_strength":regime_strength,"trade_style":trade_style})

    seen = {}
    for c in sorted(candidates, key=lambda x:x["confidence"], reverse=True):
        k = f"{c['direction']}_{c['pattern_label']}"
        if k not in seen: seen[k] = c
    return sorted(seen.values(), key=lambda x:x["confidence"], reverse=True)[:3]

def load_journal():
    if "trade_journal" not in st.session_state: st.session_state.trade_journal = []
    return st.session_state.trade_journal

def log_trade(ticker, sig, opt, gates_passed, gates_total, elevate):
    journal = load_journal()
    journal.append({
        "Date":        datetime.now().strftime("%m/%d %H:%M"),
        "Ticker":      ticker,
        "Action":      "CALL" if sig["direction"]=="bullish" else "PUT",
        "Pattern":     sig["pattern_label"],
        "Strike":      f"${opt['strike']:.2f}",
        "Entry":       f"${opt['entry']:.2f}",
        "Target":      f"${opt['target']:.2f}",
        "Stop":        f"${opt['stop']:.2f}",
        "Premium":     f"${opt['premium']:.2f}",
        "Contracts":   opt["contracts"],
        "Max Loss":    f"${opt['max_loss']:.0f}",
        "Pot. Profit": f"${opt['profit_at_target']:,.0f}",
        "Confidence":  f"{sig['confidence']}%",
        "Gate Score":  f"{gates_passed}/{gates_total}",
        "Elevated":    "YES" if elevate else "no",
        "Expiry":      opt["expiration"],
        "Result":      "Open",
        "P&L $":       "",
    })
    st.session_state.trade_journal = journal[-200:]

def get_journal_stats():
    journal = load_journal()
    if not journal: return {}
    stats = {}
    for t in journal:
        tk = t["Ticker"]
        if tk not in stats: stats[tk] = {"total":0,"wins":0,"losses":0,"open":0,"calls":0,"puts":0}
        stats[tk]["total"] += 1
        r = t.get("Result","Open")
        if r=="Open":          stats[tk]["open"]   += 1
        elif "Win"  in r:      stats[tk]["wins"]   += 1
        elif "Loss" in r:      stats[tk]["losses"] += 1
        if t["Action"]=="CALL": stats[tk]["calls"] += 1
        else:                   stats[tk]["puts"]  += 1
    return stats

def build_share_text(ticker, sig, opt, gates_passed, gates_total, elevate, market_status):
    direction = "CALL" if sig["direction"]=="bullish" else "PUT"
    elevated  = "YES - ALL GATES PASSED" if elevate else f"NO - {gates_passed}/7 gates"
    sep = "=" * 32
    return (f"OPTIONS SCREENER v6.0 SIGNAL\n{sep}\n"
            f"{ticker} - BUY {direction}\n"
            f"Pattern:   {sig['pattern_label']}\n"
            f"Conf:      {sig['confidence']}%\n"
            f"Gate:      {gates_passed}/7 | Elevated: {elevated}\n{sep}\n"
            f"Strike:    ${opt['strike']:.2f}\n"
            f"Premium:   ${opt['premium']:.2f}/share\n"
            f"Entry:     ${opt['entry']:.2f}\n"
            f"Target:    ${opt['target']:.2f}\n"
            f"Stop:      ${opt['stop']:.2f}\n"
            f"R:R:       {opt['rr']}x\n"
            f"Delta:     {opt['delta']:.2f}\n"
            f"Contracts: {opt['contracts']}\n"
            f"Position:  ${opt['position_dollars']:.0f} ({opt['pct_of_account']}% of acct)\n"
            f"Max Loss:  ${opt['max_loss']:.0f}\n"
            f"Profit:    ${opt['profit_at_target']:,.0f}\n"
            f"Expires:   {opt['expiration']}\n{sep}\n"
            f"EXIT RULES:\n"
            f"Take 50% when option hits ${opt['exit_take_half']:.2f} (100% gain)\n"
            f"Close 100% if stock closes beyond ${opt['exit_stop_stock']:.2f}\n{sep}\n"
            f"Market: {market_status}\n"
            f"Time:   {datetime.now().strftime('%m/%d/%Y %H:%M')}\n"
            f"NOT FINANCIAL ADVICE")

# ── AI Trade Brief ───────────────────────────────────────────────────────────
def get_ai_brief(ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status):
    """
    Calls Claude API with full signal context.
    Returns a structured verdict: rating, reasoning, key risk.
    """
    import urllib.request
    import json

    is_bull     = sig["direction"] == "bullish"
    action      = "CALL" if is_bull else "PUT"
    gate_lines  = "\n".join(["  - " + k + ": " + ("PASS" if v["pass"] else "FAIL") + " (" + v["label"] + ")" for k,v in gates.items()])
    div         = detect_rsi_divergence_text(sig)

    prompt = f"""You are an expert options trader reviewing a technical setup. Give a concise professional assessment.

TICKER: {ticker}
SIGNAL: BUY {action}
Pattern: {sig['pattern_label']}
Confidence Score: {sig['confidence']}%
Gate Score: {gates_passed}/7

PRICE DATA:
- Entry: ${opt['entry']:.2f}
- Strike: ${opt['strike']:.2f}
- Target: ${opt['target']:.2f}
- Stop: ${opt['stop']:.2f}
- R:R Ratio: {opt['rr']}x
- Delta: {opt['delta']:.2f}
- Premium: ${opt['premium']:.2f}
- Expiration: {opt['expiration']}

7-POINT GATE RESULTS:
{gate_lines}

ADDITIONAL CONTEXT:
- IV Rank: {iv_rank if iv_rank is not None else 'unavailable'}%
- Earnings: {'None within 14 days' if earnings_days is None else f'In {earnings_days} days - HIGH RISK'}
- Entry timing: {conf_status}

Respond in exactly this format, no extra text:
RATING: [Strong Setup / Moderate Setup / Weak Setup / Do Not Trade]
REASONING: [2-3 sentences on why the setup quality is good or bad based on the data above]
KEY RISK: [1 sentence on the single biggest risk to this trade]
EDGE: [1 sentence on what gives this trade its edge if taken]"""

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()

def detect_rsi_divergence_text(sig):
    return sig.get("rsi_div", "not checked")

def parse_ai_brief(text):
    """Parse the structured AI response into parts."""
    lines  = text.strip().splitlines()
    parsed = {}
    for line in lines:
        if line.startswith("RATING:"):    parsed["rating"]    = line.replace("RATING:","").strip()
        elif line.startswith("REASONING:"): parsed["reasoning"] = line.replace("REASONING:","").strip()
        elif line.startswith("KEY RISK:"): parsed["risk"]      = line.replace("KEY RISK:","").strip()
        elif line.startswith("EDGE:"):    parsed["edge"]      = line.replace("EDGE:","").strip()
    return parsed

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## OPTIONS SCREENER v6")
    st.markdown("---")
    selected_ticker = st.selectbox("TICKER", WATCHLIST)
    custom = st.text_input("Or type any ticker","").upper().strip()
    if custom: selected_ticker = custom
    selected_tf = st.selectbox("TIMEFRAME", list(TIMEFRAMES.keys()), index=2)
    st.markdown("---")
    st.markdown("**TRADE STYLE**")
    trade_style = st.radio("Mode", ["⚡ Quick (weekly/0DTE)", "📅 Swing (2-4 week)"], index=1, horizontal=True)
    trade_style = "quick" if "Quick" in trade_style else "swing"
    if trade_style == "quick":
        st.caption("Tight ATR targets. 20min-2hr hold. Market hours only.")
    else:
        st.caption("Pattern measured move. Multi-day hold.")
    st.markdown("---")
    st.markdown("**PATTERNS TO SCAN**")
    tog_db    = st.toggle("Double Bottom (calls)", value=True)
    tog_br_up = st.toggle("Break & Retest Up (calls)", value=True)
    tog_dt    = st.toggle("Double Top (puts)",    value=True)
    tog_br_dn = st.toggle("Break & Retest Down (puts)", value=True)
    toggles   = {"db":tog_db, "dt":tog_dt, "br":tog_br_up or tog_br_dn}
    st.markdown("---")
    st.markdown("**ACCOUNT SETTINGS**")
    account_size = st.number_input("Account Size ($)", value=10000, step=1000)
    risk_pct     = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    if trade_style == "quick":
        dte = st.selectbox("Days to Expiration", [0,1,2,3,5,7], index=2,
                           help="0 = 0DTE (today), 1-7 = this week's expiry")
    else:
        dte = st.selectbox("Days to Expiration", [14,21,30,45,60], index=2)
    st.markdown("---")
    st.markdown("**AUTO REFRESH**")
    refresh_on       = st.toggle("Live refresh (manual)", value=False)
    refresh_interval = st.selectbox("Interval",["1 min","5 min","15 min"],index=1) if refresh_on else None
    st.markdown("---")
    if POLYGON_API_KEY: st.success("LIVE DATA")
    else:               st.warning("DEMO MODE")
    if ANTHROPIC_API_KEY: st.success("AI BRIEF READY")
    else:                 st.info("AI Brief: add ANTHROPIC_API_KEY to enable")

# Smart auto-refresh - activates automatically when watch queue has pending signals
init_watch_queue()
_queue_active = any(item["status"] != "CONFIRMED" for item in st.session_state.watch_queue.values())
if AUTOREFRESH_AVAILABLE:
    if _queue_active:
        # Watch queue running - auto refresh every 60 seconds no manual toggle needed
        st_autorefresh(interval=60000, key="watch_autorefresh")
    elif refresh_on and refresh_interval:
        ms = {"1 min":60000,"5 min":300000,"15 min":900000}.get(refresh_interval,300000)
        st_autorefresh(interval=ms, key="manual_autorefresh")

tf_mult,tf_span,tf_days = TIMEFRAMES[selected_tf]
df            = fetch_ohlcv(selected_ticker, tf_mult, tf_span, tf_days)
current_price = fetch_current_price(selected_ticker) or float(df["close"].iloc[-1])
prev_close    = float(df["close"].iloc[-2]) if len(df)>1 else current_price
pct_change    = ((current_price-prev_close)/prev_close)*100
iv_rank, hv   = fetch_iv_rank(selected_ticker)
earnings_days = check_earnings(selected_ticker)

# ── ATR calculation (14-period) ───────────────────────────────────────────────
def calc_atr(df, period=14):
    if len(df) < period + 1: return None
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 2)

atr = calc_atr(df)

# ── Higher timeframe confluence ───────────────────────────────────────────────
# Pull daily bars to check if the higher timeframe trend agrees with the signal
@st.cache_data(ttl=300)
def fetch_htf_trend(ticker):
    """Fetch daily data and return trend + RSI for confluence check."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 20: return None, None, None
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        close = raw["close"].astype(float)
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])
        # Daily RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - (100/(1+(gain/loss)))).iloc[-1])
        trend = "bullish" if price > ema20 else "bearish"
        return trend, round(rsi, 1), round(ema20, 2)
    except:
        return None, None, None

htf_trend, htf_rsi, htf_ema = fetch_htf_trend(selected_ticker)

# Liquidity check (cached, runs silently in background)
liq_ok, liq_vol, liq_oi, liq_msg = check_liquidity(selected_ticker)

# ── Background watch loop - runs every refresh for ALL watched tickers ────────
any_new_confirm = run_background_watch_checks(tf_mult, tf_span, tf_days)

# Sound alert when any ticker just confirmed
if any_new_confirm:
    st.markdown("""
    <audio autoplay>
      <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAA
EAAQARAAAAIgAA//8AZGFOaghzCAFyCAFzCAFyCAFzCAFzCAFzCAFyCAFzCAFzCAFy
CAFzCAFzCAFzCAFyCAFzCAFzCAFyCAFzCAFzCAFzCAFyCAFzCAFzCAFyCAFzCAFzCAF" type="audio/wav">
    </audio>
    """, unsafe_allow_html=True)

# ── Watch queue banner - always visible regardless of selected ticker ─────────
init_watch_queue()
queue = st.session_state.watch_queue
if queue:
    for key, item in list(queue.items()):
        elapsed_mins = int((datetime.now() - item["added_at"]).total_seconds() / 60)
        elapsed_secs = int((datetime.now() - item["added_at"]).total_seconds() % 60)
        last_chk = ""
        if item["last_checked"]:
            secs_ago = int((datetime.now() - item["last_checked"]).total_seconds())
            last_chk = f" | checked {secs_ago}s ago"

        # Build candle history dots
        candle_html = ""
        for c in item.get("candles", []):
            if c == "green":   candle_html += "<span style='color:#00d4aa;font-size:1rem'>&#9650;</span> "
            elif c == "red":   candle_html += "<span style='color:#ff4d6d;font-size:1rem'>&#9660;</span> "
            else:              candle_html += "<span style='color:#8899aa;font-size:0.8rem'>&#9644;</span> "

        status = item["status"]
        if status == "CONFIRMED":
            banner_style = "background:#061a10;border:2px solid #00d4aa;border-radius:8px;padding:12px 16px;margin:4px 0;"
            icon  = "✅"
            stxt  = "<span style='color:#00d4aa;font-weight:700;font-size:1rem'>ENTRY CONFIRMED</span>"
        elif status == "WAITING":
            banner_style = "background:#0d1219;border:2px solid #f0c040;border-radius:8px;padding:12px 16px;margin:4px 0;"
            icon  = "👁"
            stxt  = "<span style='color:#f0c040;font-weight:700'>WATCHING</span>"
        else:
            banner_style = "background:#1a0a0a;border:2px solid #ff4d6d;border-radius:8px;padding:12px 16px;margin:4px 0;"
            icon  = "⏳"
            stxt  = "<span style='color:#ff4d6d;font-weight:700'>ENTRY EARLY</span>"

        col_banner, col_dismiss = st.columns([6,1])
        with col_banner:
            st.markdown(f"""
            <div style='{banner_style}'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                        <span style='font-size:1.1rem'>{icon}</span>
                        <b style='margin-left:6px'>{item['ticker']} {item['action']}</b>
                        <span style='color:#8899aa;font-size:0.82rem;margin-left:8px'>{item['pattern']} | Strike ${item['strike']:.2f}</span>
                    </div>
                    <div style='color:#8899aa;font-size:0.75rem;font-family:monospace'>{elapsed_mins}m {elapsed_secs}s{last_chk}</div>
                </div>
                <div style='margin-top:6px'>{stxt} &nbsp; {candle_html}</div>
                <div style='color:#e0e6f0;font-size:0.82rem;margin-top:4px'>{item['message']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_dismiss:
            if st.button("Dismiss", key=f"dismiss_{key}"):
                remove_from_watch_queue(key)
                st.rerun()

mstatus, mtext = get_market_status()
css_class = {"open":"market-open","pre":"market-pre","after":"market-pre","closed":"market-closed"}.get(mstatus,"market-closed")
st.markdown(f"<div class='{css_class}'>{mtext}</div>", unsafe_allow_html=True)

if earnings_days is not None:
    if earnings_days <= 1:   st.error(f"EARNINGS {'TODAY' if earnings_days==0 else 'TOMORROW'} on {selected_ticker} - Avoid new options positions.")
    elif earnings_days <= 7: st.error(f"EARNINGS IN {earnings_days} DAYS on {selected_ticker} - 7-point gate will block.")
    else:                    st.warning(f"Earnings in {earnings_days} days on {selected_ticker} - premiums may be inflated.")

c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1:
    color   = "#00d4aa" if pct_change>=0 else "#ff4d6d"
    arrow   = "UP" if pct_change>=0 else "DN"
    prepost = "" if mstatus=="open" else " <span style='color:#f0c040;font-size:0.72rem'>(delayed)</span>"
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.8rem'>{selected_ticker} . {selected_tf}</div><div class='big-price'>${current_price:,.2f}{prepost}</div><div style='color:{color}'>{arrow} {pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
with c2:
    ema20v = float(df["close"].ewm(span=20).mean().iloc[-1])
    above  = current_price > ema20v
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>TREND</div><div style='font-weight:700;color:{'#00d4aa' if above else '#ff4d6d'}'>{'BULL' if above else 'BEAR'}</div></div>", unsafe_allow_html=True)
with c3:
    vol = float(df["volume"].iloc[-1])
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>VOLUME</div><div style='font-weight:700'>{vol/1e6:.1f}M</div></div>", unsafe_allow_html=True)
with c4:
    iv_color = "#00d4aa" if iv_rank is not None and iv_rank<50 else "#f0c040" if iv_rank is not None and iv_rank<70 else "#ff4d6d"
    iv_text  = f"{iv_rank}%" if iv_rank is not None else "N/A"
    st.markdown(f"<div class='metric-card'><div style='color:#8899aa;font-size:0.75rem'>IV RANK</div><div style='font-weight:700;color:{iv_color}'>{iv_text}</div></div>", unsafe_allow_html=True)

div = detect_rsi_divergence(df)
if div:
    css = f"divergence-{'bull' if div['type']=='bullish' else 'bear'}"
    st.markdown(f"<div class='{css}'><b>{div['label']}</b><br>{div['detail']}</div>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["SIGNALS","CHART","BACKTEST","SCAN","JOURNAL"])

with tab1:
    candidates = build_candidates(df, selected_ticker, toggles, account_size, risk_pct, dte, trade_style=trade_style, atr=atr)
    if not candidates:
        st.markdown("""<div style='background:#111827;border:2px solid #1e2d40;border-radius:12px;padding:24px;text-align:center;color:#8899aa'>
            <div style='font-size:1rem;font-weight:700;margin:8px 0'>NO SIGNALS FOUND</div>
            <div style='font-size:0.85rem'>Try Daily or 4 Hour timeframe, enable more patterns, or check a different ticker.</div>
        </div>""", unsafe_allow_html=True)
    else:
        rank_labels   = ["BEST","BETTER","GOOD"]
        rank_classes  = ["rank-best","rank-better","rank-good"]
        badge_classes = ["badge-best","badge-better","badge-good"]
        conf_classes  = ["conf-num-best","conf-num-better","conf-num-good"]
        rank_icons    = ["🥇","🥈","🥉"]

        for i, sig in enumerate(candidates):
            rl = rank_labels[i]  if i<3 else f"#{i+1}"
            rc = rank_classes[i] if i<3 else "rank-good"
            bc = badge_classes[i]if i<3 else "badge-good"
            cc = conf_classes[i] if i<3 else "conf-num-good"
            ri = rank_icons[i]   if i<3 else ""

            is_bull   = sig["direction"] == "bullish"
            dir_color = "#00d4aa" if is_bull else "#ff4d6d"
            dir_label = "BUY CALL" if is_bull else "BUY PUT"

            # Trade style badge
            sig_style = sig.get("trade_style", trade_style)
            if sig_style == "quick":
                style_badge = "<span style='background:#1a0a3a;color:#aa88ff;font-family:monospace;font-size:0.68rem;padding:2px 7px;border-radius:10px;margin-left:6px'>⚡ QUICK</span>"
            else:
                style_badge = "<span style='background:#0a1a2a;color:#6699cc;font-family:monospace;font-size:0.68rem;padding:2px 7px;border-radius:10px;margin-left:6px'>📅 SWING</span>"

            # Liquidity warning (silent fail - only shows if explicitly illiquid)
            liq_warn = "" if liq_ok else "<span style='color:#f0c040;font-size:0.75rem;margin-left:8px'>⚠ Low liquidity</span>"

            # Regime indicator (1 line, subtle)
            sig_regime   = sig.get("regime","unknown")
            regime_icon  = "📈" if sig_regime=="trending" else "↔️" if sig_regime=="choppy" else ""
            regime_label = sig_regime.upper() if sig_regime != "unknown" else ""

            conflict_html = ""
            if sig.get("conflict"):
                pname = sig.get("conflict_pattern","pattern")
                conflict_html = f"<div class='conflict-warn'>Pattern {pname} found but trend overrides - showing {'PUT' if not is_bull else 'CALL'}.</div>"

            # Quick trade warning if market closed
            quick_warn_html = ""
            if sig_style == "quick" and mstatus != "open":
                quick_warn_html = "<div style='background:#1a150a;border:1px solid #f0c040;border-radius:6px;padding:8px 12px;margin-bottom:6px;color:#f0c040;font-size:0.8rem'>⚡ Quick trades require market to be open. Levels shown are based on current price.</div>"

            dots_html = ""
            for f in sig["factors"].values():
                dot = "dot-green" if f["pass"] else "dot-red"
                dots_html += f"<div class='factor-row'><span class='{dot}'></span><span style='color:{'#e0e6f0' if f['pass'] else '#8899aa'}'>{f['label']}</span></div>"

            st.markdown(f"""
            {conflict_html}
            {quick_warn_html}
            <div class='{rc}'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <div>
                        <span class='rank-badge {bc}'>{ri} {rl}</span>{style_badge}{liq_warn}
                        <div style='font-size:1.1rem;font-weight:700;color:{dir_color};margin-top:4px'>{dir_label} - {selected_ticker}</div>
                        <div style='color:#8899aa;font-size:0.82rem;margin-top:2px'>{sig['pattern_label']} &nbsp;<span style='font-size:0.75rem'>{regime_icon} {regime_label}</span></div>
                    </div>
                    <div class='{cc}'>{sig['confidence']}%</div>
                </div>
                <div style='margin-top:10px'>{dots_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if sig["confidence"] >= 60:
                opt = calc_trade(sig["entry"],sig["stop"],sig["target"],sig["direction"],dte,account_size,risk_pct,current_price,atr=atr)
                gates, gates_passed, elevate = run_seven_point_gate(df,sig,opt,iv_rank,earnings_days,opt["actual_dte"])
                est_days, dte_rec = estimate_move_timeframe(sig["pattern_label"])
                gate_color = "#00d4aa" if gates_passed>=6 else "#f0c040" if gates_passed>=4 else "#ff4d6d"
                elev_badge = "<span style='background:#00d4aa22;color:#00d4aa;padding:2px 8px;border-radius:10px;font-size:0.72rem;margin-left:8px'>PRIME SETUP</span>" if elevate else ""
                gates_dots = ""
                for gname, gdata in gates.items():
                    if gdata["critical"] and not gdata["pass"]: dot = "dot-red"
                    elif gdata["pass"]:                          dot = "dot-green"
                    else:                                        dot = "dot-yellow"
                    g_color = "#e0e6f0" if gdata["pass"] else "#8899aa"
                    # Sanitize label - remove any characters that could break HTML
                    g_label = str(gdata["label"]).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;")
                    gates_dots += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + g_color + ";font-size:0.78rem'><b>" + gname + ":</b> " + g_label + "</span></div>"

                # Build as plain string - no f-string so special chars in labels cant break it
                gate_html = (
                    "<div class='gate-box'>"
                    "<div style='display:flex;align-items:center;margin-bottom:8px'>"
                    "<span style='color:" + gate_color + ";font-family:monospace;font-size:0.78rem;font-weight:700'>7-POINT GATE: " + str(gates_passed) + "/7 PASSED</span>"
                    + elev_badge +
                    "</div>"
                    + gates_dots +
                    "<div style='color:#8899aa;font-size:0.75rem;margin-top:6px'>Pattern needs ~" + str(est_days) + " days to play out | Recommended DTE: " + str(dte_rec) + "+ days</div>"
                    "</div>"
                )
                st.markdown(gate_html, unsafe_allow_html=True)

                # ── HTF Confluence ────────────────────────────────────────────
                if htf_trend is not None:
                    htf_agrees = htf_trend == sig["direction"]
                    htf_color  = "#00d4aa" if htf_agrees else "#ff4d6d"
                    htf_icon   = "✅" if htf_agrees else "⚠️"
                    htf_label  = ("DAILY TREND CONFIRMS" if htf_agrees else "DAILY TREND CONFLICTS")
                    htf_detail = "Daily chart agrees — higher timeframe is aligned." if htf_agrees else "Daily chart is moving the other way. Extra caution — counter-trend trade."
                    htf_html = (
                        "<div style='background:#0d1219;border:1px solid " + htf_color + "33;border-radius:8px;padding:10px 14px;margin-top:6px'>"
                        "<div style='display:flex;align-items:center;gap:8px'>"
                        "<span>" + htf_icon + "</span>"
                        "<span style='color:" + htf_color + ";font-family:monospace;font-size:0.72rem;font-weight:700'>" + htf_label + "</span>"
                        "<span style='color:#8899aa;font-size:0.78rem;margin-left:4px'>Daily trend: " + htf_trend.upper() + " | RSI " + str(htf_rsi) + " | EMA20 $" + str(htf_ema) + "</span>"
                        "</div>"
                        "<div style='color:#8899aa;font-size:0.78rem;margin-top:4px'>" + htf_detail + "</div>"
                        "</div>"
                    )
                    st.markdown(htf_html, unsafe_allow_html=True)

                # ── Move probability ──────────────────────────────────────────
                tr_color = "#00d4aa" if opt["target_realistic"]=="Likely" else "#f0c040" if opt["target_realistic"]=="Possible" else "#ff4d6d"
                atr_txt  = (str(opt["atr_multiples"]) + "x ATR needed") if opt["atr_multiples"] else ""
                move_html = (
                    "<div style='background:#0d1219;border:1px solid #1e2d40;border-radius:8px;padding:10px 14px;margin-top:6px'>"
                    "<div style='display:flex;justify-content:space-between;align-items:center'>"
                    "<span style='color:#8899aa;font-family:monospace;font-size:0.72rem'>MOVE REQUIRED</span>"
                    "<span style='color:" + tr_color + ";font-weight:700;font-size:0.85rem'>" + opt["target_realistic"].upper() + "</span>"
                    "</div>"
                    "<div style='margin-top:4px;font-size:0.82rem'>"
                    "Price needs to move <b style='color:#e0e6f0'>" + str(opt["move_pct"]) + "%</b>"
                    + (" &nbsp;|&nbsp; <span style='color:#8899aa'>" + atr_txt + "</span>" if atr_txt else "") +
                    "</div>"
                    "<div style='color:#8899aa;font-size:0.75rem;margin-top:2px'>"
                    + ("Likely = &le;2x ATR &nbsp; Possible = 2-4x ATR &nbsp; Ambitious = 4x+ ATR" if opt["atr_multiples"] else "") +
                    "</div>"
                    "</div>"
                )
                st.markdown(move_html, unsafe_allow_html=True)

                if not opt["delta_ok"]:
                    st.markdown(f"<div style='background:#1a150a;border:1px solid #f0c040;border-radius:6px;padding:8px 12px;margin-top:6px;color:#f0c040;font-size:0.8rem'>Delta {opt['delta']:.2f} outside 0.35-0.85 ideal range</div>", unsafe_allow_html=True)

                delta_color = "#00d4aa" if opt["delta_ok"] else "#f0c040"
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
                    <div style='margin-top:8px;padding-top:8px;border-top:1px solid #1e2d40;display:flex;justify-content:space-between;align-items:center'>
                        <div><div style='color:#8899aa;font-size:0.72rem'>EXPIRES</div><div style='font-weight:700'>{opt['expiration']}</div></div>
                        <div style='text-align:right'><div style='color:#8899aa;font-size:0.72rem'>POSITION SIZE</div><div style='font-weight:700'>${opt['position_dollars']:.0f} <span style='color:#8899aa;font-size:0.75rem'>({opt['pct_of_account']}% of account)</span></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if sig_style == "quick":
                    exit_hold = "Close within 20-60 minutes regardless of outcome. Do not hold into close."
                    exit_take = f"Take 50% off at ${opt['exit_take_half']:.2f}/sh (100% gain on option). Let rest run with tight mental stop."
                else:
                    exit_take = f"Take 50% off when option reaches ${opt['exit_take_half']:.2f}/sh (100% gain). Let remaining 50% run with no stop."
                    exit_hold = "Never hold through earnings. Never add to a losing position. Never let a winner turn into a loser."

                st.markdown(f"""
                <div class='exit-rules'>
                    <div style='color:#00d4aa;font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:6px'>EXIT RULES - DECIDE BEFORE YOU ENTER</div>
                    <div style='margin:4px 0'><b>Take 50% off:</b> {exit_take}</div>
                    <div style='margin:4px 0'><b>Close 100%</b> if {selected_ticker} closes {'below' if is_bull else 'above'} <b style='color:#ff4d6d'>${opt['exit_stop_stock']:.2f}</b> - pattern failed, no questions asked.</div>
                    <div style='margin:4px 0;color:#8899aa;font-size:0.8rem'>{exit_hold}</div>
                </div>
                """, unsafe_allow_html=True)

                # Entry timing check + watch queue - market hours only
                watch_key = f"{selected_ticker}_{sig['direction']}"
                already_watching = watch_key in st.session_state.get("watch_queue", {})
                conf_status = "N/A"

                if mstatus == "open":
                    conf_result = check_entry_confirmation(df, sig["direction"])
                    conf_status = conf_result["status"]
                    if conf_status == "CONFIRMED":
                        conf_bg = "#061a10"; conf_border = "#00d4aa"; conf_color = "#00d4aa"; conf_icon = "✅"
                    elif conf_status == "WAITING":
                        conf_bg = "#0d1219"; conf_border = "#f0c040"; conf_color = "#f0c040"; conf_icon = "👁"
                    else:
                        conf_bg = "#1a0a0a"; conf_border = "#ff4d6d"; conf_color = "#ff4d6d"; conf_icon = "⏳"

                    candle_html = ""
                    for c in conf_result.get("candles", []):
                        if c == "green":   candle_html += "<span style='color:#00d4aa'>&#9650;</span> "
                        elif c == "red":   candle_html += "<span style='color:#ff4d6d'>&#9660;</span> "
                        else:              candle_html += "<span style='color:#8899aa'>&#9644;</span> "

                    st.markdown(f"""
                    <div style='background:{conf_bg};border:1px solid {conf_border};border-radius:8px;padding:10px 14px;margin-top:8px'>
                        <div style='color:{conf_color};font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:4px'>ENTRY TIMING CHECK</div>
                        <div style='display:flex;align-items:center;gap:10px'>
                            <span style='font-size:1.1rem'>{conf_icon}</span>
                            <span style='font-weight:700;color:{conf_color}'>{conf_status}</span>
                            <span style='color:#8899aa;font-size:0.82rem'>Recent candles: {candle_html}</span>
                        </div>
                        <div style='color:#e0e6f0;font-size:0.82rem;margin-top:4px'>{conf_result["message"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Auto-add 7/7 elevated signals to watch queue
                    if elevate and not already_watching and conf_status != "CONFIRMED":
                        add_to_watch_queue(selected_ticker, sig["direction"], sig, opt)
                else:
                    st.markdown("<div style='background:#0d1219;border:1px solid #1e2d40;border-radius:8px;padding:10px 14px;margin-top:8px;color:#8899aa;font-size:0.82rem'>⏸ Entry timing check runs during market hours only (9:30 AM - 4:00 PM ET)</div>", unsafe_allow_html=True)

                if mstatus == "open":
                    if ANTHROPIC_API_KEY:
                        ai_key = f"ai_result_{selected_ticker}_{i}"
                        if st.button(f"🤖 Get AI Brief #{i+1}", key=f"ai_{i}"):
                            with st.spinner("Analyzing setup..."):
                                try:
                                    ai_text   = get_ai_brief(selected_ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status)
                                    ai_parsed = parse_ai_brief(ai_text)
                                    st.session_state[ai_key] = ai_parsed
                                except Exception as e:
                                    st.session_state[ai_key] = {"error": str(e)}

                        if ai_key in st.session_state:
                            ai = st.session_state[ai_key]
                            if "error" in ai:
                                st.error(f"AI call failed: {ai['error']}")
                            else:
                                rating = ai.get("rating","")
                                if "Strong" in rating:     r_color = "#00d4aa"; r_bg = "#061a10"; r_border = "#00d4aa"
                                elif "Moderate" in rating: r_color = "#f0c040"; r_bg = "#1a150a"; r_border = "#f0c040"
                                else:                      r_color = "#ff4d6d"; r_bg = "#1a0a0a"; r_border = "#ff4d6d"
                                st.markdown(f"""
                                <div style='background:{r_bg};border:1px solid {r_border};border-radius:8px;padding:14px;margin-top:8px'>
                                    <div style='color:#8899aa;font-family:monospace;font-size:0.72rem;letter-spacing:1px;margin-bottom:6px'>AI TRADE BRIEF</div>
                                    <div style='font-size:1.1rem;font-weight:700;color:{r_color};margin-bottom:10px'>🤖 {rating}</div>
                                    <div style='margin:6px 0;font-size:0.85rem'><span style='color:#8899aa'>REASONING</span><br>{ai.get("reasoning","")}</div>
                                    <div style='margin:6px 0;font-size:0.85rem'><span style='color:#ff4d6d'>KEY RISK</span><br>{ai.get("risk","")}</div>
                                    <div style='margin:6px 0;font-size:0.85rem'><span style='color:#00d4aa'>EDGE</span><br>{ai.get("edge","")}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='ai-placeholder'>🤖 AI Trade Brief - Add ANTHROPIC_API_KEY in Railway to enable</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:#0d1219;border:1px solid #1e2d40;border-radius:8px;padding:10px 14px;margin-top:8px;color:#8899aa;font-size:0.82rem'>🤖 AI Trade Brief runs during market hours only (9:30 AM - 4:00 PM ET)</div>", unsafe_allow_html=True)

                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    if st.button(f"Log to Journal #{i+1}", key=f"log_{i}"):
                        log_trade(selected_ticker, sig, opt, gates_passed, 7, elevate)
                        st.success("Logged!")
                with bcol2:
                    share_text = build_share_text(selected_ticker,sig,opt,gates_passed,7,elevate,mtext)
                    st.download_button(f"Share #{i+1}", data=share_text,
                        file_name=f"{selected_ticker}_signal_{datetime.now().strftime('%m%d_%H%M')}.txt",
                        mime="text/plain", key=f"share_{i}")
                with bcol3:
                    if not already_watching:
                        if st.button(f"Watch #{i+1}", key=f"watch_{i}"):
                            add_to_watch_queue(selected_ticker, sig["direction"], sig, opt)
                            st.success("Added to watch queue!")
                            st.rerun()
                    else:
                        if st.button(f"Stop Watching #{i+1}", key=f"unwatch_{i}"):
                            remove_from_watch_queue(watch_key)
                            st.rerun()

            if i < len(candidates)-1:
                st.markdown("<hr style='border-color:#1e2d40;margin:12px 0'>", unsafe_allow_html=True)

with tab2:
    chart_db = [s for s in detect_double_bottom(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_dt = [s for s in detect_double_top(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_br = [s for s in detect_break_and_retest(df,selected_ticker,rr_min=2.0) if s.confirmed]
    chart_setups = chart_db + chart_dt + chart_br
    tf_formats = {"5 Min":"%H:%M","15 Min":"%H:%M","1 Hour":"%b %d %H:%M","4 Hour":"%b %d","Daily":"%b %d '%y"}
    tick_format = tf_formats.get(selected_tf,"%b %d")
    price_min   = float(df["low"].min())*0.995
    price_max   = float(df["high"].max())*1.005
    price_range = price_max - price_min
    raw_tick    = price_range/8
    magnitude   = 10**(len(str(int(max(raw_tick,1))))-1)
    tick_interval = max(round(raw_tick/magnitude)*magnitude, 0.01)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df["timestamp"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name=selected_ticker,increasing_line_color="#00d4aa",decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00d4aa",decreasing_fillcolor="#ff4d6d",line_width=1),row=1,col=1)
    ema_line  = df["close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df["timestamp"],y=ema_line,name="EMA 20",
        line=dict(color="#f0c040",width=1.5,dash="dot"),hovertemplate="EMA 20: $%{y:.2f}<extra></extra>"),row=1,col=1)
    tp        = (df["high"]+df["low"]+df["close"])/3
    vwap_line = (tp*df["volume"]).cumsum()/df["volume"].cumsum()
    fig.add_trace(go.Scatter(x=df["timestamp"],y=vwap_line,name="VWAP",
        line=dict(color="#9966ff",width=1.5,dash="dash"),hovertemplate="VWAP: $%{y:.2f}<extra></extra>"),row=1,col=1)
    for s in chart_setups[:3]:
        lc = "#00d4aa" if s.direction=="bullish" else "#ff4d6d"
        fig.add_hline(y=s.entry_price,line_dash="solid",line_color=lc,line_width=1.5,opacity=0.8,
            annotation_text=f"  Entry ${s.entry_price:.2f}",annotation_font_color=lc,annotation_font_size=11,row=1,col=1)
        fig.add_hline(y=s.target,line_dash="dash",line_color="#00d4aa",line_width=1,opacity=0.6,
            annotation_text=f"  Target ${s.target:.2f}",annotation_font_color="#00d4aa",annotation_font_size=11,row=1,col=1)
        fig.add_hline(y=s.stop_loss,line_dash="dot",line_color="#ff4d6d",line_width=1,opacity=0.6,
            annotation_text=f"  Stop ${s.stop_loss:.2f}",annotation_font_color="#ff4d6d",annotation_font_size=11,row=1,col=1)
        fig.add_hrect(y0=min(s.entry_price,s.target),y1=max(s.entry_price,s.target),fillcolor="rgba(0,212,170,0.05)",line_width=0,row=1,col=1)
    vol_colors = ["#00d4aa" if c>=o else "#ff4d6d" for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"],y=df["volume"],marker_color=vol_colors,opacity=0.5,name="Volume",
        hovertemplate="%{x}<br>Vol: %{y:,.0f}<extra></extra>"),row=2,col=1)
    fig.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0d1219",font=dict(color="#e0e6f0",size=12),
        height=520,xaxis_rangeslider_visible=False,margin=dict(l=10,r=80,t=10,b=10),
        legend=dict(bgcolor="rgba(13,18,25,0.8)",bordercolor="#1e2d40",borderwidth=1,x=0.01,y=0.99),
        hovermode="x unified",
        modebar_remove=["pan","lasso2d","select2d","autoScale2d","hoverCompareCartesian","hoverClosestCartesian","toggleSpikelines","zoomIn2d","zoomOut2d"])
    fig.update_xaxes(gridcolor="#1e2d40",tickformat=tick_format,nticks=8,showspikes=True,spikecolor="#1e2d40",spikedash="solid",spikethickness=1)
    fig.update_yaxes(gridcolor="#1e2d40",tickformat="$.2f",dtick=tick_interval,range=[price_min,price_max],showspikes=True,spikecolor="#1e2d40",row=1,col=1)
    fig.update_yaxes(gridcolor="#1e2d40",tickformat=".2s",row=2,col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<div class='section-title'>BACKTEST</div>", unsafe_allow_html=True)
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Analyzing..."):
            report, equity = run_backtest(df, selected_ticker)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Win Rate",f"{report.win_rate}%"); c2.metric("Trades",report.total_trades)
        c3.metric("Avg R:R",f"{report.avg_rr}x");    c4.metric("Expectancy",f"{report.expectancy}R")
        if len(equity) > 1:
            fig_eq = go.Figure(go.Scatter(y=equity,mode="lines+markers",line=dict(color="#00d4aa",width=2),fill="tozeroy",fillcolor="rgba(0,212,170,0.1)"))
            fig_eq.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0d1219",font=dict(color="#e0e6f0"),height=280,title="Equity Curve",margin=dict(l=0,r=0,t=40,b=0))
            fig_eq.update_xaxes(gridcolor="#1e2d40"); fig_eq.update_yaxes(gridcolor="#1e2d40")
            st.plotly_chart(fig_eq, use_container_width=True)
        if report.trades:
            st.dataframe(pd.DataFrame([{"Pattern":t.pattern,"Result":t.outcome.upper(),"Entry":f"${t.entry_price:.2f}","Exit":f"${t.exit_price:.2f}","P&L":f"{t.pnl_pct:+.1f}%"} for t in report.trades]),use_container_width=True)
    else:
        st.info("Click Run Backtest to analyze this ticker.")

with tab4:
    st.markdown("<div class='section-title'>WATCHLIST SCAN</div>", unsafe_allow_html=True)
    if st.button("SCAN ALL TICKERS", type="primary"):
        results = []
        prog = st.progress(0); status = st.empty()
        for i, ticker in enumerate(WATCHLIST):
            status.text(f"Scanning {ticker}...")
            prog.progress((i+1)/len(WATCHLIST))
            try:
                tdf   = fetch_ohlcv(ticker, tf_mult, tf_span, tf_days)
                price = fetch_current_price(ticker) or float(tdf["close"].iloc[-1])
                cands = build_candidates(tdf, ticker, toggles, account_size, risk_pct, dte, trade_style=trade_style, atr=atr)
                tiv,_ = fetch_iv_rank(ticker)
                if cands:
                    best = cands[0]
                    opt  = calc_trade(best["entry"],best["stop"],best["target"],best["direction"],dte,account_size,risk_pct,price,atr=atr)
                    _,gp,elevate = run_seven_point_gate(tdf,best,opt,tiv,check_earnings(ticker),opt["actual_dte"])
                    results.append({"Ticker":ticker,"Price":f"${price:.2f}",
                        "IV Rank":f"{tiv}%" if tiv else "N/A",
                        "Signal":"CALL" if best["direction"]=="bullish" else "PUT",
                        "Pattern":best["pattern_label"],"Conf":f"{best['confidence']}%",
                        "Gate":f"{gp}/7","Status":"PRIME" if elevate else ("WATCH" if best['confidence']>=60 else "WEAK")})
            except: pass
        prog.empty(); status.empty()
        if results:
            results.sort(key=lambda x:(x["Status"]=="PRIME",int(x["Conf"].replace("%",""))),reverse=True)
            st.success(f"Found {len(results)} signals")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No signals right now.")

with tab5:
    st.markdown("<div class='section-title'>TRADE JOURNAL</div>", unsafe_allow_html=True)
    journal = load_journal()
    stats   = get_journal_stats()
    if stats:
        st.markdown("**Performance by Ticker**")
        stat_rows = []
        for t, v in sorted(stats.items(), key=lambda x:x[1]["total"], reverse=True):
            wr = f"{int(v['wins']/(v['wins']+v['losses'])*100)}%" if (v['wins']+v['losses'])>0 else "N/A"
            stat_rows.append({"Ticker":t,"Signals":v["total"],"Calls":v["calls"],"Puts":v["puts"],"Open":v["open"],"Wins":v["wins"],"Losses":v["losses"],"Win Rate":wr})
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)
        st.markdown("<hr style='border-color:#1e2d40;margin:16px 0'>", unsafe_allow_html=True)
    if not journal:
        st.info("No trades logged yet. Click 'Log to Journal' on any signal to start tracking.")
    else:
        st.markdown("**Update a Trade Result**")
        uc1,uc2,uc3 = st.columns(3)
        with uc1: update_idx = st.number_input("Trade # (newest=1)", min_value=1, max_value=len(journal), value=1)
        with uc2: result_choice = st.selectbox("Result", ["Open","Win - Full","Win - Partial","Loss","Expired Worthless"])
        with uc3: pnl_input = st.text_input("P&L $ (optional)", "")
        if st.button("Update Result"):
            idx = len(journal) - update_idx
            journal[idx]["Result"] = result_choice
            if pnl_input: journal[idx]["P&L $"] = pnl_input
            st.session_state.trade_journal = journal
            st.success(f"Updated trade #{update_idx}")
            st.rerun()
        st.markdown("<hr style='border-color:#1e2d40;margin:12px 0'>", unsafe_allow_html=True)
        df_journal = pd.DataFrame(journal[::-1])
        st.dataframe(df_journal, use_container_width=True)
        col_a,col_b,col_c = st.columns(3)
        with col_a:
            if st.button("Clear Journal"):
                st.session_state.trade_journal = []
                st.rerun()
        with col_b:
            csv = df_journal.to_csv(index=False)
            st.download_button("Export Journal CSV", data=csv, file_name=f"journal_{datetime.now().strftime('%m%d%Y')}.csv", mime="text/csv")
        with col_c:
            total  = len(journal)
            wins   = sum(1 for t in journal if "Win" in t.get("Result",""))
            losses = sum(1 for t in journal if "Loss" in t.get("Result",""))
            if wins+losses > 0: st.metric("Live Win Rate", f"{int(wins/(wins+losses)*100)}%", f"{wins}W / {losses}L")

st.markdown("<div style='text-align:center;padding:20px;color:#8899aa;font-size:0.75rem;border-top:1px solid #1e2d40;margin-top:20px'>OPTIONS SCREENER v6.0 - NOT FINANCIAL ADVICE</div>", unsafe_allow_html=True)
