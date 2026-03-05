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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Full scan universe - 120 most liquid options tickers across all sectors
SCAN_UNIVERSE = [
    # Mega cap / Index
    "SPY","QQQ","IWM","DIA","AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B",
    # Tech / Semis
    "AMD","INTC","AVGO","QCOM","MU","AMAT","LRCX","KLAC","MRVL","CRDO","SMCI","ARM","TSM",
    "PLTR","SNOW","DDOG","NET","CRWD","ZS","PANW","FTNT","OKTA","S","SQ","COIN","HOOD",
    "NBIS","VRT","AAOI","ASTS","ZETA","RDW","IREN","WDC",
    # Large cap growth
    "NFLX","UBER","LYFT","ABNB","SHOP","MELI","SE","GRAB","BABA","JD","PDD",
    "RBLX","U","TTWO","EA","ATVI",
    # Financials
    "JPM","BAC","GS","MS","C","WFC","BLK","V","MA","PYPL","AXP",
    # Healthcare / Biotech
    "UNH","JNJ","PFE","MRNA","BNTX","ABBV","LLY","BMY","GILD","REGN","BIIB",
    # Energy
    "XOM","CVX","OXY","SLB","HAL","MPC","PSX",
    # Consumer
    "AMZN","WMT","TGT","COST","HD","LOW","NKE","LULU","MCD","SBUX","CMG",
    # Industrial / EV
    "GE","CAT","DE","BA","LMT","RTX","RIVN","LCID","F","GM",
    # ETF sectors
    "XLK","XLF","XLE","XLV","XLY","XLI","GLD","SLV","TLT","HYG",
]
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))  # deduplicate

# Sector ETF map for sector alignment check
SECTOR_ETF = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AMD":"XLK","INTC":"XLK","AVGO":"XLK",
    "QCOM":"XLK","MU":"XLK","AMAT":"XLK","LRCX":"XLK","KLAC":"XLK","MRVL":"XLK",
    "CRDO":"XLK","SMCI":"XLK","ARM":"XLK","TSM":"XLK","WDC":"XLK",
    "PLTR":"XLK","SNOW":"XLK","DDOG":"XLK","NET":"XLK","CRWD":"XLK","ZS":"XLK",
    "PANW":"XLK","FTNT":"XLK","OKTA":"XLK","S":"XLK","NBIS":"XLK","VRT":"XLK",
    "AAOI":"XLK","ASTS":"XLK","ZETA":"XLK","IREN":"XLK",
    "SQ":"XLF","COIN":"XLF","HOOD":"XLF","PYPL":"XLF","V":"XLF","MA":"XLF",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","C":"XLF","WFC":"XLF",
    "BLK":"XLF","AXP":"XLF",
    "UNH":"XLV","JNJ":"XLV","PFE":"XLV","MRNA":"XLV","BNTX":"XLV","ABBV":"XLV",
    "LLY":"XLV","BMY":"XLV","GILD":"XLV","REGN":"XLV","BIIB":"XLV",
    "XOM":"XLE","CVX":"XLE","OXY":"XLE","SLB":"XLE","HAL":"XLE","MPC":"XLE","PSX":"XLE",
    "WMT":"XLY","TGT":"XLY","COST":"XLY","HD":"XLY","LOW":"XLY","NKE":"XLY",
    "LULU":"XLY","MCD":"XLY","SBUX":"XLY","CMG":"XLY","AMZN":"XLY",
    "NFLX":"XLY","UBER":"XLY","LYFT":"XLY","ABNB":"XLY","RBLX":"XLY",
    "GE":"XLI","CAT":"XLI","DE":"XLI","BA":"XLI","LMT":"XLI","RTX":"XLI",
    "RIVN":"XLI","LCID":"XLI","F":"XLI","GM":"XLI",
}
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

def calc_trade(entry, stop, target, direction, days_to_exp, account, risk_pct, current_price, iv=0.45, atr=None, trade_style="swing"):
    import math as _math
    # Guard against NaN/None prices (market closed, no data)
    def _clean(v, fallback=0.0):
        try:
            f = float(v)
            return fallback if (_math.isnan(f) or _math.isinf(f)) else f
        except (TypeError, ValueError):
            return fallback

    current_price = _clean(current_price, 100.0)
    entry  = _clean(entry,  current_price)
    stop   = _clean(stop,   current_price * 0.97)
    target = _clean(target, current_price * 1.05)

    is_call    = direction == "bullish"
    exp_date   = get_expiration_date(days_to_exp)
    actual_dte = max((exp_date - date.today()).days, 1)

    # Strike selection: Quick = ATM (best for fast moves), Swing = 2% OTM (leveraged)
    if trade_style == "quick":
        raw_strike = current_price  # ATM for quick trades
    else:
        raw_strike = current_price * 1.02 if is_call else current_price * 0.98
    strike  = round(raw_strike / 0.5) * 0.5

    # IV adjustment: quick trades use higher IV estimate (short-dated premiums are inflated)
    iv_adj  = min(iv * 1.3, 0.80) if actual_dte <= 7 else iv
    premium = round(current_price * iv_adj * (max(actual_dte, 1)/365)**0.5 * 0.4, 2)
    premium = max(premium, 0.05)
    breakeven = (strike + premium) if is_call else (strike - premium)

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

    # Option profit estimate using delta approximation
    # This reflects how the option trades mid-contract, not at expiry
    # Delta * stock move = option price change (first order approximation)
    # We also add a small gamma component for larger moves
    stock_move = abs(stock_target - current_price)
    # Estimate option value gain: delta * move + 0.5 * gamma * move^2
    # Gamma approximation: ~delta*(1-delta)/stock_price/sqrt(actual_dte/365)
    gamma_est  = (abs_delta * (1 - abs_delta)) / max(current_price * (actual_dte/365)**0.5, 1)
    option_gain_per_share = abs_delta * stock_move + 0.5 * gamma_est * (stock_move ** 2)
    # Cap gain at 10x premium (realistic for options)
    option_gain_per_share = min(option_gain_per_share, premium * 10)
    profit_per   = round(option_gain_per_share * 100, 2)
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

def init_auto_scan():
    if "auto_scan_enabled"  not in st.session_state: st.session_state.auto_scan_enabled  = False
    if "auto_scan_results"  not in st.session_state: st.session_state.auto_scan_results   = None
    if "auto_scan_last_run" not in st.session_state: st.session_state.auto_scan_last_run  = None
    if "auto_scan_go_now"   not in st.session_state: st.session_state.auto_scan_go_now    = []
    if "auto_scan_prev_go"  not in st.session_state: st.session_state.auto_scan_prev_go   = []
    if "auto_scan_watching" not in st.session_state: st.session_state.auto_scan_watching  = []
    if "auto_scan_on_deck"  not in st.session_state: st.session_state.auto_scan_on_deck   = []
    if "auto_scan_mkt"      not in st.session_state: st.session_state.auto_scan_mkt       = "neutral"
    if "auto_scan_settings" not in st.session_state: st.session_state.auto_scan_settings  = {
        "scan_list": "watchlist", "max_premium": 5.0, "style": "both"
    }

init_auto_scan()

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
    Confidence scoring - base layer (50 pts max).
    Final score = base + TF confluence + extra confluence = 50-100.
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

    rsi_div = detect_rsi_divergence(df)
    rsi_div_match = rsi_div is not None and (
        (is_bull and rsi_div.get("type") == "bullish") or
        (not is_bull and rsi_div.get("type") == "bearish")
    )
    vol_expanding = cur_vol > avg_vol * 1.3
    vol_present   = cur_vol > avg_vol * 1.1

    factors = {
        "Pattern":{"pass":True,          "label":"Pattern confirmed"},
        "RSI Div":{"pass":rsi_div_match, "label":f"RSI divergence {'confirmed' if rsi_div_match else 'not detected'}"},
        "Volume": {"pass":vol_expanding, "label":f"Volume {'spike' if vol_expanding else 'expanding' if vol_present else 'weak'} ({cur_vol/1e6:.1f}M vs avg {avg_vol/1e6:.1f}M)"},
        "EMA":    {"pass":(price>ema20 if is_bull else price<ema20),"label":f"Price {'above' if is_bull else 'below'} EMA 20 (${ema20:.2f})"},
        "VWAP":   {"pass":(price>vwap  if is_bull else price<vwap), "label":f"Price {'above' if is_bull else 'below'} VWAP (${vwap:.2f})"},
    }

    # Base score: each factor = 10pts, max 50
    raw_score  = sum(1 for f in factors.values() if f["pass"])
    base_score = raw_score * 10  # 0-50

    return factors, raw_score, base_score, rsi, vwap, ema20

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


def fetch_multi_tf(ticker, trade_style):
    """
    Fetches the correct timeframes automatically based on trade style.
    Quick: 5min (primary) + 15min (confirmation)
    Swing: 1hr (primary) + 4hr (confirmation) + Daily (trend anchor)
    Returns dict of {label: df}
    """
    import yfinance as yf

    @st.cache_data(ttl=60)
    def _fetch(ticker, interval, period):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty: return None
            df = df.reset_index()
            df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
            df = df.rename(columns={"datetime":"timestamp","date":"timestamp"})
            return df[["timestamp","open","high","low","close","volume"]].dropna().reset_index(drop=True)
        except:
            return None

    if trade_style == "quick":
        tf5  = _fetch(ticker, "5m",  "2d")
        tf15 = _fetch(ticker, "15m", "5d")
        return {
            "5min":  tf5  if tf5  is not None and len(tf5)  > 20 else None,
            "15min": tf15 if tf15 is not None and len(tf15) > 20 else None,
        }
    else:
        tf1h  = _fetch(ticker, "1h",  "14d")
        tf4h  = _fetch(ticker, "1h",  "30d")   # yfinance max 4h is limited, use 1h proxy
        tf1d  = _fetch(ticker, "1d",  "90d")
        return {
            "1hr":   tf1h if tf1h  is not None and len(tf1h)  > 20 else None,
            "4hr":   tf4h if tf4h  is not None and len(tf4h)  > 40 else None,
            "daily": tf1d if tf1d  is not None and len(tf1d)  > 20 else None,
        }

def check_vwap_confluence(df_5min, direction):
    """
    Quick trade extra confluence: VWAP reclaim/rejection on 5min.
    For calls: previous candle closed BELOW vwap, current candle closes ABOVE = actual reclaim
    For puts:  previous candle closed ABOVE vwap, current candle closes BELOW = actual rejection
    This ensures price is actively crossing VWAP in the signal direction,
    not just sitting above/below it.
    Returns: (passes: bool, label: str)
    """
    if df_5min is None or len(df_5min) < 5:
        return False, "5min data unavailable"
    close = df_5min["close"].astype(float)
    high  = df_5min["high"].astype(float)
    low   = df_5min["low"].astype(float)
    vol   = df_5min["volume"].astype(float)
    tp    = (high + low + close) / 3
    vwap  = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])
    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    is_bull = direction == "bullish"
    if is_bull:
        # Actual reclaim: prev closed below, current closed above
        reclaim = prev < vwap and price > vwap
        # Also accept: holding above VWAP with prev also above (momentum continuation)
        holding = prev > vwap and price > vwap
        passes  = reclaim or holding
        if reclaim:
            label = f"5min VWAP reclaimed ↑ (${vwap:.2f}) — strong"
        elif holding:
            label = f"5min holding above VWAP (${vwap:.2f})"
        else:
            label = f"5min below VWAP (${vwap:.2f}) — no reclaim yet"
    else:
        # Actual rejection: prev closed above, current closed below
        rejection = prev > vwap and price < vwap
        holding   = prev < vwap and price < vwap
        passes    = rejection or holding
        if rejection:
            label = f"5min VWAP rejected ↓ (${vwap:.2f}) — strong"
        elif holding:
            label = f"5min holding below VWAP (${vwap:.2f})"
        else:
            label = f"5min above VWAP (${vwap:.2f}) — no rejection yet"
    return passes, label

def check_ema50_slope(df_daily, direction):
    """
    Swing trade extra confluence: 50 EMA slope on Daily.
    Slope = (current EMA50 - EMA50 5 bars ago) / EMA50 5 bars ago
    Calls need rising slope, Puts need falling slope.
    Returns: (passes: bool, label: str)
    """
    if df_daily is None or len(df_daily) < 55:
        return False, "Daily data unavailable for EMA50"
    close  = df_daily["close"].astype(float)
    ema50  = close.ewm(span=50).mean()
    current = float(ema50.iloc[-1])
    prior   = float(ema50.iloc[-6])
    slope_pct = (current - prior) / prior * 100
    is_bull = direction == "bullish"
    if is_bull:
        passes = slope_pct > 0
        label  = f"Daily EMA50 rising (+{slope_pct:.2f}%)" if passes else f"Daily EMA50 falling ({slope_pct:.2f}%)"
    else:
        passes = slope_pct < 0
        label  = f"Daily EMA50 falling ({slope_pct:.2f}%)" if passes else f"Daily EMA50 rising (+{slope_pct:.2f}%)"
    return passes, label

def check_tf_trend_agreement(dfs, direction):
    """
    Checks how many timeframes agree with the signal direction.
    Returns (agreeing_count, total_checked, details_list)
    """
    details = []
    agreeing = 0
    for label, df in dfs.items():
        if df is None: continue
        close = df["close"].astype(float)
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        price = float(close.iloc[-1])
        trend = "bullish" if price > ema20 else "bearish"
        agrees = trend == direction
        if agrees: agreeing += 1
        details.append({
            "tf": label,
            "trend": trend,
            "agrees": agrees,
            "ema20": round(ema20, 2),
            "price": round(price, 2),
        })
    return agreeing, len(details), details

def build_multi_tf_candidates(ticker, toggles, account, risk_pct,
                               dte, trade_style, atr=None):
    """
    Automatically fetches the right timeframes and builds candidates
    with multi-TF confluence baked in.
    Quick:  15min primary pattern + 5min trend + 5min VWAP
    Swing:  1hr primary pattern + 4hr trend + Daily EMA50 slope
    """
    tfs = fetch_multi_tf(ticker, trade_style)

    # Pick primary df for pattern detection
    if trade_style == "quick":
        _15m = tfs.get("15min"); _5m = tfs.get("5min")
        primary_df = _15m if _15m is not None else _5m
        confirm_df = _5m
    else:
        _1h  = tfs.get("1hr"); _4h = tfs.get("4hr"); _1d = tfs.get("daily")
        primary_df = _1h
        confirm_df = _4h if _4h is not None else _1d
        daily_df   = _1d

    if primary_df is None:
        return [], tfs   # no data

    # Build candidates using primary timeframe
    cands = build_candidates(primary_df, ticker, toggles, account, risk_pct,
                             dte, trade_style=trade_style, atr=atr)

    # Enhance each candidate with multi-TF confluence
    for c in cands:
        direction = c["direction"]
        tf_agreement, tf_total, tf_details = check_tf_trend_agreement(tfs, direction)

        if trade_style == "quick":
            # Extra confluence: 5min VWAP
            extra_pass, extra_label = check_vwap_confluence(confirm_df, direction)
            extra_name = "5min VWAP"
        else:
            # Extra confluence: Daily EMA50 slope
            extra_pass, extra_label = check_ema50_slope(
                tfs.get("daily"), direction)
            extra_name = "Daily EMA50"

        # ── Clean 50-100 final score ─────────────────────────────────────────────
        # Base (from score_setup): 0-50 pts  (each of 5 factors = 10 pts)
        # TF confluence:           0-30 pts  (each agreeing TF = 10 pts, max 3 TFs = 30)
        # Extra confluence:        0-20 pts  (VWAP reclaim or EMA50 slope)
        # Total max = 100, min shown = 50
        base = c["confidence"]  # already 50-95 from build_candidates

        # TF layer: up to 30 pts from agreeing timeframes
        if tf_total > 0:
            tf_pts = int((tf_agreement / tf_total) * 30)
        else:
            tf_pts = 15  # neutral if no TF data

        # Extra confluence layer: 20 pts if passes, 0 if not
        extra_pts = 20 if extra_pass else 0

        # Combine and clamp to 50-100
        raw_final = base + tf_pts + extra_pts
        # Normalize so max possible (50+30+20=100) maps cleanly
        # But base is already 50-95, so we need to scale down
        # Simpler: score = 50 + (factors/5)*25 + (tfs/total)*15 + extra*10
        factor_pts = c.get("score", 0)  # 0-5 raw factors passing
        score_50_100 = (
            50
            + int((factor_pts / 5) * 25)
            + (int((tf_agreement / tf_total) * 15) if tf_total > 0 else 8)
            + (10 if extra_pass else 0)
        )
        c["confidence"] = min(100, max(50, score_50_100))
        c["tf_details"]   = tf_details
        c["tf_agreement"] = tf_agreement
        c["tf_total"]     = tf_total
        c["extra_confluence"] = {
            "name":  extra_name,
            "pass":  extra_pass,
            "label": extra_label,
        }
        c["primary_tf"] = "15min" if trade_style == "quick" else "1hr"
        c["confirm_tfs"] = list(tfs.keys())

    return cands, tfs

def build_candidates(df, ticker, toggles, account, risk_pct, dte, trade_style="swing", atr=None):
    trend_dir,trend_score,trend_factors,t_ema,t_vwap,t_rsi = get_trend(df)
    price   = float(df["close"].iloc[-1])
    regime, regime_strength = detect_market_regime(df)
    is_quick = trade_style == "quick"
    # Define regime_bonus once here so it's always available even if no patterns found
    regime_bonus = 5 if regime == "trending" else -5
    candidates = []
    raw = []
    if toggles["db"]: raw += [s for s in detect_double_bottom(df,ticker,rr_min=2.0) if s.confirmed]
    if toggles["dt"]: raw += [s for s in detect_double_top(df,ticker,rr_min=2.0)    if s.confirmed]
    if toggles["br"]: raw += [s for s in detect_break_and_retest(df,ticker,rr_min=2.0) if s.confirmed]

    for setup in raw:
        if abs(setup.entry_price - price) / price > 0.05: continue
        factors, raw_score, weighted_conf, rsi, vwap, ema20 = score_setup(df, setup)
        conflict = setup.direction != trend_dir and trend_score >= 3

        # TF alignment: handled in build_multi_tf_candidates, use base score here
        final_conf = max(50, min(95, 50 + weighted_conf))

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
            conf_val = max(50, min(90, 50 + int(trend_score/5*50)))
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
        trend_conf = max(50, min(90, 50 + int(trend_score/5*50)))
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

def render_signal_cards(candidates, ticker, dte, trade_style, key_prefix,
                        df, current_price, atr, iv_rank, earnings_days,
                        mstatus, mtext, account_size, risk_pct,
                        htf_trend, htf_rsi, htf_ema, liq_ok):
    """Renders signal cards for a given candidate list. Called once per column."""
    if not candidates:
        st.markdown(
            "<div style='background:#111827;border:1px solid #1e2d40;border-radius:12px;"
            "padding:20px;text-align:center;color:#8899aa;font-size:0.85rem'>"
            "No signals found for this mode.</div>", unsafe_allow_html=True)
        return

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
        sig_style = trade_style  # passed as parameter
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

        # Multi-TF confluence rows
        tf_details = sig.get("tf_details", [])
        extra_conf = sig.get("extra_confluence", {})
        tf_html = ""
        if tf_details:
            tf_html += "<div style='margin-top:8px;padding-top:8px;border-top:1px solid #1e2d40'>"
            tf_html += "<div style='color:#8899aa;font-family:monospace;font-size:0.68rem;letter-spacing:1px;margin-bottom:4px'>TIMEFRAME CONFLUENCE</div>"
            for td in tf_details:
                dot = "dot-green" if td["agrees"] else "dot-red"
                c_color = "#e0e6f0" if td["agrees"] else "#8899aa"
                tf_html += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + c_color + ";font-size:0.78rem'><b>" + td["tf"].upper() + ":</b> " + td["trend"].upper() + " (EMA20 $" + str(td["ema20"]) + ")</span></div>"
            if extra_conf:
                dot = "dot-green" if extra_conf.get("pass") else "dot-yellow"
                c_color = "#e0e6f0" if extra_conf.get("pass") else "#8899aa"
                tf_html += "<div class='factor-row'><span class='" + dot + "'></span><span style='color:" + c_color + ";font-size:0.78rem'><b>" + str(extra_conf.get("name","")) + ":</b> " + str(extra_conf.get("label","")) + "</span></div>"
            tf_html += "</div>"

        st.markdown(f"""
        {conflict_html}
        {quick_warn_html}
        <div class='{rc}'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                    <span class='rank-badge {bc}'>{ri} {rl}</span>{style_badge}{liq_warn}
                    <div style='font-size:1.1rem;font-weight:700;color:{dir_color};margin-top:4px'>{dir_label} - {ticker}</div>
                    <div style='color:#8899aa;font-size:0.82rem;margin-top:2px'>{sig['pattern_label']} &nbsp;<span style='font-size:0.75rem'>{regime_icon} {regime_label}</span></div>
                </div>
                <div style='text-align:right'>
                    <div class='{cc}'>{sig['confidence']}%</div>
                    <div style='font-size:0.7rem;font-family:monospace;margin-top:2px;color:#8899aa'>{"GO" if sig['confidence']>=90 else "STRONG" if sig['confidence']>=80 else "WATCH" if sig['confidence']>=70 else "WEAK" if sig['confidence']>=60 else "WAIT"}</div>
                </div>
            </div>
            <div style='margin-top:10px'>{dots_html}{tf_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if sig["confidence"] >= 60:
            opt = calc_trade(sig["entry"],sig["stop"],sig["target"],sig["direction"],dte,account_size,risk_pct,current_price,atr=atr,trade_style=trade_style)
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
                <div style='margin:4px 0'><b>Close 100%</b> if {ticker} closes {'below' if is_bull else 'above'} <b style='color:#ff4d6d'>${opt['exit_stop_stock']:.2f}</b> - pattern failed, no questions asked.</div>
                <div style='margin:4px 0;color:#8899aa;font-size:0.8rem'>{exit_hold}</div>
            </div>
            """, unsafe_allow_html=True)

            # Entry timing check + watch queue - market hours only
            watch_key = f"{ticker}_{sig['direction']}"
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

                # Auto-add to watch queue: 7/7 elevated auto-adds, 5/7+ shows Watch button prominently
                if elevate and not already_watching and conf_status != "CONFIRMED":
                    add_to_watch_queue(ticker, sig["direction"], sig, opt)
                elif gates_passed >= 5 and not already_watching and conf_status != "CONFIRMED":
                    # 5+ gates - don't auto-add but make the Watch button obvious
                    st.markdown(f"<div style='background:#0d1219;border:1px solid #f0c040;border-radius:6px;padding:6px 12px;margin-top:4px;color:#f0c040;font-size:0.8rem'>⚡ {gates_passed}/7 gates — strong setup. Hit Watch to track entry timing.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background:#0d1219;border:1px solid #1e2d40;border-radius:8px;padding:10px 14px;margin-top:8px;color:#8899aa;font-size:0.82rem'>⏸ Entry timing check runs during market hours only (9:30 AM - 4:00 PM ET)</div>", unsafe_allow_html=True)

            if mstatus == "open":
                if ANTHROPIC_API_KEY:
                    ai_key = f"ai_result_{ticker}_{i}"
                    if st.button(f"🤖 Get AI Brief #{i+1}", key=f"{key_prefix}_ai_{i}"):
                        with st.spinner("Analyzing setup..."):
                            try:
                                ai_text   = get_ai_brief(ticker, sig, opt, gates, gates_passed, iv_rank, earnings_days, conf_status)
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
                if st.button(f"Log to Journal #{i+1}", key=f"{key_prefix}_log_{i}"):
                    log_trade(ticker, sig, opt, gates_passed, 7, elevate)
                    st.success("Logged!")
            with bcol2:
                share_text = build_share_text(ticker,sig,opt,gates_passed,7,elevate,mtext)
                st.download_button(f"Share #{i+1}", data=share_text,
                    file_name=f"{ticker}_signal_{datetime.now().strftime('%m%d_%H%M')}.txt",
                    mime="text/plain", key=f"{key_prefix}_share_{i}")
            with bcol3:
                if not already_watching:
                    if st.button(f"Watch #{i+1}", key=f"{key_prefix}_watch_{i}"):
                        add_to_watch_queue(ticker, sig["direction"], sig, opt)
                        st.success("Added to watch queue!")
                        st.rerun()
                else:
                    if st.button(f"Stop Watching #{i+1}", key=f"{key_prefix}_unwatch_{i}"):
                        remove_from_watch_queue(watch_key)
                        st.rerun()

        if i < len(candidates) - 1:
            st.markdown("<hr style='border-color:#1e2d40;margin:12px 0'>", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION SCAN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_market_internals():
    """
    Checks SPY and QQQ trend to determine overall market bias.
    Returns: bias ("bullish"/"bearish"/"neutral"), strength (0-100)
    """
    try:
        import yfinance as yf
        results = {}
        for sym in ["SPY","QQQ"]:
            df = yf.download(sym, period="5d", interval="15m", progress=False, auto_adjust=True)
            if df.empty: continue
            df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
            close = df["close"].astype(float)
            ema20 = float(close.ewm(span=20).mean().iloc[-1])
            ema50 = float(close.ewm(span=50).mean().iloc[-1])
            price = float(close.iloc[-1])
            rsi   = calc_rsi(close)
            results[sym] = {
                "above_ema20": price > ema20,
                "above_ema50": price > ema50,
                "rsi": rsi,
                "price": price,
                "ema20": round(ema20,2),
            }
        if not results: return "neutral", 50

        bull_signals = sum([
            results.get("SPY",{}).get("above_ema20", False),
            results.get("SPY",{}).get("above_ema50", False),
            results.get("QQQ",{}).get("above_ema20", False),
            results.get("QQQ",{}).get("above_ema50", False),
            results.get("SPY",{}).get("rsi",50) > 50,
            results.get("QQQ",{}).get("rsi",50) > 50,
        ])
        bear_signals = 6 - bull_signals

        if bull_signals >= 5:   return "bullish", int(bull_signals/6*100)
        elif bear_signals >= 5: return "bearish", int(bear_signals/6*100)
        else:                   return "neutral",  50
    except:
        return "neutral", 50

@st.cache_data(ttl=300)
def get_sector_bias(sector_etf):
    """Returns trend direction of a sector ETF."""
    try:
        import yfinance as yf
        df = yf.download(sector_etf, period="5d", interval="1h", progress=False, auto_adjust=True)
        if df.empty: return "neutral"
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
        close = df["close"].astype(float)
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        return "bullish" if price > ema20 else "bearish"
    except:
        return "neutral"

def detect_exhaustion(df, direction):
    """
    Elite exhaustion detection.
    ONE strong signal is enough to confirm — we score 0-4 but only need 1/4.
    Confirmed = any single strong signal present.
    Score used for confidence weighting.
    """
    if len(df) < 20:
        return False, 0, ["Insufficient data"]

    close   = df["close"].astype(float)
    high    = df["high"].astype(float)
    low     = df["low"].astype(float)
    open_   = df["open"].astype(float)
    volume  = df["volume"].astype(float)
    avg_vol = float(volume.iloc[-20:].mean())
    is_bull = direction == "bullish"
    reasons = []
    score   = 0

    # 1. Exhaustion candle — big body + high volume in last 6 bars
    for j in range(-6, 0):
        body   = float(open_.iloc[j]) - float(close.iloc[j]) if is_bull else float(close.iloc[j]) - float(open_.iloc[j])
        rng    = float(high.iloc[j]) - float(low.iloc[j])
        is_big = rng > 0 and body / rng > 0.55
        is_vol = float(volume.iloc[j]) > avg_vol * 1.4
        if body > 0 and is_big and is_vol:
            score += 1
            reasons.append("Capitulation candle confirmed" if is_bull else "Climax candle confirmed")
            break
    else:
        reasons.append("No climax/capitulation candle")

    # 2. Reversal candle — hammer/doji for calls, shooting star/doji for puts
    last_body  = abs(float(close.iloc[-1]) - float(open_.iloc[-1]))
    last_range = float(high.iloc[-1]) - float(low.iloc[-1])
    is_doji    = last_range > 0 and last_body / last_range < 0.3
    if is_bull:
        lower_wick = min(float(open_.iloc[-1]), float(close.iloc[-1])) - float(low.iloc[-1])
        is_hammer  = last_range > 0 and lower_wick / last_range > 0.45
        if is_hammer or is_doji:
            score += 1
            reasons.append("Hammer/doji reversal candle")
        else:
            reasons.append("No reversal candle yet")
    else:
        upper_wick = float(high.iloc[-1]) - max(float(open_.iloc[-1]), float(close.iloc[-1]))
        is_star    = last_range > 0 and upper_wick / last_range > 0.45
        if is_star or is_doji:
            score += 1
            reasons.append("Shooting star/doji reversal candle")
        else:
            reasons.append("No reversal candle yet")

    # 3. RSI divergence
    div = detect_rsi_divergence(df)
    if div and ((is_bull and div.get("type") == "bullish") or (not is_bull and div.get("type") == "bearish")):
        score += 1
        reasons.append("RSI divergence confirmed")
    else:
        reasons.append("No RSI divergence")

    # 4. Structure — higher low for calls, lower high for puts
    if is_bull:
        lows = [float(low.iloc[i]) for i in [-15, -8, -1]]
        if lows[-1] > lows[-2]:
            score += 1
            reasons.append("Higher low structure forming")
        else:
            reasons.append("Lower low — structure not confirmed")
    else:
        highs = [float(high.iloc[i]) for i in [-15, -8, -1]]
        if highs[-1] < highs[-2]:
            score += 1
            reasons.append("Lower high structure forming")
        else:
            reasons.append("Higher high — structure not confirmed")

    # ONE strong signal is enough — elite traders don't wait for perfect storms
    confirmed = score >= 1
    return confirmed, score, reasons


def precision_score(ticker, direction, df_primary, df_confirm,
                    iv_rank, earnings_days, market_bias,
                    sector_bias, atr, dte, account_size, risk_pct,
                    trade_style):
    """
    Elite scoring framework.

    TIER 1 — Hard stops (any single fail = no trade):
      - Earnings within 5 days
      - IV rank > 70% (too expensive to buy)
      - Market directly opposing with conviction
      - Time of day: before 9:45am or after 3:30pm ET

    TIER 2 — Quality signals, need 4 of 5:
      - Trend aligned on primary TF
      - Pattern confirmed with volume
      - At least ONE exhaustion signal
      - VWAP relationship clean
      - RSI divergence present

    TIER 3 — Execution quality scoring:
      - ATR confirms move is realistic
      - R:R minimum 2:1
      - Options liquidity present
      - IV rank in sweet spot (20-50%)
      - Market / sector tailwind

    Score: 50 base, up to 100
    """
    import pytz
    from datetime import datetime as _dt

    # ── TIER 1: Hard stops ────────────────────────────────────────────────────
    if earnings_days is not None and earnings_days <= 5:
        return None, "Earnings within 5 days"

    if iv_rank is not None and iv_rank > 70:
        return None, f"IV too high ({iv_rank}%)"

    if market_bias == "bullish" and direction == "bearish":
        return None, "Market strongly bullish — no puts"
    if market_bias == "bearish" and direction == "bullish":
        return None, "Market strongly bearish — no calls"

    # Time of day hard stop (scan only during quality hours)
    try:
        et  = pytz.timezone("America/New_York")
        now = _dt.now(et).time()
        from datetime import time as _t
        if now < _t(9, 45) or now > _t(15, 30):
            return None, "Outside quality trading hours (9:45-3:30 ET)"
    except Exception:
        pass  # outside market hours scanning is fine for setup detection

    # ── TIER 2: Quality signals — need 4 of 5 ────────────────────────────────
    signals_hit = 0
    signal_detail = []

    # Signal 1: Trend aligned on primary TF
    try:
        trend_dir, trend_score, _, _, _, _ = get_trend(df_primary)
        if trend_dir == direction:
            signals_hit += 1
            signal_detail.append("✅ Trend aligned")
        else:
            signal_detail.append("❌ Trend opposing")
    except Exception:
        signal_detail.append("❌ Trend unavailable")

    # Signal 2: Pattern confirmed with volume
    try:
        avg_vol = float(df_primary["volume"].iloc[-20:].mean())
        cur_vol = float(df_primary["volume"].iloc[-3:].mean())
        if cur_vol > avg_vol * 1.1:
            signals_hit += 1
            signal_detail.append("✅ Volume confirming")
        else:
            signal_detail.append("❌ Volume weak")
    except Exception:
        signal_detail.append("❌ Volume unavailable")

    # Signal 3: At least ONE exhaustion signal
    exh_confirmed, exh_score, exh_reasons = detect_exhaustion(df_primary, direction)
    if exh_confirmed:  # now just needs score >= 1
        signals_hit += 1
        signal_detail.append("✅ Exhaustion signal present")
    else:
        signal_detail.append("❌ No exhaustion signal")

    # Signal 4: VWAP relationship
    try:
        vwap_ok, vwap_label = check_vwap_confluence(df_primary, direction)
        if vwap_ok:
            signals_hit += 1
            signal_detail.append(f"✅ VWAP {vwap_label}")
        else:
            signal_detail.append(f"❌ VWAP {vwap_label}")
    except Exception:
        signal_detail.append("❌ VWAP unavailable")

    # Signal 5: RSI divergence
    try:
        div = detect_rsi_divergence(df_primary)
        if div and ((direction == "bullish" and div.get("type") == "bullish") or
                    (direction == "bearish" and div.get("type") == "bearish")):
            signals_hit += 1
            signal_detail.append("✅ RSI divergence")
        else:
            signal_detail.append("❌ No RSI divergence")
    except Exception:
        signal_detail.append("❌ RSI unavailable")

    # Require 3 of 5 minimum (relaxed from 4 — market isn't always perfect)
    # 4 of 5 = strong, 3 of 5 = valid, <3 = skip
    if signals_hit < 3:
        return None, f"Only {signals_hit}/5 quality signals aligned"

    # ── TIER 3: Execution quality scoring ─────────────────────────────────────
    score = 50

    # Signals quality (up to 25 pts)
    score += signals_hit * 5  # 3 signals = +15, 4 = +20, 5 = +25

    # Exhaustion depth (up to 10 pts) — more signals = more conviction
    score += min(exh_score * 3, 10)  # 1 signal = +3, 2 = +6, 3 = +9, 4 = +10

    # TF confirmation — does confirm TF agree? (up to 10 pts)
    tf_agree = 0; tf_total = 0
    if df_confirm is not None:
        tf_total = 1
        try:
            c  = df_confirm["close"].astype(float)
            em = float(c.ewm(span=20).mean().iloc[-1])
            pr = float(c.iloc[-1])
            if (pr > em and direction == "bullish") or (pr < em and direction == "bearish"):
                tf_agree = 1
                score += 10
        except Exception:
            pass

    # Market + sector tailwind (up to 10 pts)
    if market_bias == direction:    score += 6
    elif market_bias == "neutral":  score += 3
    if sector_bias  == direction:   score += 4
    elif sector_bias == "neutral":  score += 2

    # IV rank sweet spot 20-50% (up to 5 pts)
    if iv_rank is not None:
        if 20 <= iv_rank <= 50:  score += 5
        elif 15 <= iv_rank <= 65: score += 2

    # Options liquidity present (up to 5 pts — critical for execution)
    liq_ok, liq_vol, liq_oi, _ = check_liquidity(ticker)
    if liq_ok:
        score += 3 if liq_vol >= 500 else 2 if liq_vol >= 100 else 1

    final = min(100, max(50, score))

    return final, {
        "exhaustion_confirmed": exh_confirmed,
        "exhaustion_score":     exh_score,
        "exhaustion_reasons":   exh_reasons,
        "signals_hit":          signals_hit,
        "signal_detail":        signal_detail,
        "tf_agree":             tf_agree,
        "tf_total":             tf_total,
        "market_bias":          market_bias,
        "sector_bias":          sector_bias,
        "liq_ok":               liq_ok,
        "liq_vol":              liq_vol,
    }

def scan_single_ticker(ticker, toggles, account_size, risk_pct,
                        dte_quick, dte_swing, max_premium,
                        trade_style_filter, market_bias):
    """
    Processes one ticker through the full precision stack.
    Designed to run in a thread pool.
    Returns list of result records (may be empty).
    """
    results = []
    try:
        tfs_q = fetch_multi_tf(ticker, "quick")
        tfs_s = fetch_multi_tf(ticker, "swing")

        _15m = tfs_q.get("15min"); _5m = tfs_q.get("5min")
        _1h  = tfs_s.get("1hr");  _4h = tfs_s.get("4hr"); _1d = tfs_s.get("daily")

        primary_q = _15m if _15m is not None else _5m
        primary_s = _1h

        iv_rank, _ = fetch_iv_rank(ticker)
        earn_days  = check_earnings(ticker)
        price      = fetch_current_price(ticker)
        sector_etf = SECTOR_ETF.get(ticker, "SPY")
        sec_bias   = get_sector_bias(sector_etf)
        atr        = calc_atr(_1d) if _1d is not None else None

        styles = []
        if trade_style_filter in ("quick","both") and primary_q is not None:
            styles.append(("quick", primary_q, _5m, dte_quick))
        if trade_style_filter in ("swing","both") and primary_s is not None:
            styles.append(("swing", primary_s, _4h if _4h is not None else _1d, dte_swing))

        for style, df_pri, df_con, dte in styles:
            if df_pri is None or len(df_pri) < 30: continue
            cur_price = price if price is not None else float(df_pri["close"].iloc[-1])

            cands = build_candidates(df_pri, ticker, toggles,
                                     account_size, risk_pct, dte,
                                     trade_style=style, atr=atr)
            if not cands: continue

            best      = cands[0]
            direction = best["direction"]

            opt = calc_trade(best["entry"], best["stop"], best["target"],
                              direction, dte, account_size, risk_pct,
                              cur_price, atr=atr, trade_style=style)
            if opt["premium"] > max_premium: continue

            conf, detail = precision_score(
                ticker, direction, df_pri, df_con,
                iv_rank, earn_days, market_bias,
                sec_bias, atr, dte, account_size, risk_pct, style
            )
            if conf is None or conf < 60: continue

            gates, gates_passed, elevate = run_seven_point_gate(
                df_pri, best, opt, iv_rank, earn_days, opt["actual_dte"]
            )
            conf_result  = check_entry_confirmation(df_pri, direction)
            entry_status = conf_result["status"]

            # Relative volume spike (institutional signal proxy)
            avg_vol  = float(df_pri["volume"].iloc[-20:].mean()) if len(df_pri) >= 20 else 1
            cur_vol  = float(df_pri["volume"].iloc[-1])
            rel_vol  = round(cur_vol / avg_vol, 1) if avg_vol > 0 else 1.0
            vol_spike = rel_vol >= 1.5  # 1.5x+ average = notable

            # Block trade proxy: large single candles with >2x volume
            block_detected = rel_vol >= 2.5 and abs(
                float(df_pri["close"].iloc[-1]) - float(df_pri["open"].iloc[-1])
            ) > float(df_pri["close"].iloc[-1]) * 0.003

            results.append({
                "ticker":        ticker,
                "style":         style,
                "direction":     direction,
                "action":        "CALL" if direction=="bullish" else "PUT",
                "pattern":       best["pattern_label"],
                "confidence":    conf,
                "gates_passed":  gates_passed,
                "elevate":       elevate,
                "entry_status":  entry_status,
                "opt":           opt,
                "sig":           best,
                "price":         round(cur_price, 2),
                "iv_rank":       iv_rank,
                "earn_days":     earn_days,
                "detail":        detail,
                "market_bias":   market_bias,
                "sector_bias":   sec_bias,
                "exh_confirmed": detail.get("exhaustion_confirmed", False),
                "exh_reasons":   detail.get("exhaustion_reasons", []),
                "signal_detail": detail.get("signal_detail", []),
                "signals_hit":   detail.get("signals_hit", 0),
                "rel_vol":       rel_vol,
                "vol_spike":     vol_spike,
                "block_detected":block_detected,
            })
    except Exception:
        pass
    return results

def full_scan(scan_list, toggles, account_size, risk_pct,
              dte_quick, dte_swing, max_premium, trade_style_filter,
              progress_cb=None):
    """
    Parallel scanner using ThreadPoolExecutor.
    Runs 10 tickers simultaneously — ~10x faster than sequential.
    """
    market_bias, _ = get_market_internals()
    go_now   = []
    watching = []
    on_deck  = []

    completed = 0
    total     = len(scan_list)

    # 10 workers — fast enough without hammering yfinance rate limits
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                scan_single_ticker,
                ticker, toggles, account_size, risk_pct,
                dte_quick, dte_swing, max_premium,
                trade_style_filter, market_bias
            ): ticker
            for ticker in scan_list
        }

        for future in as_completed(futures):
            completed += 1
            ticker = futures[future]
            if progress_cb:
                progress_cb(completed - 1, total, ticker)
            try:
                records = future.result()
                for r in records:
                    conf         = r["confidence"]
                    gates_passed = r["gates_passed"]
                    entry_status = r["entry_status"]
                    exh_ok       = r["exh_confirmed"]

                    signals_hit = r.get("detail", {}).get("signals_hit", 0)
                    exh_score   = r.get("detail", {}).get("exhaustion_score", 0)

                    # GO NOW: confident execution — trend + exhaustion + 4+ signals + entry confirmed
                    if (conf >= 75 and gates_passed >= 5 and
                        entry_status == "CONFIRMED" and exh_ok and signals_hit >= 4):
                        go_now.append(r)
                    # WATCHING: strong setup, waiting for final confirmation
                    elif conf >= 65 and gates_passed >= 4 and signals_hit >= 3:
                        watching.append(r)
                    # ON DECK: forming — worth knowing about
                    elif conf >= 55 and signals_hit >= 3:
                        on_deck.append(r)
            except Exception:
                continue

    go_now.sort(  key=lambda x: (x["vol_spike"], x["confidence"]), reverse=True)
    watching.sort(key=lambda x: (x["vol_spike"], x["confidence"]), reverse=True)
    on_deck.sort( key=lambda x: x["confidence"], reverse=True)

    return go_now, watching, on_deck, market_bias


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## OPTIONS SCREENER v6")
    st.markdown("---")
    selected_ticker = st.selectbox("TICKER", WATCHLIST)
    custom = st.text_input("Or type any ticker","").upper().strip()
    if custom: selected_ticker = custom
    selected_tf = st.selectbox("CHART TIMEFRAME", list(TIMEFRAMES.keys()), index=2)
    st.caption("Signals use automatic timeframes per mode.")
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
    st.markdown("**⚡ Quick DTE** (weekly/0DTE)")
    dte_quick = st.selectbox("Quick expiry", [0,1,2,3,5,7], index=2,
                             help="0 = 0DTE, 1-7 = this week", label_visibility="collapsed")
    st.markdown("**📅 Swing DTE** (multi-week)")
    dte_swing = st.selectbox("Swing expiry", [14,21,30,45,60], index=2,
                             label_visibility="collapsed")
    trade_style = "both"  # always show both
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
        col_banner, col_dismiss = st.columns([6,1])

        if status == "CONFIRMED":
            # Full trade card - boom get in now
            is_bull_w = item["direction"] == "bullish"
            dir_color_w = "#00d4aa" if is_bull_w else "#ff4d6d"
            with col_banner:
                st.markdown(f"""
                <div style='background:#061a10;border:2px solid #00d4aa;border-radius:10px;padding:14px 16px;margin:4px 0;animation:none'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
                        <div>
                            <span style='color:#00d4aa;font-family:monospace;font-size:0.72rem;letter-spacing:2px'>✅ ENTRY CONFIRMED - GET IN NOW</span><br>
                            <span style='font-size:1.1rem;font-weight:700;color:{dir_color_w}'>BUY {"CALL" if is_bull_w else "PUT"} - {item["ticker"]}</span>
                            <span style='color:#8899aa;font-size:0.82rem;margin-left:8px'>{item["pattern"]}</span>
                        </div>
                        <div style='text-align:right;color:#8899aa;font-size:0.75rem;font-family:monospace'>{elapsed_mins}m {elapsed_secs}s{last_chk}</div>
                    </div>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-size:0.85rem;margin-top:6px'>
                        <div><div style='color:#8899aa;font-size:0.7rem'>STRIKE</div><div style='font-weight:700;color:{dir_color_w}'>${item["strike"]:.2f}</div></div>
                        <div><div style='color:#8899aa;font-size:0.7rem'>ENTRY</div><div style='font-weight:700'>${item["entry"]:.2f}</div></div>
                        <div><div style='color:#8899aa;font-size:0.7rem'>TARGET</div><div style='font-weight:700;color:#00d4aa'>${item["target"]:.2f}</div></div>
                        <div><div style='color:#8899aa;font-size:0.7rem'>STOP OUT</div><div style='font-weight:700;color:#ff4d6d'>${item["stop"]:.2f}</div></div>
                    </div>
                    <div style='margin-top:8px;color:#e0e6f0;font-size:0.78rem'>
                        Candles: {candle_html} &nbsp; <span style='color:#00d4aa'>2 confirmation candles printed — execute at market price</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            if status == "WAITING":
                banner_style = "background:#0d1219;border:2px solid #f0c040;border-radius:8px;padding:12px 16px;margin:4px 0;"
                icon = "👁"; stxt = "<span style='color:#f0c040;font-weight:700'>WATCHING</span>"
            else:
                banner_style = "background:#1a0a0a;border:2px solid #ff4d6d;border-radius:8px;padding:12px 16px;margin:4px 0;"
                icon = "⏳"; stxt = "<span style='color:#ff4d6d;font-weight:700'>ENTRY EARLY</span>"
            with col_banner:
                st.markdown(f"""
                <div style='{banner_style}'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div>
                            <span style='font-size:1.1rem'>{icon}</span>
                            <b style='margin-left:6px'>{item['ticker']} {"CALL" if item["direction"]=="bullish" else "PUT"}</b>
                            <span style='color:#8899aa;font-size:0.82rem;margin-left:8px'>{item['pattern']} | Strike ${item['strike']:.2f}</span>
                        </div>
                        <div style='color:#8899aa;font-size:0.75rem;font-family:monospace'>{elapsed_mins}m {elapsed_secs}s{last_chk}</div>
                    </div>
                    <div style='margin-top:6px'>{stxt} &nbsp; {candle_html}</div>
                    <div style='color:#e0e6f0;font-size:0.82rem;margin-top:4px'>{item['message']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_dismiss:
            if st.button("✕", key=f"dismiss_{key}", help="Dismiss"):
                remove_from_watch_queue(key)
                st.rerun()

mstatus, mtext = get_market_status()
css_class = {"open":"market-open","pre":"market-pre","after":"market-pre","closed":"market-closed"}.get(mstatus,"market-closed")
st.markdown(f"<div class='{css_class}'>{mtext}</div>", unsafe_allow_html=True)

# ── AUTO-SCAN ENGINE ──────────────────────────────────────────────────────────
SCAN_INTERVAL = 300  # 5 minutes

def should_run_auto_scan():
    if mstatus != "open": return False
    if not st.session_state.auto_scan_enabled: return False
    last = st.session_state.auto_scan_last_run
    if last is None: return True
    return (datetime.now() - last).total_seconds() >= SCAN_INTERVAL

def run_auto_scan_now():
    cfg = st.session_state.auto_scan_settings
    sl  = WATCHLIST if cfg["scan_list"]=="watchlist" else SCAN_UNIVERSE
    go, watching, deck, mkt = full_scan(
        sl, toggles, account_size, risk_pct,
        dte_quick, dte_swing, cfg["max_premium"], cfg["style"]
    )
    prev_tickers = {(r["ticker"],r["style"]) for r in st.session_state.auto_scan_prev_go}
    new_go = [r for r in go if (r["ticker"],r["style"]) not in prev_tickers]
    st.session_state.auto_scan_prev_go  = st.session_state.auto_scan_go_now
    st.session_state.auto_scan_go_now   = go
    st.session_state.auto_scan_watching = watching
    st.session_state.auto_scan_on_deck  = deck
    st.session_state.auto_scan_mkt      = mkt
    st.session_state.auto_scan_last_run = datetime.now()
    return new_go

new_go_now = []
if should_run_auto_scan():
    with st.spinner("Auto-scanning market..."):
        new_go_now = run_auto_scan_now()

# ── LIVE STATUS BAR ───────────────────────────────────────────────────────────
sb1, sb2, sb3 = st.columns([3,2,2])
with sb1:
    enabled = st.session_state.auto_scan_enabled
    if mstatus == "open":
        toggle_label = "🟢 AUTO-SCAN ON" if enabled else "⚫ AUTO-SCAN OFF"
        if st.button(toggle_label, key="auto_scan_toggle"):
            st.session_state.auto_scan_enabled = not enabled
            if st.session_state.auto_scan_enabled:
                st.session_state.auto_scan_last_run = None
            st.rerun()
    else:
        st.markdown("<div style='color:#8899aa;font-size:0.78rem;padding:6px 0'>Auto-scan available during market hours</div>", unsafe_allow_html=True)
with sb2:
    last    = st.session_state.auto_scan_last_run
    enabled = st.session_state.auto_scan_enabled
    if last and enabled:
        secs_ago = int((datetime.now()-last).total_seconds())
        next_in  = max(0, SCAN_INTERVAL-secs_ago)
        st.markdown(f"<div style='font-size:0.72rem;color:#8899aa;padding:6px 0'>Last: <span style='color:#d0dae8'>{secs_ago//60}m {secs_ago%60}s ago</span><br>Next: <span style='color:#00e5aa'>{next_in//60}m {next_in%60}s</span></div>", unsafe_allow_html=True)
with sb3:
    go_c  = len(st.session_state.auto_scan_go_now)
    wat_c = len(st.session_state.auto_scan_watching)
    dk_c  = len(st.session_state.auto_scan_on_deck)
    if go_c + wat_c + dk_c > 0:
        mkt_b = st.session_state.auto_scan_mkt
        mc    = "#00e5aa" if mkt_b=="bullish" else "#ff4d6d" if mkt_b=="bearish" else "#f0c040"
        st.markdown(f"<div style='font-size:0.72rem;padding:6px 0'><span style='color:{mc}'>● {mkt_b.upper()}</span> &nbsp;<span style='color:#00e5aa'>▲{go_c} GO</span> &nbsp;<span style='color:#f0c040'>◉{wat_c} WATCH</span> &nbsp;<span style='color:#6699cc'>◎{dk_c} DECK</span></div>", unsafe_allow_html=True)

# ── GO NOW ALERT BANNER ───────────────────────────────────────────────────────
for ng in new_go_now:
    is_bull_ng = ng["direction"] == "bullish"
    dc_ng = "#00e5aa" if is_bull_ng else "#ff4d6d"
    st.markdown(f"""
    <div style='background:#061a10;border:2px solid #00e5aa;border-radius:10px;padding:14px 18px;margin:6px 0'>
        <div style='font-family:monospace;font-size:0.65rem;letter-spacing:3px;color:#00e5aa;margin-bottom:4px'>🚨 NEW GO NOW SIGNAL</div>
        <div style='font-size:1.1rem;font-weight:700;color:{dc_ng}'>{"BUY CALL" if is_bull_ng else "BUY PUT"} — {ng["ticker"]}</div>
        <div style='font-size:0.8rem;color:#8899aa;margin-top:2px'>{ng["pattern"]} · {ng["confidence"]}% · {ng["gates_passed"]}/7 gates · Strike ${ng["opt"]["strike"]:.2f} · Target ${ng["opt"]["target"]:.2f} · Stop ${ng["opt"]["stop"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.components.v1.html("""<script>
    try {
        var ctx=new(window.AudioContext||window.webkitAudioContext)();
        [440,554,659].forEach(function(f,i){
            var o=ctx.createOscillator(),g=ctx.createGain();
            o.connect(g);g.connect(ctx.destination);
            o.frequency.value=f;o.type="sine";
            g.gain.setValueAtTime(0.3,ctx.currentTime+i*0.18);
            g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+i*0.18+0.4);
            o.start(ctx.currentTime+i*0.18);o.stop(ctx.currentTime+i*0.18+0.4);
        });
    } catch(e){}
    </script>""", height=0)

# Keep the page live — rerun every second when auto-scan is on
if st.session_state.auto_scan_enabled and mstatus == "open":
    import time as _time; _time.sleep(1); st.rerun()

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

tab4,tab1,tab2,tab3,tab5 = st.tabs(["SCAN","SIGNALS","CHART","BACKTEST","JOURNAL"])

with tab1:
    cands_quick, tfs_quick = build_multi_tf_candidates(selected_ticker, toggles, account_size, risk_pct, dte_quick, "quick", atr=atr)
    cands_swing, tfs_swing = build_multi_tf_candidates(selected_ticker, toggles, account_size, risk_pct, dte_swing, "swing", atr=atr)

    no_quick = len(cands_quick) == 0
    no_swing = len(cands_swing) == 0

    if no_quick and no_swing:
        st.markdown("""<div style='background:#111827;border:2px solid #1e2d40;border-radius:12px;padding:24px;text-align:center;color:#8899aa'>
            <div style='font-size:1rem;font-weight:700;margin:8px 0'>NO SIGNALS FOUND</div>
            <div style='font-size:0.85rem'>Try Daily or 4 Hour timeframe, enable more patterns, or check a different ticker.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Use first candidate list that has signals for shared logic below
        candidates = cands_quick if not no_quick else cands_swing
        # Side-by-side columns: Quick (purple) | Swing (blue)
        col_q, col_s = st.columns(2)
        with col_q:
            st.markdown(f"<div style='background:#1a0a3a;border-radius:6px;padding:6px 12px;text-align:center;color:#aa88ff;font-family:monospace;font-size:0.75rem;letter-spacing:1px'>⚡ QUICK &nbsp;|&nbsp; {dte_quick}DTE</div>", unsafe_allow_html=True)
        with col_s:
            st.markdown(f"<div style='background:#0a1a2a;border-radius:6px;padding:6px 12px;text-align:center;color:#6699cc;font-family:monospace;font-size:0.75rem;letter-spacing:1px'>📅 SWING &nbsp;|&nbsp; {dte_swing}DTE</div>", unsafe_allow_html=True)

        with col_q:
            render_signal_cards(cands_quick, selected_ticker, dte_quick, "quick", "q",
                                df, current_price, atr, iv_rank, earnings_days,
                                mstatus, mtext, account_size, risk_pct,
                                htf_trend, htf_rsi, htf_ema, liq_ok)
        with col_s:
            render_signal_cards(cands_swing, selected_ticker, dte_swing, "swing", "s",
                                df, current_price, atr, iv_rank, earnings_days,
                                mstatus, mtext, account_size, risk_pct,
                                htf_trend, htf_rsi, htf_ema, liq_ok)

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
    st.markdown("<div class='section-title'>MARKET SCANNER</div>", unsafe_allow_html=True)

    # Auto-scan settings (stored in session state so auto-scan uses same settings)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scan_style = st.radio("Scan Mode", ["⚡ Quick","📅 Swing","Both"], index=2, horizontal=True)
        scan_style_key = "quick" if "Quick" in scan_style else "swing" if "Swing" in scan_style else "both"
        st.session_state.auto_scan_settings["style"] = scan_style_key
    with sc2:
        max_premium = st.number_input("Max Premium ($/sh)", value=5.00, step=0.50, min_value=0.50)
        st.session_state.auto_scan_settings["max_premium"] = max_premium
    with sc3:
        scan_universe_choice = st.radio("Universe", ["My Watchlist","Full Scan (120+)"], index=0, horizontal=True)
        scan_list = WATCHLIST if "Watchlist" in scan_universe_choice else SCAN_UNIVERSE
        st.session_state.auto_scan_settings["scan_list"] = "watchlist" if "Watchlist" in scan_universe_choice else "full"

    # Show auto-scan results if available, else prompt manual scan
    has_auto_results = len(st.session_state.auto_scan_go_now + st.session_state.auto_scan_watching + st.session_state.auto_scan_on_deck) > 0
    if has_auto_results:
        last_t = st.session_state.auto_scan_last_run
        last_str = last_t.strftime("%I:%M:%S %p") if last_t else "unknown"
        st.caption(f"Showing auto-scan results from {last_str} · {len(scan_list)} tickers scanned")
        go_now   = st.session_state.auto_scan_go_now
        watching = st.session_state.auto_scan_watching
        on_deck  = st.session_state.auto_scan_on_deck
        mkt_bias = st.session_state.auto_scan_mkt
        if st.button("🔄 Scan Now", use_container_width=True):
            with st.spinner("Scanning..."):
                go_now, watching, on_deck, mkt_bias = full_scan(
                    scan_list, toggles, account_size, risk_pct,
                    dte_quick, dte_swing, max_premium, scan_style_key
                )
                st.session_state.auto_scan_go_now   = go_now
                st.session_state.auto_scan_watching = watching
                st.session_state.auto_scan_on_deck  = on_deck
                st.session_state.auto_scan_mkt      = mkt_bias
                st.session_state.auto_scan_last_run = datetime.now()
                st.rerun()
    else:
        st.caption(f"Scanning {len(scan_list)} tickers through full precision stack")
        if st.button("🔍 RUN SCAN", type="primary", use_container_width=True):
            prog_bar  = st.progress(0)
            prog_text = st.empty()

            def update_progress(idx, total, ticker):
                prog_bar.progress((idx+1)/total)
                prog_text.text(f"Scanning {ticker}... ({idx+1}/{total})")

            go_now, watching, on_deck, mkt_bias = full_scan(
                scan_list, toggles, account_size, risk_pct,
                dte_quick, dte_swing, max_premium, scan_style_key,
                progress_cb=update_progress
            )
            st.session_state.auto_scan_go_now   = go_now
            st.session_state.auto_scan_watching = watching
            st.session_state.auto_scan_on_deck  = on_deck
            st.session_state.auto_scan_mkt      = mkt_bias
            st.session_state.auto_scan_last_run = datetime.now()
            prog_bar.empty(); prog_text.empty()
            st.rerun()

    # Render results from session state after manual scan
    go_now   = st.session_state.auto_scan_go_now
    watching = st.session_state.auto_scan_watching
    on_deck  = st.session_state.auto_scan_on_deck
    mkt_bias = st.session_state.auto_scan_mkt

    if go_now or watching or on_deck:
        bias_color = "#00e5aa" if mkt_bias=="bullish" else "#ff4d6d" if mkt_bias=="bearish" else "#f0c040"
        bias_icon  = "📈" if mkt_bias=="bullish" else "📉" if mkt_bias=="bearish" else "↔️"
        total_found = len(go_now)+len(watching)+len(on_deck)

        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             background:#0d1421;border:1px solid {bias_color}33;border-radius:10px;
             padding:10px 16px;margin-bottom:4px;font-size:0.72rem'>
            <div style='color:{bias_color}'>{bias_icon} MARKET: <b>{mkt_bias.upper()}</b></div>
            <div style='display:flex;gap:20px'>
                <span style='color:#00e5aa'>● {len(go_now)} GO NOW</span>
                <span style='color:#f0c040'>● {len(watching)} WATCHING</span>
                <span style='color:#6699cc'>● {len(on_deck)} ON DECK</span>
            </div>
            <div style='color:#8899aa'>{total_found} total signals</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mobile-first card renderer ───────────────────────────────────────
        def conf_color(c):
            return "#00e5aa" if c>=90 else "#40d080" if c>=80 else "#f0c040" if c>=70 else "#6699aa"
        def conf_label(c):
            return "GO ALL IN" if c>=90 else "STRONG" if c>=80 else "WATCH IT" if c>=70 else "WAIT"

        def mobile_card(r, bucket, idx):
            is_bull = r["direction"] == "bullish"
            dc  = "#00e5aa" if is_bull else "#ff4d6d"
            cc  = conf_color(r["confidence"])
            cl  = conf_label(r["confidence"])
            opt = r["opt"]
            gc  = "#00e5aa" if r["gates_passed"]>=6 else "#f0c040" if r["gates_passed"]>=5 else "#ff4d6d"
            exh_ok = r.get("exh_confirmed", False)
            rv     = round(r.get("rel_vol", 1.0), 1)
            block  = r.get("block_detected", False)
            si     = "⚡" if r["style"]=="quick" else "📅"
            border = "#00e5aa44" if bucket=="go_now" else "#f0c04044" if bucket=="watching" else "#1a2535"

            # Build card using % string formatting to avoid all quote conflicts
            R = 28
            circ = round(2 * 3.14159 * R, 1)
            dash = round((r["confidence"] / 100) * circ, 1)
            act_bg  = "#00e5aa22" if is_bull else "#ff4d6d22"
            sty_bg  = "#1a0a3a"  if r["style"] == "quick" else "#0a1a2a"
            sty_fg  = "#aa88ff"  if r["style"] == "quick" else "#6699cc"
            blk_tag = "<span style='font-size:0.58rem;color:#f0c040'>⚡ BLOCK</span>" if block else ""
            exh_txt = "✅ confirmed" if exh_ok else "⏳ watching"
            action  = "CALL" if is_bull else "PUT"
            parts = [
                "<div style='background:#0d1421;border:1px solid %s;border-radius:12px;padding:14px 16px;margin-bottom:8px'>" % border,
                "<div style='display:flex;align-items:center;gap:12px'>",
                "<div style='position:relative;width:68px;height:68px;flex-shrink:0'>",
                "<svg width='68' height='68' style='transform:rotate(-90deg);display:block'>",
                "<circle cx='34' cy='34' r='%s' fill='none' stroke='#1a2535' stroke-width='5'/>" % R,
                "<circle cx='34' cy='34' r='%s' fill='none' stroke='%s' stroke-width='5' stroke-dasharray='%s %s' stroke-linecap='round'/>" % (R, cc, dash, circ),
                "</svg>",
                "<div style='position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center'>",
                "<div style='font-size:0.95rem;font-weight:700;color:%s;line-height:1'>%s</div>" % (cc, r["confidence"]),
                "<div style='font-size:0.42rem;color:%s;letter-spacing:1px;margin-top:1px'>%%</div>" % cc,
                "</div></div>",
                "<div style='flex:1;min-width:0'>",
                "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px'>",
                "<span style='font-size:1.05rem;font-weight:700;color:%s'>%s</span>" % (dc, r["ticker"]),
                "<span style='font-size:0.6rem;background:%s;color:%s;padding:2px 6px;border-radius:4px;font-weight:700'>%s</span>" % (act_bg, dc, action),
                "<span style='font-size:0.58rem;background:%s;color:%s;padding:2px 6px;border-radius:4px'>%s %s</span>" % (sty_bg, sty_fg, si, r["style"].upper()),
                blk_tag,
                "</div>",
                "<div style='font-size:0.69rem;color:#8899aa'>%s</div>" % r["pattern"],
                "<div style='font-size:0.65rem;color:#8899aa;margin-top:2px'>%sx vol &nbsp;·&nbsp; %s</div>" % (rv, exh_txt),
                "</div>",
                "<div style='text-align:right;flex-shrink:0'>",
                "<div style='font-size:0.56rem;font-weight:700;color:%s;background:%s22;padding:2px 7px;border-radius:6px;letter-spacing:1px;margin-bottom:5px;display:inline-block'>%s</div>" % (cc, cc, cl),
                "<div style='font-size:0.65rem;color:#8899aa'>Gate <span style='color:%s;font-weight:700'>%s/7</span></div>" % (gc, r["gates_passed"]),
                "<div style='font-size:0.65rem;color:#8899aa;margin-top:3px'>Strike <span style='color:#d0dae8;font-weight:700'>$%.2f</span></div>" % opt["strike"],
                "</div></div></div>",
            ]
            st.markdown("".join(parts), unsafe_allow_html=True)

            with st.expander(f"📊 {r['ticker']} full details"):
                c1, c2 = st.columns(2)
                items_l = [("TARGET", f"${opt['target']:.2f}", "#00e5aa"),
                           ("PREMIUM", f"${opt['premium']:.2f}/sh", "#d0dae8"),
                           ("MAX LOSS", f"${opt['max_loss']:,.0f}", "#ff4d6d"),
                           ("IV RANK", f"{r['iv_rank']}%" if r["iv_rank"] else "N/A", "#f0c040")]
                items_r = [("STOP OUT", f"${opt['stop']:.2f}", "#ff4d6d"),
                           ("EST PROFIT", f"${opt['profit_at_target']:,.0f}", "#00e5aa"),
                           ("R:R", f"{opt['rr_option']:.1f}x", "#00e5aa" if opt["rr_option"]>=2 else "#f0c040"),
                           ("EXPIRES", opt["expiration"], "#8899aa")]
                with c1:
                    for lbl, val, col in items_l:
                        st.markdown(
                            "<div style='background:#0d1421;border-radius:8px;padding:10px;margin-bottom:6px'>"
                            "<div style='font-size:0.58rem;color:#8899aa'>%s</div>"
                            "<div style='font-size:0.95rem;font-weight:700;color:%s'>%s</div></div>" % (lbl, col, val),
                            unsafe_allow_html=True)
                with c2:
                    for lbl, val, col in items_r:
                        st.markdown(
                            "<div style='background:#0d1421;border-radius:8px;padding:10px;margin-bottom:6px'>"
                            "<div style='font-size:0.58rem;color:#8899aa'>%s</div>"
                            "<div style='font-size:0.95rem;font-weight:700;color:%s'>%s</div></div>" % (lbl, col, val),
                            unsafe_allow_html=True)

                side = "below" if is_bull else "above"
                st.markdown(
                    "<div style='background:#080c12;border-radius:8px;padding:10px 12px;font-size:0.72rem;color:#8899aa;margin:2px 0 8px;line-height:1.6'>"
                    "<span style='color:#00e5aa;font-weight:700'>Take 50%% off</span> at $%.2f/sh &nbsp;·&nbsp;"
                    "<span style='color:#ff4d6d;font-weight:700'>Close all</span> if %s $%.2f</div>" % (opt['exit_take_half'], side, opt['stop']),
                    unsafe_allow_html=True)

                sig_detail = r.get("signal_detail", [])
                exh        = r.get("exh_reasons", [])
                signals_hit = r.get("signals_hit", 0)
                if sig_detail:
                    st.markdown("<div style='font-size:0.58rem;color:#8899aa;letter-spacing:2px;margin-bottom:4px'>SIGNAL CHECK (%s/5)</div>" % signals_hit, unsafe_allow_html=True)
                    for item in sig_detail:
                        good = item.startswith("✅")
                        tcol = "#e0e6f0" if good else "#8899aa"
                        st.markdown("<div style='font-size:0.73rem;color:%s;padding:2px 0'>%s</div>" % (tcol, item), unsafe_allow_html=True)
                if exh:
                    st.markdown("<div style='font-size:0.58rem;color:#8899aa;letter-spacing:2px;margin:6px 0 4px'>EXHAUSTION DETAIL</div>", unsafe_allow_html=True)
                    for reason in exh:
                        good = any(x in reason for x in ["confirmed","forming","Higher low","Lower high","Climax","Capitulation","Hammer","doji","star","reclaim","holding","rising","falling"])
                        col  = "#00e5aa" if good else "#ff4d6d"
                        tcol = "#e0e6f0" if good else "#8899aa"
                        dot  = "●" if good else "○"
                        st.markdown("<div style='font-size:0.71rem;color:%s;padding:1px 0'><span style='color:%s'>%s</span> %s</div>" % (tcol, col, dot, reason), unsafe_allow_html=True)
                if st.button(f"Log {r['ticker']} {r['action']}", key=f"log_{bucket}_{idx}", use_container_width=True):
                    log_trade(r["ticker"], r["sig"], r["opt"], r["gates_passed"], 7, r["elevate"])
                    st.success("Logged!")

        def section_hdr(label, color, count):
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:20px 0 8px'>
                <div style='width:3px;height:16px;background:{color};border-radius:2px;flex-shrink:0'></div>
                <span style='font-size:0.65rem;letter-spacing:3px;color:{color};font-weight:700'>{label}</span>
                <div style='flex:1;height:1px;background:#1a2535'></div>
                <span style='font-size:0.62rem;color:#8899aa'>{count} signal{"s" if count!=1 else ""}</span>
            </div>""", unsafe_allow_html=True)

        def empty_bkt(msg):
            st.markdown(f"<div style='padding:14px;color:#8899aa;font-size:0.78rem;background:#0d1421;border-radius:10px;text-align:center'>{msg}</div>", unsafe_allow_html=True)

        section_hdr("GO NOW", "#00e5aa", len(go_now))
        if go_now:
            for i, r in enumerate(go_now[:5]):   mobile_card(r, "go_now",   i)
        else:
            empty_bkt("No GO NOW signals — exhaustion not confirmed or gates not cleared.")

        section_hdr("WATCHING", "#f0c040", len(watching))
        if watching:
            for i, r in enumerate(watching[:8]): mobile_card(r, "watching", i)
        else:
            empty_bkt("No setups in confirmation phase right now.")

        section_hdr("ON DECK", "#6699cc", len(on_deck))
        if on_deck:
            for i, r in enumerate(on_deck[:10]): mobile_card(r, "on_deck",  i)
        else:
            empty_bkt("No developing setups found.")

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

