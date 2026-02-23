import os

files = {
"pattern_detection.py": '''
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Optional

@dataclass
class DoubleBottomSetup:
    ticker: str
    bottom1_idx: int
    bottom2_idx: int
    bottom1_price: float
    bottom2_price: float
    neckline: float
    confirmed: bool
    entry_price: float
    stop_loss: float
    target: float
    rr_ratio: float
    iv_rank: Optional[float] = None
    volume_confirmed: bool = False

@dataclass
class BreakRetestSetup:
    ticker: str
    breakout_idx: int
    retest_idx: int
    key_level: float
    breakout_price: float
    retest_price: float
    direction: str
    confirmed: bool
    entry_price: float
    stop_loss: float
    target: float
    rr_ratio: float

def find_swing_lows(prices, prominence=0.02, distance=5):
    inverted = -prices.values
    abs_prominence = prices.mean() * prominence
    peaks, _ = find_peaks(inverted, prominence=abs_prominence, distance=distance)
    return peaks

def find_swing_highs(prices, prominence=0.02, distance=5):
    abs_prominence = prices.mean() * prominence
    peaks, _ = find_peaks(prices.values, prominence=abs_prominence, distance=distance)
    return peaks

def detect_double_bottom(df, ticker, tolerance=0.03, min_bars_between=5, max_bars_between=60, rr_min=2.0, volume_multiplier=1.2):
    setups = []
    lows = find_swing_lows(df["low"], prominence=0.015, distance=min_bars_between)
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            idx1, idx2 = lows[i], lows[j]
            if (idx2-idx1) < min_bars_between or (idx2-idx1) > max_bars_between:
                continue
            p1, p2 = df["low"].iloc[idx1], df["low"].iloc[idx2]
            if abs(p1-p2)/p1 > tolerance:
                continue
            between = df["high"].iloc[idx1:idx2]
            neckline = between.max()
            avg_bottom = (p1+p2)/2
            if (neckline-avg_bottom)/avg_bottom < 0.02:
                continue
            post_break = df["close"].iloc[idx2:]
            breakout_mask = post_break > neckline*1.001
            confirmed = breakout_mask.any()
            if confirmed:
                breakout_bar = breakout_mask.idxmax()
                entry_price = df["close"].iloc[breakout_bar]
                avg_vol = df["volume"].iloc[idx1:idx2].mean()
                breakout_vol = df["volume"].iloc[breakout_bar]
                vol_confirmed = breakout_vol > avg_vol*volume_multiplier
            else:
                entry_price = neckline*1.001
                vol_confirmed = False
            stop_loss = min(p1,p2)*0.99
            pattern_height = neckline - avg_bottom
            target = neckline + pattern_height
            risk = entry_price - stop_loss
            if risk <= 0:
                continue
            rr = (target-entry_price)/risk
            if rr < rr_min:
                continue
            setups.append(DoubleBottomSetup(ticker=ticker, bottom1_idx=idx1, bottom2_idx=idx2, bottom1_price=p1, bottom2_price=p2, neckline=neckline, confirmed=confirmed, entry_price=entry_price, stop_loss=stop_loss, target=target, rr_ratio=round(rr,2), volume_confirmed=vol_confirmed))
    return setups

def detect_break_and_retest(df, ticker, level_tolerance=0.005, min_bars_after_break=2, max_bars_after_break=30, rr_min=2.0, direction="both"):
    setups = []
    if direction in ("bullish","both"):
        swing_highs = find_swing_highs(df["high"], prominence=0.02, distance=5)
        for sh_idx in swing_highs:
            level = df["high"].iloc[sh_idx]
            post = df.iloc[sh_idx+1:]
            break_mask = post["close"] > level*(1+level_tolerance)
            if not break_mask.any():
                continue
            break_bar_pos = break_mask.idxmax()
            break_bar_idx = df.index.get_loc(break_bar_pos)
            retest_window = df.iloc[break_bar_idx+min_bars_after_break:break_bar_idx+max_bars_after_break]
            if retest_window.empty:
                continue
            retest_mask = (retest_window["low"] <= level*(1+level_tolerance)) & (retest_window["close"] > level*(1-level_tolerance))
            if not retest_mask.any():
                continue
            retest_bar_pos = retest_mask.idxmax()
            retest_bar_idx = df.index.get_loc(retest_bar_pos)
            retest_price = df["close"].iloc[retest_bar_idx]
            confirmed = df["close"].iloc[retest_bar_idx] > level
            entry_price = level*1.002
            stop_loss = level*(1-level_tolerance*3)
            risk = entry_price - stop_loss
            if risk <= 0:
                continue
            target = entry_price + (risk*rr_min)
            rr = (target-entry_price)/risk
            setups.append(BreakRetestSetup(ticker=ticker, breakout_idx=break_bar_idx, retest_idx=retest_bar_idx, key_level=level, breakout_price=df["close"].iloc[break_bar_idx], retest_price=retest_price, direction="bullish", confirmed=confirmed, entry_price=entry_price, stop_loss=stop_loss, target=target, rr_ratio=round(rr,2)))
    return setups

def add_confluence_filters(df, setup, rsi_period=14):
    close = df["close"]
    volume = df["volume"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain/loss
    rsi = 100-(100/(1+rs))
    current_rsi = rsi.iloc[-1]
    typical_price = (df["high"]+df["low"]+df["close"])/3
    vwap = (typical_price*volume).cumsum()/volume.cumsum()
    current_vwap = vwap.iloc[-1]
    current_price = close.iloc[-1]
    ema20 = close.ewm(span=20).mean().iloc[-1]
    return {"rsi": round(current_rsi,2), "rsi_bullish": 40 < current_rsi < 60, "price_above_vwap": current_price > current_vwap, "price_above_ema20": current_price > ema20, "vwap": round(current_vwap,2), "ema20": round(ema20,2)}
''',

"backtester.py": '''
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from pattern_detection import detect_double_bottom, detect_break_and_retest, add_confluence_filters

@dataclass
class TradeResult:
    ticker: str
    pattern: str
    entry_price: float
    stop_loss: float
    target: float
    rr_ratio: float
    outcome: str
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    confluence_score: int = 0

@dataclass
class BacktestReport:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    expectancy: float = 0.0
    trades: list = field(default_factory=list)

def simulate_trade(df, setup, entry_bar, max_bars=40):
    entry = setup.entry_price
    stop = setup.stop_loss
    target = setup.target
    pattern = setup.__class__.__name__.replace("Setup","")
    confluence = add_confluence_filters(df.iloc[:entry_bar+1], setup)
    conf_score = sum([confluence.get("rsi_bullish",False), confluence.get("price_above_vwap",False), confluence.get("price_above_ema20",False)])
    outcome = "open"
    exit_price = entry
    bars_held = 0
    future = df.iloc[entry_bar+1:entry_bar+1+max_bars]
    for i, (_, bar) in enumerate(future.iterrows()):
        bars_held = i+1
        if bar["low"] <= stop:
            outcome = "loss"; exit_price = stop; break
        if bar["high"] >= target:
            outcome = "win"; exit_price = target; break
    if outcome == "open":
        exit_price = future["close"].iloc[-1] if not future.empty else entry
    pnl_pct = ((exit_price-entry)/entry)*100
    return TradeResult(ticker=getattr(setup,"ticker","UNK"), pattern=pattern, entry_price=entry, stop_loss=stop, target=target, rr_ratio=setup.rr_ratio, outcome=outcome, exit_price=exit_price, pnl_pct=round(pnl_pct,2), bars_held=bars_held, confluence_score=conf_score)

def run_backtest(df, ticker, pattern="both", db_rr_min=2.0, br_rr_min=2.0, min_confluence_score=0, in_sample_pct=0.7):
    split = int(len(df)*in_sample_pct)
    in_sample = df.iloc[:split].reset_index(drop=True)
    out_sample = df.reset_index(drop=True)
    all_setups = []
    if pattern in ("double_bottom","both"):
        for s in detect_double_bottom(in_sample, ticker, rr_min=db_rr_min):
            if s.confirmed: all_setups.append((s, s.bottom2_idx))
    if pattern in ("break_retest","both"):
        for s in detect_break_and_retest(in_sample, ticker, rr_min=br_rr_min):
            if s.confirmed: all_setups.append((s, s.retest_idx))
    report = BacktestReport()
    equity_curve = [0.0]
    for setup, entry_bar in all_setups:
        if entry_bar >= len(out_sample)-5: continue
        result = simulate_trade(out_sample, setup, entry_bar)
        if result.confluence_score < min_confluence_score: continue
        report.trades.append(result)
        report.total_trades += 1
        if result.outcome == "win": report.wins += 1; equity_curve.append(equity_curve[-1]+result.rr_ratio)
        elif result.outcome == "loss": report.losses += 1; equity_curve.append(equity_curve[-1]-1)
        else: report.open_trades += 1; equity_curve.append(equity_curve[-1])
    if report.total_trades > 0:
        closed = [t for t in report.trades if t.outcome != "open"]
        wins = [t for t in closed if t.outcome == "win"]
        losses = [t for t in closed if t.outcome == "loss"]
        report.win_rate = round(len(wins)/len(closed)*100,1) if closed else 0
        report.avg_rr = round(np.mean([t.rr_ratio for t in report.trades]),2)
        report.avg_win_pct = round(np.mean([t.pnl_pct for t in wins]),2) if wins else 0
        report.avg_loss_pct = round(np.mean([t.pnl_pct for t in losses]),2) if losses else 0
        eq = np.array(equity_curve)
        report.max_drawdown = round(float((eq - np.maximum.accumulate(eq)).min()),2)
        wr = report.win_rate/100
        avg_win_r = np.mean([t.rr_ratio for t in wins]) if wins else 0
        report.expectancy = round((wr*avg_win_r)-((1-wr)*1),2)
        returns = [t.rr_ratio if t.outcome=="win" else -1 for t in closed]
        if len(returns)>1 and np.std(returns)>0:
            report.sharpe = round(np.mean(returns)/np.std(returns)*np.sqrt(252),2)
    return report, equity_curve
''',

"options_chain.py": '''
import numpy as np, requests
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional

POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"

@dataclass
class OptionsRecommendation:
    ticker: str
    underlying_price: float
    iv_rank: float
    iv_percentile: float
    current_iv: float
    strategy: str
    direction: str
    setup_entry: float
    setup_stop: float
    setup_target: float
    expiration_used: str = ""
    dte: int = 0
    notes: list = field(default_factory=list)

def fetch_iv_rank(ticker, lookback_days=252):
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime(\"%Y-%m-%d\")}/{end.strftime(\"%Y-%m-%d\")}?adjusted=true&sort=asc&limit=365&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data.get("results"): return 50.0, 50.0, 0.30
        closes = [r["c"] for r in data["results"]]
        if len(closes) < 20: return 50.0, 50.0, 0.30
        log_returns = np.diff(np.log(closes))
        hv_series = []
        for i in range(20, len(log_returns)):
            hv_series.append(np.std(log_returns[i-20:i])*np.sqrt(252))
        if not hv_series: return 50.0, 50.0, 0.30
        current_iv = hv_series[-1]
        iv_range = max(hv_series) - min(hv_series)
        iv_rank = ((current_iv-min(hv_series))/iv_range*100) if iv_range > 0 else 50.0
        iv_pct = (sum(1 for v in hv_series if v <= current_iv)/len(hv_series))*100
        return round(iv_rank,1), round(iv_pct,1), round(current_iv,4)
    except:
        return 50.0, 50.0, 0.30

def get_options_recommendation(setup, account_size=10000, risk_per_trade_pct=0.01):
    ticker = setup.ticker
    direction = getattr(setup, "direction", "bullish")
    iv_rank, iv_pct, current_iv = fetch_iv_rank(ticker)
    if iv_rank > 60:
        strategy = "Bull Put Spread (credit)" if direction=="bullish" else "Bear Call Spread (credit)"
    elif iv_rank < 30:
        strategy = "Long Call" if direction=="bullish" else "Long Put"
    else:
        strategy = "Bull Call Spread (debit)" if direction=="bullish" else "Bear Put Spread (debit)"
    price = setup.entry_price
    long_strike = round(price*0.975/0.5)*0.5
    short_strike = round(price*1.05/0.5)*0.5
    long_mid = round(price*0.042,2)
    short_mid = round(long_mid*0.42,2)
    net_debit = round(long_mid-short_mid,2)
    spread_w = abs(short_strike-long_strike)
    max_profit = round((spread_w-net_debit)*100,2)
    max_loss = round(net_debit*100,2)
    breakeven = round(long_strike+net_debit,2)
    contracts = max(1, int((account_size*risk_per_trade_pct)/max_loss)) if max_loss else 1
    today = date.today()
    exp30 = (today+timedelta(days=30)).strftime("%Y-%m-%d")
    notes = [
        f"LONG CALL ${long_strike} / SHORT CALL ${short_strike} | Exp: {exp30} (30 DTE)",
        f"Net Debit: ${net_debit} | Max Profit: ${max_profit} | Max Loss: ${max_loss}",
        f"Breakeven: ${breakeven} | Contracts (1% risk): {contracts}",
        f"IV Rank: {iv_rank}% — {strategy}",
    ]
    return OptionsRecommendation(ticker=ticker, underlying_price=price, iv_rank=iv_rank, iv_percentile=iv_pct, current_iv=current_iv, strategy=strategy, direction=direction, setup_entry=setup.entry_price, setup_stop=setup.stop_loss, setup_target=setup.target, expiration_used=exp30, dte=30, notes=notes)
''',
}

for fname, content in files.items():
    with open(fname, "w") as f:
        f.write(content.strip())
    print(f"Created {fname}")
