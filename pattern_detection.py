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
                entry_price = float(df["close"].iloc[-1])
                avg_vol = df["volume"].iloc[idx1:idx2].mean()
                breakout_vol = df["volume"].iloc[-1]
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
