<<<<<<< HEAD
"""
Pattern Detection Module
Detects: Double Bottom, Double Top, Break & Retest
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Setup:
    ticker: str
    pattern: str
    direction: str  # 'bullish' or 'bearish'
=======
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
>>>>>>> 6ec97029 (initial upload)
    entry_price: float
    stop_loss: float
    target: float
    rr_ratio: float
<<<<<<< HEAD
    confirmed: bool
    neckline: float = 0.0
    # Double bottom fields
    bottom1_idx: Optional[int] = None
    bottom2_idx: Optional[int] = None
    # Double top fields
    top1_idx: Optional[int] = None
    top2_idx: Optional[int] = None


def find_swing_lows(df: pd.DataFrame, window: int = 5) -> List[int]:
    """Find swing low indices."""
    lows = []
    low = df["low"].values
    for i in range(window, len(low) - window):
        if low[i] == min(low[i - window: i + window + 1]):
            lows.append(i)
    return lows


def find_swing_highs(df: pd.DataFrame, window: int = 5) -> List[int]:
    """Find swing high indices."""
    highs = []
    high = df["high"].values
    for i in range(window, len(high) - window):
        if high[i] == max(high[i - window: i + window + 1]):
            highs.append(i)
    return highs


def detect_double_bottom(df: pd.DataFrame, ticker: str, rr_min: float = 2.0,
                         tolerance: float = 0.03, window: int = 5) -> List[Setup]:
    """
    Detect double bottom patterns (bullish reversal).
    Two swing lows at similar price levels followed by a neckline break.
    """
    setups = []
    if len(df) < 30:
        return setups

    swing_lows = find_swing_lows(df, window)
    if len(swing_lows) < 2:
        return setups

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    current_price = close[-1]

    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            idx1 = swing_lows[i]
            idx2 = swing_lows[j]

            # Must be separated enough
            if idx2 - idx1 < 5:
                continue

            price1 = low[idx1]
            price2 = low[idx2]

            # Two bottoms within tolerance of each other
            avg_bottom = (price1 + price2) / 2
            if abs(price1 - price2) / avg_bottom > tolerance:
                continue

            # Neckline = highest high between the two bottoms
            neckline = float(max(high[idx1:idx2 + 1]))

            # Check if price has broken above neckline (confirmation)
            recent_close = close[idx2:]
            confirmed = any(c > neckline for c in recent_close)

            if not confirmed:
                continue

            # Entry just above neckline
            entry = round(neckline * 1.002, 2)
            stop = round(min(price1, price2) * 0.99, 2)
            risk = entry - stop
            if risk <= 0:
                continue

            reward = risk * 2.0
            target = round(entry + reward, 2)
            rr = round(reward / risk, 2)

            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker,
                pattern="DoubleBottom",
                direction="bullish",
                entry_price=entry,
                stop_loss=stop,
                target=target,
                rr_ratio=rr,
                confirmed=True,
                neckline=neckline,
                bottom1_idx=idx1,
                bottom2_idx=idx2,
            ))

    # Return most recent setups
    return setups[-3:] if setups else []


def detect_double_top(df: pd.DataFrame, ticker: str, rr_min: float = 2.0,
                      tolerance: float = 0.03, window: int = 5) -> List[Setup]:
    """
    Detect double top patterns (bearish reversal).
    Two swing highs at similar price levels followed by a neckline break down.
    """
    setups = []
    if len(df) < 30:
        return setups

    swing_highs = find_swing_highs(df, window)
    if len(swing_highs) < 2:
        return setups

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    for i in range(len(swing_highs) - 1):
        for j in range(i + 1, len(swing_highs)):
            idx1 = swing_highs[i]
            idx2 = swing_highs[j]

            # Must be separated enough
            if idx2 - idx1 < 5:
                continue

            price1 = high[idx1]
            price2 = high[idx2]

            # Two tops within tolerance of each other
            avg_top = (price1 + price2) / 2
            if abs(price1 - price2) / avg_top > tolerance:
                continue

            # Neckline = lowest low between the two tops
            neckline = float(min(low[idx1:idx2 + 1]))

            # Check if price has broken below neckline (confirmation)
            recent_close = close[idx2:]
            confirmed = any(c < neckline for c in recent_close)

            if not confirmed:
                continue

            # Entry just below neckline
            entry = round(neckline * 0.998, 2)
            stop = round(max(price1, price2) * 1.01, 2)
            risk = stop - entry
            if risk <= 0:
                continue

            reward = risk * 2.0
            target = round(entry - reward, 2)
            rr = round(reward / risk, 2)

            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker,
                pattern="DoubleTop",
                direction="bearish",
                entry_price=entry,
                stop_loss=stop,
                target=target,
                rr_ratio=rr,
                confirmed=True,
                neckline=neckline,
                top1_idx=idx1,
                top2_idx=idx2,
            ))

    return setups[-3:] if setups else []


def detect_break_and_retest(df: pd.DataFrame, ticker: str, rr_min: float = 2.0,
                             window: int = 5) -> List[Setup]:
    """
    Detect break and retest patterns (both directions).
    Price breaks a key level, pulls back to retest it, then continues.
    """
    setups = []
    if len(df) < 40:
        return setups

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    current_price = close[-1]

    swing_highs = find_swing_highs(df, window)
    swing_lows = find_swing_lows(df, window)

    # Bullish break & retest — break above resistance, retest as support
    for sh_idx in swing_highs:
        if sh_idx > len(df) - 10:
            continue
        resistance = high[sh_idx]

        # Look for a break above this level
        post = close[sh_idx + 1:]
        break_bars = [k for k, c in enumerate(post) if c > resistance * 1.005]
        if not break_bars:
            continue
        break_bar = sh_idx + 1 + break_bars[0]

        # Look for retest — price comes back close to the broken level
        after_break = close[break_bar:]
        retest_bars = [k for k, c in enumerate(after_break)
                       if resistance * 0.995 < c < resistance * 1.015]
        if not retest_bars:
            continue
        retest_bar = break_bar + retest_bars[0]

        # Price must have bounced after retest
        if retest_bar >= len(close) - 2:
            continue
        post_retest = close[retest_bar:]
        bounced = any(c > resistance * 1.01 for c in post_retest)
        if not bounced:
            continue

        entry = round(float(close[retest_bar]) * 1.002, 2)
        stop = round(resistance * 0.985, 2)
        risk = entry - stop
        if risk <= 0:
            continue
        target = round(entry + risk * 2.0, 2)
        rr = round((target - entry) / risk, 2)

        if rr < rr_min:
            continue

        setups.append(Setup(
            ticker=ticker,
            pattern="BreakRetest",
            direction="bullish",
            entry_price=entry,
            stop_loss=stop,
            target=target,
            rr_ratio=rr,
            confirmed=True,
            neckline=resistance,
        ))

    # Bearish break & retest — break below support, retest as resistance
    for sl_idx in swing_lows:
        if sl_idx > len(df) - 10:
            continue
        support = low[sl_idx]

        # Look for a break below this level
        post = close[sl_idx + 1:]
        break_bars = [k for k, c in enumerate(post) if c < support * 0.995]
        if not break_bars:
            continue
        break_bar = sl_idx + 1 + break_bars[0]

        # Look for retest — price comes back close to broken level
        after_break = close[break_bar:]
        retest_bars = [k for k, c in enumerate(after_break)
                       if support * 0.985 < c < support * 1.005]
        if not retest_bars:
            continue
        retest_bar = break_bar + retest_bars[0]

        if retest_bar >= len(close) - 2:
            continue
        post_retest = close[retest_bar:]
        rejected = any(c < support * 0.99 for c in post_retest)
        if not rejected:
            continue

        entry = round(float(close[retest_bar]) * 0.998, 2)
        stop = round(support * 1.015, 2)
        risk = stop - entry
        if risk <= 0:
            continue
        target = round(entry - risk * 2.0, 2)
        rr = round((entry - target) / risk, 2)

        if rr < rr_min:
            continue

        setups.append(Setup(
            ticker=ticker,
            pattern="BreakRetest",
            direction="bearish",
            entry_price=entry,
            stop_loss=stop,
            target=target,
            rr_ratio=rr,
            confirmed=True,
            neckline=support,
        ))

    return setups[-3:] if setups else []


def add_confluence_filters(df: pd.DataFrame, setup: Setup) -> dict:
    """
    Calculate confluence factors for a given setup.
    Returns dict of factor results.
    """
    close = df["close"]
    price = float(close.iloc[-1])
    is_bull = setup.direction == "bullish"

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    # VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = float((tp * df["volume"]).cumsum().iloc[-1] / df["volume"].cumsum().iloc[-1])

    # EMA 20
    ema20 = float(close.ewm(span=20).mean().iloc[-1])

    # Volume
    avg_vol = float(df["volume"].iloc[-20:].mean())
    cur_vol = float(df["volume"].iloc[-1])

    return {
        "rsi": rsi,
        "vwap": vwap,
        "ema20": ema20,
        "volume_ratio": cur_vol / avg_vol if avg_vol > 0 else 1.0,
        "above_vwap": price > vwap,
        "above_ema": price > ema20,
        "rsi_in_zone": 35 < rsi < 65,
        "volume_spike": cur_vol > avg_vol * 1.2,
    }
=======
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
>>>>>>> 6ec97029 (initial upload)
