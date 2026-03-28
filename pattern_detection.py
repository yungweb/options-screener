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
    entry_price: float
    stop_loss: float
    target: float
    rr_ratio: float
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


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TRADE PATTERNS (intraday — 5min/15min timeframes)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_vwap_reclaim(df: pd.DataFrame, ticker: str, rr_min: float = 1.5) -> List[Setup]:
    """
    VWAP Reclaim (bullish) / VWAP Rejection (bearish) — quick trade only.

    Bullish: price was below VWAP, reclaims it with a strong close above,
             confirmed by above-average volume on the reclaim candle.
    Bearish: price was above VWAP, rejects it with a strong close below,
             confirmed by above-average volume on the rejection candle.

    ATR-based levels: tight stop, 1x ATR target.
    """
    setups = []
    if len(df) < 20:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    open_  = df["open"].values
    volume = df["volume"].values

    # VWAP
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    vwap = vwap.values

    # ATR (14)
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    if atr <= 0:
        atr = float(close[-1]) * 0.01

    avg_vol = float(np.mean(volume[-20:]))
    price   = float(close[-1])

    # Need at least 3 bars to check reclaim/rejection
    for i in range(-5, -1):
        prev_close = float(close[i - 1])
        curr_close = float(close[i])
        prev_vwap  = float(vwap[i - 1])
        curr_vwap  = float(vwap[i])
        bar_vol    = float(volume[i])
        vol_ok     = bar_vol > avg_vol * 1.1

        # Bullish reclaim: prev close below VWAP, curr close above VWAP
        if prev_close < prev_vwap and curr_close > curr_vwap and vol_ok:
            entry  = round(curr_close * 1.001, 2)
            stop   = round(curr_vwap - atr * 0.3, 2)
            risk   = entry - stop
            if risk <= 0:
                continue
            target = round(entry + atr * 1.0, 2)
            rr     = round((target - entry) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="VWAPReclaim",
                direction="bullish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=curr_vwap,
            ))

        # Bearish rejection: prev close above VWAP, curr close below VWAP
        elif prev_close > prev_vwap and curr_close < curr_vwap and vol_ok:
            entry  = round(curr_close * 0.999, 2)
            stop   = round(curr_vwap + atr * 0.3, 2)
            risk   = stop - entry
            if risk <= 0:
                continue
            target = round(entry - atr * 1.0, 2)
            rr     = round((entry - target) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="VWAPRejection",
                direction="bearish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=curr_vwap,
            ))

    return setups[-2:] if setups else []


def detect_bull_bear_flag(df: pd.DataFrame, ticker: str,
                           rr_min: float = 1.5,
                           trade_style: str = "quick") -> List[Setup]:
    """
    Bull Flag (bullish continuation) / Bear Flag (bearish continuation).

    Quick version (5/15min): tight pole 3-8 bars, flag 3-6 bars, ATR target.
    Swing version (1hr/daily): pole 5-15 bars, flag 5-10 bars, measured move target.

    Bull Flag:
    - Strong up move (pole): 3 consecutive closes up, total move > 1.5x ATR
    - Consolidation (flag): slight pullback, lower highs, low volume
    - Breakout: close above flag high with volume expansion

    Bear Flag: mirror image.
    """
    setups = []
    min_bars = 25 if trade_style == "quick" else 40
    if len(df) < min_bars:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    # ATR
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    if atr <= 0:
        atr = float(close[-1]) * 0.01

    avg_vol = float(np.mean(volume[-20:]))
    n       = len(close)

    pole_min  = 3 if trade_style == "quick" else 5
    pole_max  = 8 if trade_style == "quick" else 15
    flag_min  = 3 if trade_style == "quick" else 5
    flag_max  = 6 if trade_style == "quick" else 12

    # Scan for pole start
    for pole_start in range(n - pole_max - flag_max - 3, n - flag_min - 5):
        if pole_start < 5:
            continue

        # ── BULL FLAG ─────────────────────────────────────────────────────
        for pole_len in range(pole_min, pole_max + 1):
            pole_end = pole_start + pole_len
            if pole_end >= n - flag_min - 2:
                break

            pole_move = close[pole_end] - close[pole_start]
            pole_pct  = pole_move / close[pole_start]

            # Pole must be strong upward move
            if pole_move < atr * 1.5 or pole_pct < 0.005:
                continue
            if not all(close[pole_start + k] < close[pole_start + k + 1]
                       for k in range(pole_len - 1)):
                continue

            # Flag: consolidation after pole
            for flag_len in range(flag_min, flag_max + 1):
                flag_end = pole_end + flag_len
                if flag_end >= n - 1:
                    break

                flag_highs = high[pole_end: flag_end + 1]
                flag_lows  = low[pole_end:  flag_end + 1]
                flag_vols  = volume[pole_end: flag_end + 1]

                # Flag: lower highs (slight pullback) + below avg volume
                lower_highs = flag_highs[-1] < flag_highs[0]
                low_vol     = float(np.mean(flag_vols)) < avg_vol * 1.0
                # Pullback not too deep — max 50% of pole
                pullback = close[pole_end] - min(flag_lows)
                not_too_deep = pullback < pole_move * 0.5

                if not (lower_highs and low_vol and not_too_deep):
                    continue

                # Breakout bar
                flag_high_price = float(max(flag_highs))
                breakout_bar    = flag_end + 1
                if breakout_bar >= n:
                    break

                broke_out = close[breakout_bar] > flag_high_price * 1.002
                vol_spike  = volume[breakout_bar] > avg_vol * 1.2

                if not (broke_out and vol_spike):
                    continue

                entry  = round(float(close[breakout_bar]) * 1.001, 2)
                stop   = round(float(min(flag_lows)) * 0.998, 2)
                risk   = entry - stop
                if risk <= 0:
                    continue

                # Measured move = pole height
                target = round(entry + pole_move, 2) if trade_style == "swing" else round(entry + atr * 1.2, 2)
                rr     = round((target - entry) / risk, 2)
                if rr < rr_min:
                    continue

                setups.append(Setup(
                    ticker=ticker, pattern="BullFlag",
                    direction="bullish",
                    entry_price=entry, stop_loss=stop,
                    target=target, rr_ratio=rr,
                    confirmed=True, neckline=flag_high_price,
                ))

        # ── BEAR FLAG ─────────────────────────────────────────────────────
        for pole_len in range(pole_min, pole_max + 1):
            pole_end = pole_start + pole_len
            if pole_end >= n - flag_min - 2:
                break

            pole_move = close[pole_start] - close[pole_end]
            pole_pct  = pole_move / close[pole_start]

            if pole_move < atr * 1.5 or pole_pct < 0.005:
                continue
            if not all(close[pole_start + k] > close[pole_start + k + 1]
                       for k in range(pole_len - 1)):
                continue

            for flag_len in range(flag_min, flag_max + 1):
                flag_end = pole_end + flag_len
                if flag_end >= n - 1:
                    break

                flag_highs = high[pole_end: flag_end + 1]
                flag_lows  = low[pole_end:  flag_end + 1]
                flag_vols  = volume[pole_end: flag_end + 1]

                higher_lows  = flag_lows[-1] > flag_lows[0]
                low_vol      = float(np.mean(flag_vols)) < avg_vol * 1.0
                pullback     = max(flag_highs) - close[pole_end]
                not_too_deep = pullback < pole_move * 0.5

                if not (higher_lows and low_vol and not_too_deep):
                    continue

                flag_low_price = float(min(flag_lows))
                breakout_bar   = flag_end + 1
                if breakout_bar >= n:
                    break

                broke_down = close[breakout_bar] < flag_low_price * 0.998
                vol_spike  = volume[breakout_bar] > avg_vol * 1.2

                if not (broke_down and vol_spike):
                    continue

                entry  = round(float(close[breakout_bar]) * 0.999, 2)
                stop   = round(float(max(flag_highs)) * 1.002, 2)
                risk   = stop - entry
                if risk <= 0:
                    continue

                target = round(entry - pole_move, 2) if trade_style == "swing" else round(entry - atr * 1.2, 2)
                rr     = round((entry - target) / risk, 2)
                if rr < rr_min:
                    continue

                setups.append(Setup(
                    ticker=ticker, pattern="BearFlag",
                    direction="bearish",
                    entry_price=entry, stop_loss=stop,
                    target=target, rr_ratio=rr,
                    confirmed=True, neckline=flag_low_price,
                ))

    return setups[-3:] if setups else []


def detect_opening_range_breakout(df: pd.DataFrame, ticker: str,
                                   rr_min: float = 1.5) -> List[Setup]:
    """
    Opening Range Breakout (ORB) — quick trade only.

    First 15-30 minutes sets the range (high and low).
    Breakout above range high with volume = CALL signal.
    Breakout below range low with volume = PUT signal.

    Uses first N bars as the opening range. Works best on 5min data.
    Requires at least 6 bars (30 min of 5min data).
    """
    setups = []
    if len(df) < 10:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    # Opening range = first 6 bars (30 min on 5min chart)
    orb_bars = min(6, len(df) // 4)
    orb_high = float(np.max(high[:orb_bars]))
    orb_low  = float(np.min(low[:orb_bars]))
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return setups

    avg_vol = float(np.mean(volume[-20:]))

    # ATR
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = float(pd.Series(tr).rolling(min(14, len(df))).mean().iloc[-1])
    if atr <= 0:
        atr = orb_range

    # Look for breakout bars after the ORB
    for i in range(orb_bars, len(close)):
        bar_vol = float(volume[i])
        vol_ok  = bar_vol > avg_vol * 1.2

        # Bullish breakout
        if close[i] > orb_high * 1.002 and vol_ok:
            entry  = round(float(close[i]) * 1.001, 2)
            stop   = round(orb_high - orb_range * 0.3, 2)
            risk   = entry - stop
            if risk <= 0:
                continue
            target = round(entry + orb_range * 1.5, 2)
            rr     = round((target - entry) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="ORBullish",
                direction="bullish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=orb_high,
            ))
            break  # one ORB signal per session

        # Bearish breakout
        elif close[i] < orb_low * 0.998 and vol_ok:
            entry  = round(float(close[i]) * 0.999, 2)
            stop   = round(orb_low + orb_range * 0.3, 2)
            risk   = stop - entry
            if risk <= 0:
                continue
            target = round(entry - orb_range * 1.5, 2)
            rr     = round((entry - target) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="ORBearish",
                direction="bearish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=orb_low,
            ))
            break

    return setups


def detect_momentum_continuation(df: pd.DataFrame, ticker: str,
                                  rr_min: float = 1.5) -> List[Setup]:
    """
    Momentum Continuation — quick trade only.

    3 consecutive closes in the same direction with expanding volume.
    Signals that institutional momentum is behind the move.
    Entry on the 4th bar in the same direction.
    """
    setups = []
    if len(df) < 10:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = float(pd.Series(tr).rolling(min(14, len(df))).mean().iloc[-1])
    if atr <= 0:
        atr = float(close[-1]) * 0.01

    avg_vol = float(np.mean(volume[-20:]))

    # Check last 3 bars for momentum
    for i in range(-4, -1):
        c1, c2, c3 = close[i - 2], close[i - 1], close[i]
        v1, v2, v3 = volume[i - 2], volume[i - 1], volume[i]

        # Bullish: 3 higher closes with expanding volume
        bull = c3 > c2 > c1 and v3 > v2 > v1 and v3 > avg_vol * 1.1
        # Bearish: 3 lower closes with expanding volume
        bear = c3 < c2 < c1 and v3 > v2 > v1 and v3 > avg_vol * 1.1

        if bull:
            entry  = round(float(c3) * 1.001, 2)
            stop   = round(float(c3) - atr * 0.5, 2)
            risk   = entry - stop
            if risk <= 0:
                continue
            target = round(entry + atr * 1.0, 2)
            rr     = round((target - entry) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="MomentumBull",
                direction="bullish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=float(c3),
            ))

        elif bear:
            entry  = round(float(c3) * 0.999, 2)
            stop   = round(float(c3) + atr * 0.5, 2)
            risk   = stop - entry
            if risk <= 0:
                continue
            target = round(entry - atr * 1.0, 2)
            rr     = round((entry - target) / risk, 2)
            if rr < rr_min:
                continue
            setups.append(Setup(
                ticker=ticker, pattern="MomentumBear",
                direction="bearish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=float(c3),
            ))

    return setups[-1:] if setups else []


# ═══════════════════════════════════════════════════════════════════════════════
# SWING TRADE PATTERNS (multi-day — 1hr/daily timeframes)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_ascending_descending_triangle(df: pd.DataFrame, ticker: str,
                                          rr_min: float = 2.0,
                                          window: int = 5) -> List[Setup]:
    """
    Ascending Triangle (bullish) / Descending Triangle (bearish) — swing only.

    Ascending: flat resistance + rising lows = bullish compression breakout.
    Descending: flat support + falling highs = bearish compression breakdown.

    Requires at least 2 touches of the flat level and a confirmed break.
    """
    setups = []
    if len(df) < 40:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values

    swing_highs = find_swing_highs(df, window)
    swing_lows  = find_swing_lows(df, window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return setups

    # ── ASCENDING TRIANGLE ────────────────────────────────────────────────
    # Find flat resistance: 2+ swing highs within 1% of each other
    for i in range(len(swing_highs) - 1):
        for j in range(i + 1, len(swing_highs)):
            h1_idx = swing_highs[i]
            h2_idx = swing_highs[j]
            h1     = high[h1_idx]
            h2     = high[h2_idx]

            if abs(h1 - h2) / max(h1, h2) > 0.01:
                continue
            if h2_idx - h1_idx < 8:
                continue

            resistance = (h1 + h2) / 2

            # Rising lows between the two highs
            lows_between = [l for l in swing_lows if h1_idx < l < h2_idx]
            if len(lows_between) < 1:
                continue
            first_low = low[lows_between[0]]
            last_low  = low[lows_between[-1]]
            if last_low <= first_low:
                continue  # need rising lows

            # Confirmed breakout above resistance
            post = close[h2_idx:]
            broke = any(c > resistance * 1.005 for c in post)
            if not broke:
                continue

            entry  = round(resistance * 1.003, 2)
            stop   = round(last_low * 0.99, 2)
            risk   = entry - stop
            if risk <= 0:
                continue
            height = resistance - first_low
            target = round(entry + height, 2)
            rr     = round((target - entry) / risk, 2)
            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker, pattern="AscTriangle",
                direction="bullish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=resistance,
            ))

    # ── DESCENDING TRIANGLE ───────────────────────────────────────────────
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            l1_idx = swing_lows[i]
            l2_idx = swing_lows[j]
            l1     = low[l1_idx]
            l2     = low[l2_idx]

            if abs(l1 - l2) / max(l1, l2) > 0.01:
                continue
            if l2_idx - l1_idx < 8:
                continue

            support = (l1 + l2) / 2

            highs_between = [h for h in swing_highs if l1_idx < h < l2_idx]
            if len(highs_between) < 1:
                continue
            first_high = high[highs_between[0]]
            last_high  = high[highs_between[-1]]
            if last_high >= first_high:
                continue  # need falling highs

            post  = close[l2_idx:]
            broke = any(c < support * 0.995 for c in post)
            if not broke:
                continue

            entry  = round(support * 0.997, 2)
            stop   = round(last_high * 1.01, 2)
            risk   = stop - entry
            if risk <= 0:
                continue
            height = first_high - support
            target = round(entry - height, 2)
            rr     = round((entry - target) / risk, 2)
            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker, pattern="DescTriangle",
                direction="bearish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=support,
            ))

    return setups[-3:] if setups else []


def detect_head_and_shoulders(df: pd.DataFrame, ticker: str,
                               rr_min: float = 2.0,
                               window: int = 5) -> List[Setup]:
    """
    Head & Shoulders (bearish) / Inverse H&S (bullish) — swing only.

    H&S: left shoulder high, higher head, lower right shoulder.
         Neckline break = entry.
    Inverse H&S: mirror image, bullish reversal.

    Requires confirmed neckline break with price below/above it.
    """
    setups = []
    if len(df) < 50:
        return setups

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values

    swing_highs = find_swing_highs(df, window)
    swing_lows  = find_swing_lows(df, window)

    # ── HEAD & SHOULDERS (bearish) ────────────────────────────────────────
    if len(swing_highs) >= 3:
        for i in range(len(swing_highs) - 2):
            ls_idx = swing_highs[i]       # left shoulder
            h_idx  = swing_highs[i + 1]   # head
            rs_idx = swing_highs[i + 2]   # right shoulder

            ls = high[ls_idx]
            h  = high[h_idx]
            rs = high[rs_idx]

            # Head must be highest, shoulders roughly equal
            if h <= ls or h <= rs:
                continue
            if abs(ls - rs) / max(ls, rs) > 0.03:
                continue
            if h_idx - ls_idx < 5 or rs_idx - h_idx < 5:
                continue

            # Neckline = average of the two troughs between shoulders and head
            troughs_left  = [l for l in swing_lows if ls_idx < l < h_idx]
            troughs_right = [l for l in swing_lows if h_idx  < l < rs_idx]
            if not troughs_left or not troughs_right:
                continue

            tl = low[troughs_left[-1]]
            tr = low[troughs_right[0]]
            neckline = (tl + tr) / 2

            # Confirmed break below neckline
            post  = close[rs_idx:]
            broke = any(c < neckline * 0.998 for c in post)
            if not broke:
                continue

            entry  = round(neckline * 0.997, 2)
            stop   = round(rs * 1.01, 2)
            risk   = stop - entry
            if risk <= 0:
                continue
            height = h - neckline
            target = round(entry - height, 2)
            rr     = round((entry - target) / risk, 2)
            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker, pattern="HeadShoulders",
                direction="bearish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=neckline,
            ))

    # ── INVERSE HEAD & SHOULDERS (bullish) ────────────────────────────────
    if len(swing_lows) >= 3:
        for i in range(len(swing_lows) - 2):
            ls_idx = swing_lows[i]
            h_idx  = swing_lows[i + 1]
            rs_idx = swing_lows[i + 2]

            ls = low[ls_idx]
            h  = low[h_idx]
            rs = low[rs_idx]

            if h >= ls or h >= rs:
                continue
            if abs(ls - rs) / max(ls, rs) > 0.03:
                continue
            if h_idx - ls_idx < 5 or rs_idx - h_idx < 5:
                continue

            peaks_left  = [p for p in swing_highs if ls_idx < p < h_idx]
            peaks_right = [p for p in swing_highs if h_idx  < p < rs_idx]
            if not peaks_left or not peaks_right:
                continue

            pl = high[peaks_left[-1]]
            pr = high[peaks_right[0]]
            neckline = (pl + pr) / 2

            post  = close[rs_idx:]
            broke = any(c > neckline * 1.002 for c in post)
            if not broke:
                continue

            entry  = round(neckline * 1.003, 2)
            stop   = round(rs * 0.99, 2)
            risk   = entry - stop
            if risk <= 0:
                continue
            height = neckline - h
            target = round(entry + height, 2)
            rr     = round((target - entry) / risk, 2)
            if rr < rr_min:
                continue

            setups.append(Setup(
                ticker=ticker, pattern="InvHeadShoulders",
                direction="bullish",
                entry_price=entry, stop_loss=stop,
                target=target, rr_ratio=rr,
                confirmed=True, neckline=neckline,
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
