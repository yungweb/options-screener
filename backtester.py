"""
Backtester - tests double bottom, double top, and break & retest patterns
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from pattern_detection import detect_double_bottom, detect_double_top, detect_break_and_retest


@dataclass
class Trade:
    pattern: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    outcome: str  # 'win' or 'loss'
    pnl_pct: float
    rr_ratio: float


@dataclass
class BacktestReport:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_rr: float
    expectancy: float
    trades: List[Trade]


def run_backtest(df: pd.DataFrame, ticker: str) -> Tuple[BacktestReport, List[float]]:
    """Run backtest on historical data for all patterns."""
    all_setups = []

    # Detect all patterns
    try:
        db = detect_double_bottom(df, ticker, rr_min=1.5)
        all_setups.extend(db)
    except:
        pass
    try:
        dt = detect_double_top(df, ticker, rr_min=1.5)
        all_setups.extend(dt)
    except:
        pass
    try:
        br = detect_break_and_retest(df, ticker, rr_min=1.5)
        all_setups.extend(br)
    except:
        pass

    trades = []
    equity = [0.0]
    close = df["close"].values

    for setup in all_setups:
        if not setup.confirmed:
            continue

        entry = setup.entry_price
        stop = setup.stop_loss
        target = setup.target
        risk = abs(entry - stop)

        if risk <= 0:
            continue

        # Simulate: check if price hit target or stop after entry
        # Use last 20 bars as the "future" for simulation
        future_prices = close[-20:] if len(close) >= 20 else close

        hit_target = False
        hit_stop = False

        for price in future_prices:
            if setup.direction == "bullish":
                if price >= target:
                    hit_target = True
                    break
                if price <= stop:
                    hit_stop = True
                    break
            else:  # bearish
                if price <= target:
                    hit_target = True
                    break
                if price >= stop:
                    hit_stop = True
                    break

        # Determine outcome
        if hit_target:
            outcome = "win"
            exit_price = target
            pnl_pct = round(abs(target - entry) / entry * 100, 1)
            rr = round(abs(target - entry) / risk, 2)
            equity.append(equity[-1] + rr)
        elif hit_stop:
            outcome = "loss"
            exit_price = stop
            pnl_pct = round(-abs(stop - entry) / entry * 100, 1)
            rr = setup.rr_ratio
            equity.append(equity[-1] - 1.0)
        else:
            # Neither hit — skip this trade
            continue

        trades.append(Trade(
            pattern=setup.pattern,
            direction=setup.direction,
            entry_price=entry,
            exit_price=exit_price,
            stop_loss=stop,
            target=target,
            outcome=outcome,
            pnl_pct=pnl_pct,
            rr_ratio=setup.rr_ratio,
        ))

    # Build report
    if not trades:
        return BacktestReport(
            total_trades=0, wins=0, losses=0,
            win_rate=0.0, avg_rr=0.0, expectancy=0.0, trades=[]
        ), [0.0]

    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    win_rate = round(len(wins) / len(trades) * 100, 1)
    avg_rr = round(np.mean([t.rr_ratio for t in trades]), 2)

    # Expectancy = (win_rate * avg_win_R) - (loss_rate * 1)
    win_rate_dec = len(wins) / len(trades)
    loss_rate_dec = len(losses) / len(trades)
    avg_win_r = round(np.mean([t.rr_ratio for t in wins]), 2) if wins else 0
    expectancy = round((win_rate_dec * avg_win_r) - (loss_rate_dec * 1), 2)

    report = BacktestReport(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        avg_rr=avg_rr,
        expectancy=expectancy,
        trades=trades,
    )

    return report, equity
