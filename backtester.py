<<<<<<< HEAD
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
=======
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
>>>>>>> 6ec97029 (initial upload)
