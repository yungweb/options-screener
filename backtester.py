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