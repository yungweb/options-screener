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
    return 50.0, 50.0, 0.30

def get_options_recommendation(setup, account_size=10000, risk_per_trade_pct=0.01):
    direction = getattr(setup, "direction", "bullish")
    iv_rank, iv_pct, current_iv = fetch_iv_rank(setup.ticker)
    if iv_rank > 60:
        strategy = "Bull Put Spread" if direction=="bullish" else "Bear Call Spread"
    elif iv_rank < 30:
        strategy = "Long Call" if direction=="bullish" else "Long Put"
    else:
        strategy = "Bull Call Spread" if direction=="bullish" else "Bear Put Spread"
    return OptionsRecommendation(ticker=setup.ticker, underlying_price=setup.entry_price, iv_rank=iv_rank, iv_percentile=iv_pct, current_iv=current_iv, strategy=strategy, direction=direction, setup_entry=setup.entry_price, setup_stop=setup.stop_loss, setup_target=setup.target)

def print_recommendation(rec):
    print(f"Strategy: {rec.strategy}")
    print(f"IV Rank: {rec.iv_rank}%")
