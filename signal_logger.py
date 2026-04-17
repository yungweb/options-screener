import os
import requests
from datetime import datetime, timezone, date, timedelta
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FMP_API_KEY  = os.environ.get("FMP_API_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def log_signal(
    ticker,
    signal_type,
    conviction_score,
    signals_triggered,
    entry_price,
    stop_price,
    target_price,
    pattern_type=None,
    sector=None,
    market_regime=None,
    vix_at_signal=None,
    spy_trend=None,
    paper_trade=True,
):
    """Call this every time a GO NOW signal fires. Returns the inserted row id."""
    now = datetime.now(timezone.utc)

    row = {
        "ticker":            ticker,
        "signal_date":       now.date().isoformat(),
        "signal_time":       now.isoformat(),
        "signal_type":       signal_type,
        "conviction_score":  conviction_score,
        "signals_triggered": signals_triggered,
        "entry_price":       entry_price,
        "stop_price":        stop_price,
        "target_price":      target_price,
        "pattern_type":      pattern_type,
        "sector":            sector,
        "market_regime":     market_regime,
        "vix_at_signal":     vix_at_signal,
        "spy_trend":         spy_trend,
        "paper_trade":       paper_trade,
    }

    result = supabase.table("signal_outcomes").insert(row).execute()
    return result.data[0]["id"] if result.data else None


def _fmp_closes(ticker, from_date, days=7):
    """
    Fetch daily closes from FMP starting at from_date.
    Returns a list of closing prices in chronological order.
    """
    if not FMP_API_KEY:
        return []

    to_date = (date.fromisoformat(from_date) + timedelta(days=days + 5)).isoformat()

    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/full"
        "?symbol=%s&from=%s&to=%s&apikey=%s"
        % (ticker.upper(), from_date, to_date, FMP_API_KEY)
    )

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        # FMP returns newest first — reverse so index 0 = signal date
        closes = [
            {"date": d["date"], "close": d["close"], "high": d["high"], "low": d["low"]}
            for d in reversed(data)
            if d.get("date") >= from_date
        ]
        return closes
    except Exception:
        return []


def fetch_and_log_outcomes():
    """
    Run this at market close (4:15 PM ET).
    Finds all signals missing outcome data and fills them in using FMP.
    """
    today = date.today()

    result = (
        supabase.table("signal_outcomes")
        .select("*")
        .is_("outcome_logged_at", "null")
        .lte("signal_date", (today - timedelta(days=1)).isoformat())
        .execute()
    )

    rows = result.data
    if not rows:
        print("No pending signals to log outcomes for.")
        return

    print("Logging outcomes for %s signals..." % len(rows))

    for row in rows:
        ticker      = row["ticker"]
        signal_date = row["signal_date"]
        entry_price = row["entry_price"]

        if not entry_price:
            continue

        try:
            closes = _fmp_closes(ticker, signal_date, days=7)

            if not closes:
                print("  ✗ %s — no FMP data" % ticker)
                continue

            # Index 0 = signal date, index 1 = next trading day, etc.
            price_1d = closes[1]["close"] if len(closes) > 1 else None
            price_3d = closes[3]["close"] if len(closes) > 3 else None
            price_5d = closes[5]["close"] if len(closes) > 5 else None

            def pct(p):
                if p is None:
                    return None
                return round(((p - entry_price) / entry_price) * 100, 2)

            def outcome(p):
                if p is None:
                    return None
                move = (p - entry_price) / entry_price
                if move >= 0.02:
                    return "WIN"
                elif move <= -0.02:
                    return "LOSS"
                return "NEUTRAL"

            target = row.get("target_price")
            stop   = row.get("stop_price")

            highs = [c["high"] for c in closes[1:6]]
            lows  = [c["low"]  for c in closes[1:6]]

            hit_target = any(h >= target for h in highs) if target else None
            hit_stop   = any(l <= stop   for l in lows)  if stop   else None

            update = {
                "price_1d":          price_1d,
                "price_3d":          price_3d,
                "price_5d":          price_5d,
                "pct_move_1d":       pct(price_1d),
                "pct_move_3d":       pct(price_3d),
                "pct_move_5d":       pct(price_5d),
                "outcome_1d":        outcome(price_1d),
                "outcome_3d":        outcome(price_3d),
                "outcome_5d":        outcome(price_5d),
                "hit_target":        hit_target,
                "hit_stop":          hit_stop,
                "outcome_logged_at": datetime.now(timezone.utc).isoformat(),
            }

            supabase.table("signal_outcomes").update(update).eq("id", row["id"]).execute()

            print("  ✓ %s | 1d: %s%% | 3d: %s%% | 5d: %s%%" % (
                ticker, pct(price_1d), pct(price_3d), pct(price_5d)
            ))

        except Exception as e:
            print("  ✗ %s error: %s" % (ticker, e))
            continue

    print("Outcome logging complete.")


def get_signal_accuracy(days_back=30):
    """
    Returns win rate stats. Call anytime you want a performance snapshot.
    """
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()

    result = (
        supabase.table("signal_outcomes")
        .select("signal_type, outcome_1d, outcome_3d, outcome_5d, conviction_score, pattern_type")
        .gte("signal_date", cutoff)
        .not_.is_("outcome_1d", "null")
        .execute()
    )

    rows = result.data
    if not rows:
        return {}

    total    = len(rows)
    wins_1d  = sum(1 for r in rows if r["outcome_1d"] == "WIN")
    wins_3d  = sum(1 for r in rows if r["outcome_3d"] == "WIN")
    wins_5d  = sum(1 for r in rows if r["outcome_5d"] == "WIN")

    return {
        "total_signals":      total,
        "win_rate_1d":        round(wins_1d / total * 100, 1),
        "win_rate_3d":        round(wins_3d / total * 100, 1),
        "win_rate_5d":        round(wins_5d / total * 100, 1),
        "sample_period_days": days_back,
    }
