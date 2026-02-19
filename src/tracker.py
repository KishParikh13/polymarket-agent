"""
Outcome Tracker — resolves open trades against actual market outcomes.
"""

import logging
import requests
from datetime import datetime, timezone
from typing import Any

from config import GAMMA_API_BASE, ROUND_TRIP_COST_PCT
from src.db import get_conn

logger = logging.getLogger(__name__)


def _fetch_market_result(market_id: str) -> float | None:
    """
    Returns 1.0 if YES resolved, 0.0 if NO resolved, None if unresolved.
    """
    try:
        url = f"{GAMMA_API_BASE}/markets/{market_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Check if resolved
        if not data.get("closed") and not data.get("resolved"):
            return None

        outcome_prices = data.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            import json
            outcome_prices = json.loads(outcome_prices)

        if len(outcome_prices) == 2:
            yes_price = float(outcome_prices[0])
            # If YES price is ~1.0, YES won; if ~0.0, NO won
            if yes_price > 0.95:
                return 1.0
            elif yes_price < 0.05:
                return 0.0

        return None

    except Exception as e:
        logger.debug(f"Could not resolve market {market_id}: {e}")
        return None


def resolve_open_trades() -> int:
    """
    Check all open trades and resolve any that have outcomes.
    Returns the number of newly resolved trades.
    """
    conn = get_conn()
    open_trades = conn.execute(
        "SELECT * FROM trades WHERE resolved = 0"
    ).fetchall()

    resolved_count = 0
    for trade in open_trades:
        t = dict(trade)
        actual_outcome = _fetch_market_result(t["market_id"])

        if actual_outcome is None:
            continue

        # Compute P&L
        direction = t["bet_direction"]
        amount    = t["simulated_amount"]
        cost      = amount * ROUND_TRIP_COST_PCT

        if direction == "YES":
            payout = amount / t["implied_probability"] if actual_outcome == 1.0 else 0.0
        else:  # NO
            no_implied = 1 - t["implied_probability"]
            payout = amount / no_implied if actual_outcome == 0.0 else 0.0

        profit_loss = payout - amount - cost

        conn.execute(
            """
            UPDATE trades
            SET resolved = 1, actual_outcome = ?, profit_loss = ?
            WHERE id = ?
            """,
            (actual_outcome, profit_loss, t["id"]),
        )
        resolved_count += 1
        logger.info(
            f"Resolved trade #{t['id']}: outcome={'YES' if actual_outcome else 'NO'} "
            f"P&L=${profit_loss:+.2f} | {t['question'][:60]}"
        )

    conn.commit()
    conn.close()
    return resolved_count
