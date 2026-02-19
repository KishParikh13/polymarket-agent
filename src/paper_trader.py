"""
Paper Trading Engine — simulates bets with no real money.
Uses conservative Kelly sizing.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from config import (
    STARTING_BANKROLL,
    MAX_BET_FRACTION,
    KELLY_MULTIPLIER,
)
from src.db import get_conn

logger = logging.getLogger(__name__)


def _get_bankroll(conn: sqlite3.Connection) -> float:
    """
    Compute current bankroll:
      starting + sum of all resolved P&L - sum of all open bet amounts
    """
    row = conn.execute(
        "SELECT SUM(profit_loss) FROM trades WHERE resolved = 1"
    ).fetchone()
    realised_pnl = row[0] or 0.0

    row = conn.execute(
        "SELECT SUM(simulated_amount) FROM trades WHERE resolved = 0"
    ).fetchone()
    locked = row[0] or 0.0

    return STARTING_BANKROLL + realised_pnl - locked


def kelly_size(bankroll: float, confidence: float) -> float:
    """
    Conservative Kelly: bet_fraction = (confidence - 0.5) * KELLY_MULTIPLIER
    Capped at MAX_BET_FRACTION.
    """
    fraction = max(0, (confidence - 0.5) * KELLY_MULTIPLIER)
    fraction = min(fraction, MAX_BET_FRACTION)
    return round(bankroll * fraction, 2)


def place_paper_bet(
    market: dict[str, Any],
    signal: dict[str, Any],
    category: str,
) -> int | None:
    """
    Log a simulated bet to the DB.
    Returns the trade ID or None if bet size is too small.
    """
    conn = get_conn()
    bankroll = _get_bankroll(conn)
    amount = kelly_size(bankroll, signal["confidence"])

    if amount < 1.0:
        logger.info(f"Bet too small ({amount:.2f}) — skipping: {market['question'][:60]}")
        conn.close()
        return None

    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO trades
          (market_id, question, category,
           model_probability, confidence, implied_probability,
           bet_direction, simulated_amount, bankroll_at_bet, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market["id"],
            market["question"],
            category,
            signal["probability"],
            signal["confidence"],
            signal["implied_prob"],
            signal["bet_direction"],
            amount,
            bankroll,
            now,
        ),
    )
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()

    logger.info(
        f"Paper bet: ${amount:.2f} {signal['bet_direction']} | "
        f"prob={signal['probability']:.2f} conf={signal['confidence']:.2f} | "
        f"{market['question'][:60]}"
    )
    return trade_id


def get_open_positions() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE resolved = 0 ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_portfolio_summary() -> dict[str, Any]:
    conn = get_conn()
    bankroll = _get_bankroll(conn)

    open_count = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE resolved = 0"
    ).fetchone()[0]

    closed = conn.execute(
        "SELECT COUNT(*), SUM(profit_loss), SUM(simulated_amount) "
        "FROM trades WHERE resolved = 1"
    ).fetchone()
    closed_count   = closed[0] or 0
    total_pnl      = closed[1] or 0.0
    total_wagered  = closed[2] or 0.0

    roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0

    conn.close()
    return {
        "bankroll":      bankroll,
        "open_bets":     open_count,
        "closed_bets":   closed_count,
        "total_pnl":     total_pnl,
        "total_wagered": total_wagered,
        "roi_pct":       roi,
    }
