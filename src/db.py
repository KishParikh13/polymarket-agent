"""
Database initialization and helpers.
"""

import sqlite3
import os
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id           TEXT NOT NULL,
            question            TEXT NOT NULL,
            category            TEXT DEFAULT 'other',
            model_probability   REAL NOT NULL,
            confidence          REAL NOT NULL,
            implied_probability REAL NOT NULL,
            bet_direction       TEXT NOT NULL,          -- 'YES' or 'NO'
            simulated_amount    REAL NOT NULL,
            bankroll_at_bet     REAL NOT NULL,
            timestamp           TEXT NOT NULL,
            resolved            INTEGER DEFAULT 0,      -- 0=open, 1=resolved
            actual_outcome      REAL,                   -- 1.0=YES won, 0.0=NO won
            profit_loss         REAL
        );

        CREATE TABLE IF NOT EXISTS calibration_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            brier_score  REAL,
            total_trades INTEGER,
            win_rate     REAL,
            roi          REAL,
            max_drawdown REAL,
            report_json  TEXT
        );
    """)
    conn.commit()
    conn.close()
