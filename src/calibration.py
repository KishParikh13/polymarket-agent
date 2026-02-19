"""
Calibration Engine (Phase 1) — Brier score, calibration curve, ROI by category.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from config import CALIBRATION_BUCKETS, DB_PATH
from src.db import get_conn

logger = logging.getLogger(__name__)


def _get_resolved_trades() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT model_probability, confidence, actual_outcome,
               category, simulated_amount, profit_loss, implied_probability
        FROM trades
        WHERE resolved = 1 AND actual_outcome IS NOT NULL
        ORDER BY timestamp
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def brier_score(trades: list[dict]) -> float | None:
    """Lower is better. 0.25 = random, 0.0 = perfect."""
    if not trades:
        return None
    return sum(
        (t["model_probability"] - t["actual_outcome"]) ** 2
        for t in trades
    ) / len(trades)


def calibration_curve(trades: list[dict]) -> list[dict[str, float]]:
    """
    Bucket predictions into deciles; compute predicted vs actual win rate per bucket.
    Returns list of {bucket_low, bucket_high, predicted_mean, actual_mean, count}.
    """
    buckets: list[list[dict]] = [[] for _ in range(CALIBRATION_BUCKETS)]
    for t in trades:
        idx = min(int(t["model_probability"] * CALIBRATION_BUCKETS), CALIBRATION_BUCKETS - 1)
        buckets[idx].append(t)

    result = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append({
            "bucket_low":     i / CALIBRATION_BUCKETS,
            "bucket_high":    (i + 1) / CALIBRATION_BUCKETS,
            "predicted_mean": sum(t["model_probability"] for t in bucket) / len(bucket),
            "actual_mean":    sum(t["actual_outcome"]    for t in bucket) / len(bucket),
            "count":          len(bucket),
        })
    return result


def roi_by_category(trades: list[dict]) -> dict[str, dict[str, float]]:
    """ROI and win-rate broken down by market category."""
    cats: dict[str, list[dict]] = {}
    for t in trades:
        cats.setdefault(t["category"], []).append(t)

    result = {}
    for cat, ts in cats.items():
        wagered = sum(t["simulated_amount"] for t in ts)
        pnl     = sum(t["profit_loss"]     for t in ts)
        wins    = sum(1 for t in ts if t["profit_loss"] > 0)
        result[cat] = {
            "count":     len(ts),
            "wagered":   wagered,
            "pnl":       pnl,
            "roi_pct":   (pnl / wagered * 100) if wagered else 0.0,
            "win_rate":  (wins / len(ts))       if ts      else 0.0,
        }
    return result


def max_drawdown(trades: list[dict]) -> float:
    """Maximum peak-to-trough drop in cumulative P&L."""
    if not trades:
        return 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t["profit_loss"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def edge_vs_market(trades: list[dict]) -> dict[str, float]:
    """
    Compare model win-rate against market's implied win-rate.
    A positive delta means you're genuinely beating the market prior.
    """
    if not trades:
        return {}
    wins       = sum(1 for t in trades if t["profit_loss"] > 0)
    win_rate   = wins / len(trades)
    avg_implied = sum(t["implied_probability"] for t in trades) / len(trades)
    return {
        "win_rate":         win_rate,
        "avg_implied_prob": avg_implied,
        "delta":            win_rate - avg_implied,
    }


def generate_report() -> dict[str, Any]:
    """Generate a full calibration report and save a snapshot to DB."""
    trades = _get_resolved_trades()
    if not trades:
        return {"error": "No resolved trades yet. Run the agent and let markets resolve."}

    bs   = brier_score(trades)
    curve = calibration_curve(trades)
    roi   = roi_by_category(trades)
    dd    = max_drawdown(trades)
    edge  = edge_vs_market(trades)

    total_wagered = sum(t["simulated_amount"] for t in trades)
    total_pnl     = sum(t["profit_loss"]     for t in trades)

    report: dict[str, Any] = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "total_trades":     len(trades),
        "total_wagered":    total_wagered,
        "total_pnl":        total_pnl,
        "roi_pct":          (total_pnl / total_wagered * 100) if total_wagered else 0.0,
        "brier_score":      bs,
        "max_drawdown":     dd,
        "edge_vs_market":   edge,
        "calibration_curve": curve,
        "roi_by_category":  roi,
    }

    # Persist snapshot
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO calibration_snapshots
          (timestamp, brier_score, total_trades, win_rate, roi, max_drawdown, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report["generated_at"],
            bs,
            len(trades),
            edge.get("win_rate", 0.0),
            report["roi_pct"],
            dd,
            json.dumps(report),
        ),
    )
    conn.commit()
    conn.close()

    return report
