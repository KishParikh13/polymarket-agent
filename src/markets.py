"""
Market Fetcher — pulls active markets from Polymarket Gamma API.
No API key required (public endpoint).
"""

import requests
import logging
from datetime import datetime, timezone
from typing import Any

from config import GAMMA_API_BASE, MIN_VOLUME, MAX_MARKETS_PER_RUN

logger = logging.getLogger(__name__)


def fetch_active_markets() -> list[dict[str, Any]]:
    """
    Fetch active binary YES/NO markets above the volume threshold.
    Returns a list of dicts with standardised keys.
    """
    url = f"{GAMMA_API_BASE}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": 100,
        "order": "volume",
        "ascending": "false",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw: list[dict] = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        return []

    markets: list[dict[str, Any]] = []
    for m in raw:
        try:
            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
            if volume < MIN_VOLUME:
                continue

            # Only binary markets (two outcomes: YES and NO)
            outcomes = m.get("outcomes", [])
            if isinstance(outcomes, str):
                import json as _json
                outcomes = _json.loads(outcomes)
            if len(outcomes) != 2:
                continue

            # Parse prices
            outcome_prices = m.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                import json as _json
                outcome_prices = _json.loads(outcome_prices)
            if len(outcome_prices) != 2:
                continue

            yes_price = float(outcome_prices[0])
            no_price  = float(outcome_prices[1])

            markets.append({
                "id":        m.get("id") or m.get("conditionId", ""),
                "slug":      m.get("slug", ""),
                "question":  m.get("question", ""),
                "yes_price": yes_price,
                "no_price":  no_price,
                "volume":    volume,
                "end_date":  m.get("endDate") or m.get("endDateIso", ""),
                "raw":       m,
            })

            if len(markets) >= MAX_MARKETS_PER_RUN:
                break

        except Exception as e:
            logger.debug(f"Skipping malformed market: {e}")
            continue

    logger.info(f"Fetched {len(markets)} qualifying markets")
    return markets
