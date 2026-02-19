"""
News Context Fetcher (Phase 2) — fetches recent headlines for a market topic.

Uses DuckDuckGo HTML search (no API key required) to find recent news.
Headlines are passed to the LLM to improve forecasting accuracy.
"""

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Simple in-memory cache to avoid hammering DDG for the same query
_cache: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _extract_query(question: str) -> str:
    """Distil market question into a concise search query (max 80 chars)."""
    # Strip trailing punctuation and common filler phrases
    q = re.sub(r"\?$", "", question).strip()
    q = re.sub(r"\b(will|who|what|when|does|did|is|are|has|have|be)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q[:80]


def fetch_headlines(question: str, max_results: int = 5) -> list[str]:
    """
    Fetch recent news headlines relevant to a Polymarket question.

    Args:
        question:    The market question (e.g. "Will Biden win the 2024 election?").
        max_results: Max headlines to return (default 5).

    Returns:
        A list of headline strings, possibly empty if fetch fails.
    """
    query = _extract_query(question)
    cache_key = query.lower()

    # Return cached result if fresh
    if cache_key in _cache:
        ts, headlines = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            logger.debug(f"News cache hit: {query[:60]}")
            return headlines

    try:
        # DuckDuckGo HTML endpoint — no API key, no rate-limit headers needed
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f"{query} news", "ia": "news"},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"News fetch failed for '{query[:60]}': {e}")
        return []

    headlines = _parse_headlines(resp.text, max_results)
    _cache[cache_key] = (time.time(), headlines)
    logger.info(f"Fetched {len(headlines)} headlines for: {query[:60]}")
    return headlines


def _parse_headlines(html: str, max_results: int) -> list[str]:
    """Extract result titles from DuckDuckGo HTML response."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        headlines: list[str] = []
        # DDG HTML results: <a class="result__a">Title</a>
        for tag in soup.select("a.result__a, h2.result__title a"):
            text = tag.get_text(strip=True)
            if text and len(text) > 10:  # Skip noise
                headlines.append(text)
            if len(headlines) >= max_results:
                break
        return headlines

    except ImportError:
        # Fallback: naive regex extraction if bs4 not available
        matches = re.findall(r'class="result__a"[^>]*>([^<]{10,})<', html)
        return matches[:max_results]
    except Exception as e:
        logger.debug(f"Headline parse error: {e}")
        return []
