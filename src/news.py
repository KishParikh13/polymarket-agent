"""
News Context Fetcher (Phase 2) — fetches recent headlines for a market topic.

Uses Brave Search API (if BRAVE_API_KEY is set) for high-quality results,
with DuckDuckGo HTML search as fallback (no API key required).
Headlines are passed to the LLM to improve forecasting accuracy.
"""

import logging
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Simple in-memory cache to avoid hammering search APIs for the same query
_cache: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _extract_query(question: str) -> str:
    """Distil market question into a concise search query (max 80 chars)."""
    q = re.sub(r"\?$", "", question).strip()
    q = re.sub(r"\b(will|who|what|when|does|did|is|are|has|have|be)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q[:80]


def _fetch_brave(query: str, max_results: int) -> list[str]:
    """Fetch headlines from Brave Search API."""
    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/news/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": f"{query} news",
                "count": max_results,
                "freshness": "pw",   # past week
                "text_decorations": "false",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        headlines: list[str] = []
        for item in data.get("results", [])[:max_results]:
            title = item.get("title", "").strip()
            if title and len(title) > 10:
                headlines.append(title)
        logger.info(f"Brave search: {len(headlines)} headlines for '{query[:60]}'")
        return headlines
    except Exception as e:
        logger.warning(f"Brave search failed for '{query[:60]}': {e}")
        return []


def _fetch_ddg(query: str, max_results: int) -> list[str]:
    """Fallback: DuckDuckGo HTML search (no API key required)."""
    try:
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
        return _parse_ddg_headlines(resp.text, max_results)
    except Exception as e:
        logger.warning(f"DDG search failed for '{query[:60]}': {e}")
        return []


def _parse_ddg_headlines(html: str, max_results: int) -> list[str]:
    """Extract result titles from DuckDuckGo HTML response."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        headlines: list[str] = []
        for tag in soup.select("a.result__a, h2.result__title a"):
            text = tag.get_text(strip=True)
            if text and len(text) > 10:
                headlines.append(text)
            if len(headlines) >= max_results:
                break
        return headlines
    except ImportError:
        matches = re.findall(r'class="result__a"[^>]*>([^<]{10,})<', html)
        return matches[:max_results]
    except Exception as e:
        logger.debug(f"DDG headline parse error: {e}")
        return []


def fetch_headlines(question: str, max_results: int = 5) -> list[str]:
    """
    Fetch recent news headlines relevant to a Polymarket question.

    Prefers Brave Search API (BRAVE_API_KEY env var) → falls back to DuckDuckGo.

    Args:
        question:    The market question.
        max_results: Max headlines to return (default 5).

    Returns:
        A list of headline strings, possibly empty if all fetches fail.
    """
    query = _extract_query(question)
    cache_key = query.lower()

    # Return cached result if fresh
    if cache_key in _cache:
        ts, headlines = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            logger.debug(f"News cache hit: {query[:60]}")
            return headlines

    # Try Brave first (better quality, respects freshness)
    headlines = _fetch_brave(query, max_results)

    # Fallback to DuckDuckGo
    if not headlines:
        headlines = _fetch_ddg(query, max_results)

    _cache[cache_key] = (time.time(), headlines)
    return headlines
