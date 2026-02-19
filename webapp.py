"""
webapp.py — Polymarket AI Viewer
Lightweight FastAPI server: live markets + on-demand AI analysis.
No trading. No persistence. Just the signal.

Usage:
  python webapp.py              # starts on http://localhost:8765
  PORT=9000 python webapp.py   # custom port
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Setup ──────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Polymarket AI Viewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Market fetch ───────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"

def _parse_prices(prices_raw) -> tuple[float, float] | None:
    """Return (yes_price, no_price) or None if invalid."""
    try:
        if isinstance(prices_raw, str):
            prices_raw = json.loads(prices_raw)
        if len(prices_raw) != 2:
            return None
        y, n = float(prices_raw[0]), float(prices_raw[1])
        return (y, n)
    except Exception:
        return None


def _days_left(end_date_str: str) -> int | None:
    if not end_date_str:
        return None
    try:
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return max(0, (dt - datetime.now(timezone.utc)).days)
    except Exception:
        return None


def _guess_category(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["bitcoin", "eth", "crypto", "btc", "solana", "coin", "token", "blockchain", "defi", "nft"]):
        return "Crypto"
    if any(k in q for k in ["president", "election", "congress", "senate", "trump", "biden", "harris", "vote", "democrat", "republican", "political", "governor", "mayor"]):
        return "Politics"
    if any(k in q for k in ["war", "ceasefire", "nato", "ukraine", "russia", "china", "israel", "hamas", "sanction", "treaty", "geopolit", "military"]):
        return "Geopolitics"
    if any(k in q for k in ["fed", "rate", "inflation", "gdp", "unemployment", "recession", "market cap", "tariff", "trade", "economy", "interest"]):
        return "Economics"
    if any(k in q for k in ["nfl", "nba", "mlb", "nhl", "soccer", "tennis", "fifa", "super bowl", "world cup", "championship", "playoff", "match", "game", "tournament"]):
        return "Sports"
    if any(k in q for k in ["ipo", "merger", "acquisition", "stock", "earnings", "revenue", "ceo", "company", "startup", "valuation"]):
        return "Business"
    if any(k in q for k in ["ai", "openai", "gpt", "anthropic", "gemini", "llm", "tech", "apple", "google", "meta", "microsoft", "amazon", "spacex", "tesla"]):
        return "Tech"
    return "Other"


@app.get("/api/markets")
async def get_markets(
    limit: int = 200,
    min_volume: float = 500.0,
    max_days: int = 90,
):
    """Fetch live Polymarket markets. No auth required (public API)."""
    markets = []
    offset = 0
    batch_size = 100

    cutoff = datetime.now(timezone.utc) + timedelta(days=max_days)

    while len(markets) < limit:
        try:
            resp = requests.get(f"{GAMMA_API}/markets", params={
                "closed":    "false",
                "active":    "true",
                "limit":     batch_size,
                "offset":    offset,
                "order":     "volume",
                "ascending": "false",
            }, timeout=20)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
            raise HTTPException(status_code=502, detail=f"Polymarket API error: {e}")

        if not batch:
            break

        for m in batch:
            try:
                volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
                if volume < min_volume:
                    continue

                prices = _parse_prices(m.get("outcomePrices", []))
                if prices is None:
                    continue
                yes_price, no_price = prices

                # Only binary markets in reasonable price range (not already resolved)
                if not (0.02 < yes_price < 0.98):
                    continue
                if abs(yes_price + no_price - 1.0) > 0.08:
                    continue

                question = (m.get("question") or "").strip()
                if not question or len(question) < 8:
                    continue

                end_date_str = m.get("endDate") or m.get("endDateIso") or ""
                days = _days_left(end_date_str)

                # Skip already-closed or too far out
                if days is not None and end_date_str:
                    try:
                        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        if dt > cutoff:
                            continue
                        if dt < datetime.now(timezone.utc):
                            continue
                    except Exception:
                        pass

                markets.append({
                    "id":        m.get("id") or m.get("conditionId", ""),
                    "slug":      m.get("slug", ""),
                    "question":  question,
                    "yes_price": round(yes_price, 4),
                    "no_price":  round(no_price, 4),
                    "volume":    round(volume, 2),
                    "end_date":  end_date_str,
                    "days_left": days,
                    "category":  _guess_category(question),
                    "image":     m.get("image", ""),
                })

                if len(markets) >= limit:
                    break

            except Exception:
                continue

        offset += batch_size
        if len(batch) < batch_size:
            break

    return {"markets": markets, "count": len(markets)}


# ── Analysis endpoint ──────────────────────────────────────────────────────────
class MarketForAnalysis(BaseModel):
    id: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    end_date: str = ""
    days_left: int | None = None
    category: str = ""


@app.post("/api/analyze")
async def analyze_market(market: MarketForAnalysis):
    """Run AI analysis on a single market. Returns probability, confidence, reasoning, and signal."""
    from config import ANTHROPIC_API_KEY, PRIMARY_MODEL, USE_NEWS_CONTEXT
    import config

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # Temporarily disable news & consensus for speed in the UI
    orig_news     = config.USE_NEWS_CONTEXT
    orig_consensus = config.CONSENSUS_MODE
    config.USE_NEWS_CONTEXT = False
    config.CONSENSUS_MODE   = False

    try:
        from src.forecaster import _ask_model, SYSTEM_PROMPT

        market_dict = {
            "id":        market.id,
            "question":  market.question,
            "yes_price": market.yes_price,
            "no_price":  market.no_price,
            "volume":    market.volume,
            "end_date":  market.end_date,
            "days_left": market.days_left,
        }

        result = _ask_model(
            model=config.PRIMARY_MODEL,
            question=market.question,
            yes_price=market.yes_price,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Model returned no parseable result")

        prob = result["probability"]
        conf = result["confidence"]
        reasoning = result.get("reasoning", "")

        # Determine signal
        if conf < config.CONFIDENCE_THRESHOLD:
            signal = "PASS"
            signal_reason = f"Low confidence ({conf:.0%} < {config.CONFIDENCE_THRESHOLD:.0%} threshold)"
        elif prob > market.yes_price:
            edge = prob - market.yes_price
            signal = "YES"
            signal_reason = f"Model sees {prob:.0%} vs market {market.yes_price:.0%} (+{edge:.1%} edge)"
        else:
            edge = (1 - prob) - market.no_price
            if edge > 0:
                signal = "NO"
                signal_reason = f"Model sees {prob:.0%} vs market {market.yes_price:.0%} ({edge:.1%} NO edge)"
            else:
                signal = "PASS"
                signal_reason = "No edge vs market price"

        return {
            "market_id":    market.id,
            "question":     market.question,
            "model":        config.PRIMARY_MODEL,
            "probability":  round(prob, 4),
            "confidence":   round(conf, 4),
            "reasoning":    reasoning,
            "signal":       signal,
            "signal_reason": signal_reason,
            "yes_price":    market.yes_price,
            "no_price":     market.no_price,
        }

    finally:
        config.USE_NEWS_CONTEXT = orig_news
        config.CONSENSUS_MODE   = orig_consensus


# ── Root: serve index.html ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Polymarket AI Viewer</h1><p>index.html not found in static/</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8765"))
    print(f"\n🎯 Polymarket AI Viewer → http://localhost:{port}\n")
    uvicorn.run("webapp:app", host="0.0.0.0", port=port, reload=True)
