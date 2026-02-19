#!/usr/bin/env python3
"""
Polymarket Analyzer — lightweight web app.

Serves live markets and runs AI analysis on demand.
No trading, no storage — pure analysis tool.

Usage:
  python app.py          # starts on http://localhost:8000
  python app.py --port 8080
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests as http_requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Bootstrap env ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import config
config.CONSENSUS_MODE   = False   # single model for speed in UI
config.USE_NEWS_CONTEXT = False   # no news by default (user can enable)

app = FastAPI(title="Polymarket Analyzer", docs_url=None, redoc_url=None)

# ── Serve static files ─────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# ── Market fetch ───────────────────────────────────────────────────────────────
SKIP_KW = [
    "up or down", "5 minutes", "15 minutes", "30 minutes",
    "highest temperature", "lowest temperature", "°c on", "°f on",
    "set 1", "set 2", "game 1", "game 2", "map 1", "map 2",
    "bo3", "bo5", "1st period", "2nd period", "inning",
    "o/u", "spread:", "handicap", "candle", "wick",
]

CRYPTO_KW = [
    "bitcoin", "ethereum", "btc", "eth", "solana", "xrp", "dogecoin",
    "doge", "bnb", "cardano", "polygon", "chainlink", "uniswap",
    "token", "defi", "nft", "usdc", "usdt", "coinbase", "binance",
]

CATEGORY_KEYWORDS = {
    "politics":   ["election", "president", "trump", "biden", "congress", "senate",
                   "vote", "democrat", "republican", "governor", "candidate", "ballot",
                   "approval", "poll", "party", "legislation", "impeach"],
    "economics":  ["fed", "inflation", "gdp", "unemployment", "rate", "recession",
                   "market cap", "tariff", "treasury", "economy", "cpi", "interest"],
    "crypto":     CRYPTO_KW,
    "sports":     ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
                   "baseball", "tennis", "golf", "championship", "playoff", "super bowl",
                   "world cup", "league", "match", "tournament", "win the", "beats"],
    "tech":       ["apple", "google", "microsoft", "amazon", "meta", "tesla", "openai",
                   "ai", "launch", "release", "acquire", "ipo", "merger", "startup"],
}

def categorize(question: str) -> str:
    q = question.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in q for k in kws):
            return cat
    return "other"


@app.get("/api/markets")
async def get_markets(
    days:       int   = Query(30,  ge=1,   le=365),
    min_price:  float = Query(0.12, ge=0.0, le=0.5),
    max_price:  float = Query(0.88, ge=0.5, le=1.0),
    min_volume: float = Query(0,   ge=0),
    category:   str   = Query("all"),
    search:     str   = Query(""),
    no_crypto:  bool  = Query(False),
    sort:       str   = Query("volume"),  # volume | closing | price
):
    """Fetch live open markets with filters."""
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days)

    markets = []
    offset  = 0

    while len(markets) < 500:
        try:
            resp = http_requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true",
                        "limit": 100, "offset": offset,
                        "order": "volume", "ascending": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Polymarket API error: {e}")

        if not data:
            break

        for m in data:
            try:
                # Price filter
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if len(prices) != 2:
                    continue
                yes_price = float(prices[0])
                no_price  = float(prices[1])
                if not (min_price <= yes_price <= max_price):
                    continue

                # End date filter
                end_str = m.get("endDate", "")
                if not end_str:
                    continue
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt > cutoff or end_dt < now:
                    continue
                days_left = max(0, (end_dt - now).days)

                question = m.get("question", "").strip()
                if not question or len(question) < 10:
                    continue

                # Skip junk markets
                if any(k in question.lower() for k in SKIP_KW):
                    continue

                volume = float(m.get("volumeNum", 0) or 0)
                if volume < min_volume:
                    continue

                cat = categorize(question)

                # Category filter
                if no_crypto and cat == "crypto":
                    continue
                if category != "all" and cat != category:
                    continue

                # Search filter
                if search and search.lower() not in question.lower():
                    continue

                markets.append({
                    "id":        m.get("id", ""),
                    "question":  question,
                    "yes_price": round(yes_price, 4),
                    "no_price":  round(no_price, 4),
                    "volume":    round(volume),
                    "days_left": days_left,
                    "end_date":  end_str[:10],
                    "category":  cat,
                    "slug":      m.get("slug", ""),
                })

            except Exception:
                continue

        offset += 100
        if len(data) < 100:
            break

    # Sort
    if sort == "closing":
        markets.sort(key=lambda x: x["days_left"])
    elif sort == "price":
        markets.sort(key=lambda x: abs(x["yes_price"] - 0.5))  # most uncertain first
    else:
        markets.sort(key=lambda x: -x["volume"])

    return JSONResponse({"markets": markets, "total": len(markets)})


# ── Analysis endpoint ──────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    id:        str
    question:  str
    yes_price: float
    no_price:  float
    model:     Optional[str] = None   # override model for this request
    use_news:  Optional[bool] = False


@app.post("/api/analyze")
async def analyze_market(req: AnalyzeRequest):
    """Run AI forecaster on a single market and return the signal."""
    try:
        import config as cfg
        original_model   = cfg.PRIMARY_MODEL
        original_news    = cfg.USE_NEWS_CONTEXT

        if req.model:
            cfg.PRIMARY_MODEL = req.model
        if req.use_news:
            cfg.USE_NEWS_CONTEXT = True

        from src.forecaster import forecast
        from src.classifier import classify

        market = {
            "id":        req.id,
            "question":  req.question,
            "yes_price": req.yes_price,
            "no_price":  req.no_price,
            "volume":    0,
            "end_date":  "",
            "raw":       {},
        }

        signal   = forecast(market)
        category = classify(req.question)

        # Restore config
        cfg.PRIMARY_MODEL   = original_model
        cfg.USE_NEWS_CONTEXT = original_news

        if signal is None:
            return JSONResponse({
                "signal":    False,
                "reason":    "Low confidence or no edge vs market",
                "category":  category,
                "model":     req.model or config.PRIMARY_MODEL,
            })

        return JSONResponse({
            "signal":      True,
            "direction":   signal["bet_direction"],
            "probability": round(signal["probability"], 3),
            "confidence":  round(signal["confidence"], 3),
            "edge":        round(signal["edge"], 3),
            "reasoning":   signal["reasoning"],
            "category":    category,
            "model":       signal["models"][0] if signal.get("models") else "",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(static_dir, "index.html")) as f:
        return f.read()


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    print(f"\n🎯 Polymarket Analyzer → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
