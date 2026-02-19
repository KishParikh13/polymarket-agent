"""
Polymarket AI Agent — Configuration
All tunable constants live here.
"""

import os

# ── Load .env if present (python-dotenv) ───────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # don't override vars already set in shell
except ImportError:
    pass  # dotenv is optional; env vars can be set manually

# ── API ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MOONSHOT_API_KEY  = os.getenv("MOONSHOT_API_KEY", "")    # Kimi K2.5 via Moonshot
BRAVE_API_KEY     = os.getenv("BRAVE_API_KEY", "")       # Brave Search for news
GAMMA_API_BASE    = "https://gamma-api.polymarket.com"

# ── Models ─────────────────────────────────────────────────────────────────────
# Override via env: PRIMARY_MODEL=kimi-k2-turbo-preview python backtest.py
PRIMARY_MODEL     = os.getenv("PRIMARY_MODEL", "claude-sonnet-4-6")
SECONDARY_MODEL   = os.getenv("SECONDARY_MODEL", "claude-haiku-4-5")
CONSENSUS_MODE    = True                         # require both models to agree

# ── Kimi / Moonshot OpenAI-compatible backend ──────────────────────────────────
KIMI_BASE_URL     = "https://api.moonshot.ai/v1"
KIMI_PRIMARY      = "kimi-k2-turbo-preview"     # fast + cheap, great for backtests

# ── Bankroll & Sizing ──────────────────────────────────────────────────────────
STARTING_BANKROLL     = 1_000.0                  # USD (paper money)
MAX_BET_FRACTION      = 0.05                     # never more than 5% of bankroll
KELLY_MULTIPLIER      = 0.2                      # conservative Kelly scaling
CONFIDENCE_THRESHOLD  = 0.65                     # min confidence to place bet
CONSENSUS_TOLERANCE   = 0.10                     # models must agree within 10%

# ── Market Filters ─────────────────────────────────────────────────────────────
MIN_VOLUME            = 1_000.0                  # USD
MAX_MARKETS_PER_RUN   = 20                       # limit API + LLM calls per run

# ── Costs ─────────────────────────────────────────────────────────────────────
ROUND_TRIP_COST_PCT   = 0.02                     # 2% spread/fees assumption

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH               = "data/trades.db"

# ── Polymarket CLOB (live trading) ────────────────────────────────────────────
POLY_HOST             = os.getenv("POLY_HOST", "https://clob.polymarket.com")
POLY_CHAIN_ID         = int(os.getenv("POLY_CHAIN_ID", "137"))
POLY_PRIVATE_KEY      = os.getenv("POLY_PRIVATE_KEY", "")
POLY_API_KEY          = os.getenv("POLY_API_KEY", "")
POLY_API_SECRET       = os.getenv("POLY_API_SECRET", "")
POLY_API_PASSPHRASE   = os.getenv("POLY_API_PASSPHRASE", "")

# ── Live trading limits ────────────────────────────────────────────────────────
LIVE_BET_SIZE_USDC    = float(os.getenv("LIVE_BET_SIZE_USDC", "5.0"))
LIVE_MAX_EXPOSURE     = float(os.getenv("LIVE_MAX_EXPOSURE", "50.0"))

# ── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_BUCKETS   = 10                       # decile buckets for curve

# ── Phase 2: News Context ──────────────────────────────────────────────────────
USE_NEWS_CONTEXT      = True                     # include news headlines in LLM prompt
NEWS_MAX_HEADLINES    = 5                        # how many headlines to fetch per market
