"""
Polymarket AI Agent — Configuration
All tunable constants live here.
"""

import os

# ── API ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GAMMA_API_BASE    = "https://gamma-api.polymarket.com"

# ── Models ─────────────────────────────────────────────────────────────────────
PRIMARY_MODEL     = "claude-sonnet-4-6"          # main forecaster
SECONDARY_MODEL   = "claude-haiku-4-5"           # consensus check (cheaper)
CONSENSUS_MODE    = True                         # require both models to agree

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

# ── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_BUCKETS   = 10                       # decile buckets for curve
