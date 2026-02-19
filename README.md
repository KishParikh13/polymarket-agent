# Polymarket AI Agent

A paper-trading + calibration system for Polymarket prediction markets, built in two phases.

## Architecture

```
main.py              CLI entrypoint
config.py            All tunable constants
src/
  db.py              SQLite setup
  markets.py         Polymarket Gamma API → active binary markets
  classifier.py      Keyword-based category tagger (politics/sports/crypto/tech/economics)
  forecaster.py      Claude LLM → probability + confidence + consensus mode
  paper_trader.py    Conservative Kelly sizing → simulated bets logged to DB
  tracker.py         Poll resolved markets → update outcomes + P&L
  calibration.py     Brier score, calibration curve, ROI by category, max drawdown
data/
  trades.db          SQLite: all trades and calibration snapshots
```

## Setup

```bash
# 1. Clone / enter the repo
cd ~/Projects/polymarket-agent

# 2. Create a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
# or: cp .env.example .env && edit .env, then: source .env
```

## Phase 0 — Paper Trading

```bash
# Run the agent: fetch markets → LLM forecast → paper trade
python main.py run

# Check portfolio status
python main.py status

# Resolve completed markets (run periodically)
python main.py resolve
```

**What it does:**
1. Fetches the top active binary markets from Polymarket (sorted by volume, >$1k)
2. Sends each question + current price to Claude (two models in consensus mode)
3. If both models agree with ≥65% confidence and there's genuine edge vs market price → logs a paper bet
4. Kelly-sized bet: never more than 5% of bankroll, scaled by confidence
5. All bets tracked in SQLite at `data/trades.db`

## Phase 1 — Calibration

```bash
# After markets resolve, run calibration report
python main.py calibration
```

**Metrics produced:**
- **Brier Score** — calibration quality (0.0 = perfect, 0.25 = random)
- **Calibration curve** — decile buckets: does 70% confidence = 70% actual win rate?
- **ROI by category** — where's the real edge? (politics vs sports vs crypto…)
- **Edge vs market** — are you beating the market's prior probability?
- **Max drawdown** — worst peak-to-trough in cumulative P&L
- Saved to `calibration_report.json`

## When to Scale

After 200+ resolved trades:
- Brier score trending down over time ✅
- Calibration curve close to diagonal ✅
- ROI > 5% on your best category ✅
- Delta vs market implied prob > 0 ✅

Then: go live with a small USDC bankroll ($200-500), same confidence thresholds.

## Config Tuning (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.65 | Min confidence to bet |
| `CONSENSUS_MODE` | True | Require both models to agree |
| `CONSENSUS_TOLERANCE` | 0.10 | Max disagreement between models |
| `MAX_BET_FRACTION` | 0.05 | Max 5% of bankroll per bet |
| `KELLY_MULTIPLIER` | 0.20 | Conservative Kelly scaling |
| `MIN_VOLUME` | 1000 | Min market volume (USD) |
| `MAX_MARKETS_PER_RUN` | 20 | Limit LLM calls per run |

## References

- [Polymarket Gamma API](https://gamma-api.polymarket.com/markets)
- [Polymarket/agents](https://github.com/Polymarket/agents) — reference implementation
- [valory-xyz/trader](https://github.com/valory-xyz/trader) — live autonomous service
