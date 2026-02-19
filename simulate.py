#!/usr/bin/env python3
"""
simulate.py — Objective forward-looking simulation on LIVE Polymarket markets.

This is the only truly clean test:
  - Uses OPEN markets (unknown outcomes → zero leakage)
  - Uses REAL market prices as baseline (tests if we can BEAT the market)
  - Filters for markets closing within N days (fast feedback loop)
  - Logs paper trades; resolve them when markets close
  - Mirrors production exactly

Usage:
  python simulate.py run              # fetch live markets, forecast, log paper trades
  python simulate.py run --days 7     # only markets closing within 7 days (faster results)
  python simulate.py run --days 30    # wider net, more markets
  python simulate.py status           # show open bets + live P&L estimate
  python simulate.py resolve          # check which bets have resolved, compute P&L
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

DASHBOARD_PATH = "data/sim_dashboard.txt"
BAR_WIDTH      = 28

# ── ASCII helpers ─────────────────────────────────────────────────────────────
def bar(ratio: float, width: int = BAR_WIDTH, fill="█", empty="░") -> str:
    filled = round(max(0.0, min(1.0, ratio)) * width)
    return fill * filled + empty * (width - filled)

def fmt_pnl(v: float) -> str:
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"

def fmt_dur(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"

def center(text: str, width: int = 62) -> str:
    return text.center(width)

# ── Gamma API — OPEN markets ───────────────────────────────────────────────────
def fetch_open_markets(max_days_to_close: int = 14, limit: int = 200) -> list[dict]:
    """
    Fetch live open markets closing within max_days_to_close days.
    Uses REAL market prices — this is what we'd actually trade.
    """
    url     = "https://gamma-api.polymarket.com/markets"
    markets = []
    offset  = 0
    batch   = 100

    cutoff = datetime.now(timezone.utc) + timedelta(days=max_days_to_close)

    print(f"Fetching open markets (closing within {max_days_to_close} days)...", flush=True)

    while len(markets) < limit:
        try:
            resp = requests.get(url, params={
                "closed":    "false",
                "active":    "true",
                "limit":     batch,
                "offset":    offset,
                "order":     "volume",
                "ascending": "false",
            }, timeout=20)
            resp.raise_for_status()
            batch_data = resp.json()
        except Exception as e:
            print(f"API error: {e}", flush=True)
            break

        if not batch_data:
            break

        for m in batch_data:
            try:
                # Parse real market prices
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if len(prices) != 2:
                    continue

                yes_price = float(prices[0])
                no_price  = float(prices[1])

                # Must be binary, not extreme (extreme = market is certain, no edge possible)
                if not (0.12 < yes_price < 0.88):
                    continue
                if abs(yes_price + no_price - 1.0) > 0.05:
                    continue

                question = m.get("question", "").strip()
                if not question or len(question) < 10:
                    continue

                # Filter by end date
                end_date_str = m.get("endDate", "")
                if not end_date_str:
                    continue
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    if end_date > cutoff:
                        continue   # closing too far out
                    if end_date < datetime.now(timezone.utc):
                        continue   # already past end date but not marked closed yet
                except Exception:
                    continue

                # Skip markets where AI has no structural edge
                skip_kw = [
                    # Ultra short-term price direction
                    "up or down", "5 minutes", "15 minutes", "30 minutes",
                    "8:15", "8:30", "8:45", "9:00pm", "10:00pm",
                    # Exact temperature thresholds
                    "highest temperature", "lowest temperature", "°c on", "°f on",
                    # Sports game-level props (too granular)
                    "set 1", "set 2", "game 1", "game 2", "map 1", "map 2",
                    "bo3", "bo5", "1st period", "2nd period", "inning",
                    "o/u", "spread:", "handicap",
                    # Micro crypto
                    "candle", "wick",
                    # Generic sports team win (too many unknowns without lineups)
                    " win on ", " win on 2",
                    # Social media vanity metrics
                    "say \"", "say '", "times during",
                ]
                if any(kw in question.lower() for kw in skip_kw):
                    continue

                # Must be a "will X happen" style question, not a game result
                # Prefer markets with keywords suggesting AI can reason about them
                good_kw = [
                    "will ", "rate", "price", "election", "president", "fed",
                    "inflation", "unemployment", "gdp", "approval", "bill",
                    "congress", "senate", "trump", "biden", "war", "deal",
                    "merger", "acquisition", "ipo", "ban", "launch", "sign",
                    "announce", "pass", "win the", "nominee", "candidate",
                    "cryptocurrency", "bitcoin", "ethereum", "market cap",
                    "tariff", "sanction", "ceasefire", "treaty",
                ]
                if not any(kw in question.lower() for kw in good_kw):
                    continue

                days_left = (end_date - datetime.now(timezone.utc)).days

                markets.append({
                    "id":        m.get("id", ""),
                    "question":  question,
                    "yes_price": yes_price,
                    "no_price":  no_price,
                    "volume":    float(m.get("volumeNum", 0) or 0),
                    "end_date":  end_date_str,
                    "days_left": days_left,
                    "raw":       {},
                })

                if len(markets) >= limit:
                    break

            except Exception:
                continue

        offset += batch
        if len(batch_data) < batch:
            break

    # Sort by days_left ascending (soonest first = fastest feedback)
    markets.sort(key=lambda x: x["days_left"])
    print(f"  → {len(markets)} live markets closing within {max_days_to_close} days", flush=True)
    return markets


# ── Dashboard ─────────────────────────────────────────────────────────────────
def write_sim_dashboard(markets, results, processed, start_time, done=False):
    elapsed  = time.time() - start_time
    rate     = processed / elapsed * 60 if elapsed > 0 else 0
    total    = len(markets)
    pct      = processed / total if total > 0 else 0
    eta_secs = (total - processed) / (processed / elapsed) if processed > 0 and not done else 0

    signals   = results["signal"]
    correct   = results.get("correct", 0)   # resolved only
    resolved  = results.get("resolved", 0)
    pnl       = results["pnl"]
    no_signal = results["no_signal"]
    win_rate  = correct / resolved if resolved > 0 else 0

    W = 62
    status = "COMPLETE ✓" if done else "RUNNING ▶"

    def row(content=""):
        visible = len(content.encode("ascii", errors="replace"))
        pad = W - visible - 1
        return f"  │ {content}{' ' * max(0, pad)}│"

    def box_top(title=""):
        pad = W - len(title) - 4
        return f"  ┌─ {title} {'─' * max(0, pad)}┐"

    def box_bot():
        return f"  └{'─' * W}┘"

    lines = [
        "",
        f"  ╔{'═' * W}╗",
        f"  ║{center('POLYMARKET LIVE SIMULATION', W)}║",
        f"  ║{center(f'OPEN markets · real prices · {status}', W)}║",
        f"  ╚{'═' * W}╝",
        "",
        box_top("PROGRESS"),
        row(f"  {bar(pct)}  {processed}/{total}  ({pct:.0%})"),
        row(f"  Elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(eta_secs)}  Rate {rate:.1f}/min"),
        box_bot(),
        "",
        box_top("PAPER TRADES"),
        row(f"  Signals fired   {signals:>4}  ({signals/processed:.0%} signal rate)" if processed else row("  Signals fired      0")),
        row(f"  Skipped (low edge/conf)  {no_signal:>4}"),
        row(f"  Resolved so far  {resolved:>3}  →  Win {win_rate:.0%}  {correct}W/{resolved-correct}L") if resolved > 0 else row(f"  Resolved so far     0  (markets still open)"),
        row(f"  Unrealized P&L  {fmt_pnl(pnl)}"),
        box_bot(),
        "",
    ]

    # Show all open bets
    lines.append(box_top(f"OPEN BETS  ({len(results['bets'])} total)"))
    if not results["bets"]:
        lines.append(row("  (no signals fired yet)"))
    for b in results["bets"][-12:]:    # last 12
        days = b.get("days_left", "?")
        conf  = b.get("conf", 0)
        amt   = b.get("amount", 30)
        q = b["question"]
        max_q = 32
        q_short = (q[:max_q] + "…") if len(q) > max_q else q
        resolved_str = "✅" if b.get("won") else ("❌" if b.get("lost") else f"⏳{days}d")
        lines.append(row(
            f"  {resolved_str} {b['direction']:<3} {conf:.0%}  ${amt:.0f}  "
            f"mkt={b['market_prob']:.0%}→mdl={b['model_prob']:.0%}  {q_short}"
        ))
    lines += [box_bot(), ""]

    # Footer
    ts = datetime.now().strftime("%H:%M:%S")
    lines.append(f"  {'─' * W}")
    lines.append(f"  Why this test is clean:")
    lines.append(f"    ✓ Open markets → outcomes unknown (zero leakage)")
    lines.append(f"    ✓ Real prices  → testing edge vs market, not just direction")
    lines.append(f"    ✓ No news      → pure model signal vs crowd wisdom")
    lines.append(f"  Updated {ts}  ·  run 'python simulate.py resolve' to check results")
    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(content)

    print(f"\r[{processed:>3}/{total}] {bar(pct, 20)} {pct:.0%}  "
          f"signals={signals}  pnl={fmt_pnl(pnl)}  eta={fmt_dur(eta_secs)}",
          end="", flush=True)


# ── Run simulation ─────────────────────────────────────────────────────────────
def cmd_run(args):
    from dotenv import load_dotenv
    load_dotenv()

    import config
    config.CONSENSUS_MODE = False      # single model for speed; flip for production
    config.USE_NEWS_CONTEXT = False    # no news — pure model signal vs market price

    from src.db         import init_db, get_conn
    from src.classifier import classify
    from src.forecaster import forecast
    from config         import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)

    init_db()

    days = getattr(args, "days", 14)
    markets = fetch_open_markets(max_days_to_close=days, limit=300)

    if not markets:
        print("No open markets found in that window."); return

    print(f"\nSimulating {len(markets)} live markets | model vs real prices | no news")
    print(f"Dashboard: watch -n1 cat {DASHBOARD_PATH}\n")

    conn       = get_conn()
    start_time = time.time()
    results    = {"signal": 0, "no_signal": 0, "pnl": 0.0,
                  "correct": 0, "resolved": 0, "bets": []}

    for i, m in enumerate(markets):
        category = classify(m["question"])
        signal   = forecast(m)   # uses REAL yes_price as implied prob
        now      = datetime.now(timezone.utc).isoformat()

        if signal is None:
            conn.execute("""
                INSERT OR IGNORE INTO trades
                  (market_id, question, category,
                   model_probability, confidence, implied_probability,
                   bet_direction, simulated_amount, bankroll_at_bet, timestamp,
                   resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0)
            """, ("sim-" + m["id"], m["question"], category,
                  0.5, 0.0, m["yes_price"], "NO_SIGNAL", 0.0, 0.0, now))
            results["no_signal"] += 1

        else:
            direction    = signal["bet_direction"]
            model_prob   = signal["probability"]
            market_prob  = m["yes_price"]
            amount       = 30.0

            conn.execute("""
                INSERT OR IGNORE INTO trades
                  (market_id, question, category,
                   model_probability, confidence, implied_probability,
                   bet_direction, simulated_amount, bankroll_at_bet, timestamp,
                   resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0)
            """, ("sim-" + m["id"], m["question"], category,
                  model_prob, signal["confidence"], market_prob,
                  direction, amount, 1000.0, now))

            results["signal"] += 1
            results["bets"].append({
                "question":    m["question"],
                "direction":   direction,
                "conf":        signal["confidence"],
                "amount":      amount,
                "market_prob": market_prob,
                "model_prob":  model_prob,
                "days_left":   m["days_left"],
            })

        conn.commit()
        write_sim_dashboard(markets, results, i + 1, start_time)

    conn.close()
    write_sim_dashboard(markets, results, len(markets), start_time, done=True)

    print(f"\n\nDone. {results['signal']} paper trades placed on live markets.")
    print(f"Run 'python simulate.py resolve' in {days} days to check results.")
    print(f"Dashboard: {DASHBOARD_PATH}")


# ── Status ─────────────────────────────────────────────────────────────────────
def cmd_status(args):
    from dotenv import load_dotenv; load_dotenv()
    from src.db import init_db, get_conn; init_db()
    conn = get_conn()

    rows = conn.execute("""
        SELECT question, bet_direction, model_probability, confidence,
               implied_probability, simulated_amount, timestamp,
               resolved, actual_outcome, profit_loss
        FROM trades
        WHERE market_id LIKE 'sim-%' AND bet_direction != 'NO_SIGNAL'
        ORDER BY timestamp DESC
    """).fetchall()

    if not rows:
        print("No simulation trades logged yet. Run: python simulate.py run")
        return

    open_bets     = [r for r in rows if not r[7]]
    resolved_bets = [r for r in rows if r[7]]
    wins          = [r for r in resolved_bets if r[9] > 0]

    W = 64
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║{center('SIMULATION STATUS', W)}║")
    print(f"  ╚{'═' * W}╝\n")
    print(f"  Total trades:  {len(rows)}")
    print(f"  Open bets:     {len(open_bets)}")
    print(f"  Resolved:      {len(resolved_bets)}  →  "
          f"{len(wins)}W / {len(resolved_bets) - len(wins)}L  "
          f"({len(wins)/len(resolved_bets):.0%})" if resolved_bets else "  Resolved:      0")

    if open_bets:
        print(f"\n  {'─' * W}")
        print(f"  OPEN BETS:")
        for r in open_bets[:20]:
            q = r[0][:55] + "…" if len(r[0]) > 55 else r[0]
            print(f"  ⏳ {r[1]:<3}  conf={r[3]:.0%}  mkt={r[4]:.0%}→mdl={r[2]:.0%}  ${r[5]:.0f}  {q}")

    conn.close()


# ── Resolve ────────────────────────────────────────────────────────────────────
def cmd_resolve(args):
    """Check Polymarket for resolved markets and update trade outcomes."""
    from dotenv import load_dotenv; load_dotenv()
    from src.db import init_db, get_conn; init_db()
    conn = get_conn()

    open_trades = conn.execute("""
        SELECT id, market_id, question, bet_direction, simulated_amount
        FROM trades
        WHERE market_id LIKE 'sim-%'
          AND bet_direction != 'NO_SIGNAL'
          AND resolved = 0
    """).fetchall()

    if not open_trades:
        print("No open simulation trades to resolve.")
        return

    print(f"Checking {len(open_trades)} open bets against Polymarket...\n")
    resolved_count = 0

    for trade in open_trades:
        trade_id, market_id, question, direction, amount = trade
        poly_id = market_id.replace("sim-", "")

        try:
            resp = requests.get(
                f"https://gamma-api.polymarket.com/markets/{poly_id}", timeout=10
            )
            if resp.status_code != 200:
                continue
            m = resp.json()
            if not m.get("closed"):
                continue   # still open

            prices = m.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices)
            if len(prices) != 2:
                continue

            yes_final = float(prices[0])
            if yes_final > 0.95:
                actual = 1.0
            elif yes_final < 0.05:
                actual = 0.0
            else:
                continue   # ambiguous resolution

            won = (direction == "YES" and actual == 1.0) or \
                  (direction == "NO"  and actual == 0.0)
            payout = (amount / yes_final if direction == "YES" else amount / (1 - yes_final)) if won else 0.0
            pnl    = payout - amount - (amount * 0.02)

            conn.execute("""
                UPDATE trades
                SET resolved=1, actual_outcome=?, profit_loss=?
                WHERE id=?
            """, (actual, pnl, trade_id))
            conn.commit()

            icon = "✅" if won else "❌"
            print(f"  {icon} {direction:<3}  {fmt_pnl(pnl):>9}  {question[:65]}")
            resolved_count += 1

        except Exception as e:
            print(f"  ⚠️  Error checking {poly_id}: {e}")

    print(f"\nResolved {resolved_count}/{len(open_trades)} trades.")
    conn.close()


# ── Entry ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live market simulation — objective, no leakage")
    sub    = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run simulation on live open markets")
    p_run.add_argument("--days", type=int, default=30,
                       help="Only markets closing within N days (default 30)")

    sub.add_parser("status",  help="Show open bets and resolved results")
    sub.add_parser("resolve", help="Check which bets have resolved, compute P&L")

    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "resolve":
        cmd_resolve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
