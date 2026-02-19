#!/usr/bin/env python3
"""
Backtest — run the forecaster against resolved Polymarket markets.

Usage:
  python backtest.py -n 500 --fast          # 500 markets, single-model
  python backtest.py -n 500                 # 500 markets, dual-model consensus
  python backtest.py -n 500 --show          # include calibration report after
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import requests

# ── Dashboard config ──────────────────────────────────────────────────────────
DASHBOARD_PATH = "data/dashboard.txt"
RECENT_SIZE    = 10
BAR_WIDTH      = 28

# ── ASCII helpers ─────────────────────────────────────────────────────────────
def bar(ratio: float, width: int = BAR_WIDTH, fill="█", empty="░") -> str:
    filled = round(ratio * width)
    return fill * filled + empty * (width - filled)

def center(text: str, width: int = 60) -> str:
    return text.center(width)

def fmt_pnl(v: float) -> str:
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"

def fmt_dur(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"

# ── Dashboard writer ──────────────────────────────────────────────────────────
def write_dashboard(
    total: int,
    processed: int,
    results: dict,
    recent: deque,
    categories: dict,
    start_time: float,
    done: bool = False,
):
    elapsed   = time.time() - start_time
    rate      = processed / elapsed if elapsed > 0 else 0
    remaining = (total - processed) / rate if rate > 0 and not done else 0

    signals   = results["signal"]
    correct   = results["correct"]
    wrong     = results["wrong"]
    no_signal = results["no_signal"]
    total_pnl = results["pnl"]

    win_rate  = correct / signals if signals > 0 else 0
    avg_pnl   = total_pnl / signals if signals > 0 else 0
    sig_rate  = signals / processed if processed > 0 else 0

    W = 62   # total box width (inner)

    def box_top(title=""):
        if title:
            pad = W - len(title) - 2
            return f"  ┌─ {title} {'─' * max(0, pad - 2)}┐"
        return f"  ┌{'─' * (W)}┐"

    def box_bot():
        return f"  └{'─' * W}┘"

    def row(content=""):
        # pad content to fit inside box
        visible = len(content.encode("ascii", errors="replace"))
        pad = W - visible - 1
        return f"  │ {content}{' ' * max(0, pad)}│"

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    status_word = "COMPLETE ✓" if done else "RUNNING ▶"
    lines += [
        "",
        f"  ╔{'═' * W}╗",
        f"  ║{center('POLYMARKET AI BACKTEST', W)}║",
        f"  ║{center(f'{total} markets  ·  fast-mode  ·  {status_word}', W)}║",
        f"  ╚{'═' * W}╝",
        "",
    ]

    # ── Progress ──────────────────────────────────────────────────────────────
    pct        = processed / total if total > 0 else 0
    prog_bar   = bar(pct)
    pct_label  = f"{pct:.0%}"
    lines += [
        box_top("PROGRESS"),
        row(f"  {prog_bar}  {processed}/{total}  {pct_label}"),
        row(f"  Elapsed  {fmt_dur(elapsed)}   ETA  {fmt_dur(remaining)}   Rate  {rate:.1f}/min"),
        box_bot(),
        "",
    ]

    # ── Signal stats ──────────────────────────────────────────────────────────
    win_bar = bar(win_rate, 20)
    lines += [
        box_top("SIGNALS"),
        row(f"  Fired    {signals:>4}  ({sig_rate:.0%} of markets)    No-signal  {no_signal}"),
        row(f"  Win      {win_bar}  {win_rate:.0%}   {correct}W / {wrong}L"),
        row(f"  P&L      {fmt_pnl(total_pnl):<12}   avg {fmt_pnl(avg_pnl)}/trade"),
        box_bot(),
        "",
    ]

    # ── Recent results ────────────────────────────────────────────────────────
    lines.append(box_top(f"LAST {RECENT_SIZE}"))
    if not recent:
        lines.append(row("  (waiting for first result...)"))
    for r in list(recent)[::-1]:    # newest first
        if r["direction"] == "NO_SIGNAL":
            icon = "⏭ "
            stat = f"{'skip':<22}"
        else:
            icon = "✅" if r["correct"] else "❌"
            stat = f"{r['direction']:<3}  {r['conf']:.0%}  {fmt_pnl(r['pnl']):<10}"
        q = r["question"]
        max_q = W - len(stat) - 12
        q_short = (q[:max_q] + "…") if len(q) > max_q else q
        lines.append(row(f"  {r['idx']:>3}  {icon} {stat}  {q_short}"))
    lines += [box_bot(), ""]

    # ── Category breakdown ────────────────────────────────────────────────────
    lines.append(box_top("BY CATEGORY"))
    if not any(v["signal"] > 0 for v in categories.values()):
        lines.append(row("  (no signals yet)"))
    else:
        for cat, v in sorted(categories.items(), key=lambda x: -x[1]["signal"]):
            if v["signal"] == 0:
                continue
            wr    = v["correct"] / v["signal"] if v["signal"] > 0 else 0
            b     = bar(wr, 12)
            label = f"{cat:<10} {b}  {v['correct']}W/{v['wrong']}L  {wr:.0%}  {fmt_pnl(v['pnl'])}"
            lines.append(row(f"  {label}"))
    lines += [box_bot(), ""]

    # ── Footer ────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    lines.append(f"  {'─' * W}")
    lines.append(f"  Updated {ts}  ·  watch with: watch -n1 cat {DASHBOARD_PATH}")
    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(content)

    # Also echo a compact one-liner to stdout
    print(f"\r[{processed:>3}/{total}] {bar(pct, 20)} {pct:.0%}  "
          f"signals={signals}  win={win_rate:.0%}  pnl={fmt_pnl(total_pnl)}  "
          f"eta={fmt_dur(remaining)}",
          end="", flush=True)


# ── Gamma API fetch ────────────────────────────────────────────────────────────
def fetch_resolved_markets(limit: int = 1000) -> list[dict]:
    url     = "https://gamma-api.polymarket.com/markets"
    markets = []
    offset  = 0
    batch   = 100

    print(f"Fetching resolved markets from Polymarket...", flush=True)

    while len(markets) < limit:
        try:
            resp = requests.get(url, params={
                "closed":    "true",
                "limit":     batch,
                "offset":    offset,
                "order":     "volume",
                "ascending": "false",
            }, timeout=20)
            resp.raise_for_status()
            batch_data = resp.json()
        except Exception as e:
            print(f"\nAPI error: {e}", flush=True)
            break

        if not batch_data:
            break

        for m in batch_data:
            try:
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if len(prices) != 2:
                    continue

                yes_final = float(prices[0])
                if yes_final > 0.95:
                    actual_outcome = 1.0
                elif yes_final < 0.05:
                    actual_outcome = 0.0
                else:
                    continue

                question = m.get("question", "").strip()
                if not question or len(question) < 10:
                    continue

                # Skip noisy sports prop bets
                skip_kw = ["set 1", "set 2", "game 1", "game 2", "map 1", "map 2",
                           "bo3", "bo5", "quarter", "half", "inning", "1st period", "2nd period"]
                if any(kw in question.lower() for kw in skip_kw):
                    continue

                markets.append({
                    "id":             m.get("id", ""),
                    "question":       question,
                    "yes_price":      0.5,   # neutral prior — test model signal only
                    "no_price":       0.5,
                    "volume":         float(m.get("volumeNum", 0) or 0),
                    "actual_outcome": actual_outcome,
                    "end_date":       m.get("endDate", ""),
                    "raw":            {},
                })

                if len(markets) >= limit:
                    break

            except Exception:
                continue

        offset += batch
        if len(batch_data) < batch:
            break

    print(f"  → Fetched {len(markets)} clean binary markets", flush=True)
    return markets


# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest(n: int = 100, fast_mode: bool = False, show_report: bool = False):
    from dotenv import load_dotenv
    load_dotenv()

    import config
    if fast_mode:
        config.CONSENSUS_MODE = False

    from src.db         import init_db, get_conn
    from src.classifier import classify
    from src.forecaster import forecast
    from config         import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    init_db()

    markets = fetch_resolved_markets(limit=n * 3)
    markets = markets[:n]

    if not markets:
        print("No markets fetched.")
        return

    print(f"\nStarting backtest: {len(markets)} markets | fast={fast_mode}")
    print(f"Dashboard: watch -n1 cat {DASHBOARD_PATH}\n")

    conn       = get_conn()
    start_time = time.time()
    recent     = deque(maxlen=RECENT_SIZE)
    results    = {"signal": 0, "no_signal": 0, "correct": 0, "wrong": 0, "pnl": 0.0}
    categories: dict[str, dict] = {}

    def get_cat(cat):
        if cat not in categories:
            categories[cat] = {"signal": 0, "correct": 0, "wrong": 0, "pnl": 0.0}
        return categories[cat]

    for i, m in enumerate(markets):
        category = classify(m["question"])
        signal   = forecast(m)
        now      = datetime.now(timezone.utc).isoformat()

        if signal is None:
            conn.execute("""
                INSERT INTO trades
                  (market_id, question, category,
                   model_probability, confidence, implied_probability,
                   bet_direction, simulated_amount, bankroll_at_bet, timestamp,
                   resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("bt-" + m["id"], m["question"], category,
                  0.5, 0.0, 0.5, "NO_SIGNAL", 0.0, 0.0, now,
                  m["actual_outcome"], 0.0))
            results["no_signal"] += 1
            recent.append({"idx": i + 1, "direction": "NO_SIGNAL", "correct": False,
                           "conf": 0.0, "pnl": 0.0, "question": m["question"]})

        else:
            direction = signal["bet_direction"]
            amount    = 30.0
            payout    = amount / 0.5 if (
                (direction == "YES" and m["actual_outcome"] == 1.0) or
                (direction == "NO"  and m["actual_outcome"] == 0.0)
            ) else 0.0
            pnl         = payout - amount - (amount * 0.02)
            was_correct = payout > 0

            conn.execute("""
                INSERT INTO trades
                  (market_id, question, category,
                   model_probability, confidence, implied_probability,
                   bet_direction, simulated_amount, bankroll_at_bet, timestamp,
                   resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, ("bt-" + m["id"], m["question"], category,
                  signal["probability"], signal["confidence"], signal["implied_prob"],
                  direction, amount, 1000.0, now, m["actual_outcome"], pnl))

            results["signal"] += 1
            results["pnl"]    += pnl
            if was_correct:
                results["correct"] += 1
            else:
                results["wrong"] += 1

            cat = get_cat(category)
            cat["signal"] += 1
            cat["pnl"]    += pnl
            if was_correct:
                cat["correct"] += 1
            else:
                cat["wrong"] += 1

            recent.append({"idx": i + 1, "direction": direction, "correct": was_correct,
                           "conf": signal["confidence"], "pnl": pnl, "question": m["question"]})

        conn.commit()
        write_dashboard(len(markets), i + 1, results, recent, categories, start_time)

    conn.close()

    # Final dashboard with done=True
    write_dashboard(len(markets), len(markets), results, recent, categories, start_time, done=True)
    print(f"\n\nDone! {results['signal']} signals, {results['correct']}W/{results['wrong']}L, "
          f"P&L {fmt_pnl(results['pnl'])}")

    if show_report and results["signal"] >= 10:
        print("\nCalibration report:")
        import main as m_main
        m_main.cmd_calibration(argparse.Namespace())


def main():
    parser = argparse.ArgumentParser(description="Backtest forecaster on resolved Polymarket markets")
    parser.add_argument("-n",     type=int, default=100, help="Number of markets (default 100)")
    parser.add_argument("--fast", action="store_true",   help="Single-model mode (no consensus, 2x faster)")
    parser.add_argument("--show", action="store_true",   help="Show calibration report when done")
    args = parser.parse_args()
    run_backtest(n=args.n, fast_mode=args.fast, show_report=args.show)


if __name__ == "__main__":
    main()
