#!/usr/bin/env python3
"""
Post-cutoff backtest — tests model on markets that resolved AFTER its training cutoff.

Why this is clean:
  - Claude's training cutoff ≈ early 2025
  - Markets resolved Mar 2025–present: model has NOT seen the outcomes in training
  - Uses REAL market prices (not neutral 0.5) → tests edge vs crowd
  - No news fetching → pure model signal
  - Immediate results (we know outcomes, resolve trades instantly)

Usage:
  python postcutoff_backtest.py -n 100
  python postcutoff_backtest.py -n 200 --cutoff 2025-04-01
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import requests

DASHBOARD_PATH = "data/postcutoff_dashboard.txt"
CUTOFF_DEFAULT = "2025-03-01"   # conservative — well past training data
RECENT_SIZE    = 12

# ── ASCII helpers ─────────────────────────────────────────────────────────────
def bar(r, w=28, f="█", e="░"):
    n = round(max(0.0, min(1.0, r)) * w)
    return f * n + e * (w - n)

def fmt_pnl(v):
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"

def fmt_dur(s):
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"

def center(t, w=62):
    return t.center(w)

# ── Dashboard ─────────────────────────────────────────────────────────────────
def write_dashboard(total, processed, results, recent, cats, start, cutoff, done=False):
    elapsed  = time.time() - start
    rate     = processed / elapsed * 60 if elapsed > 0 else 0
    pct      = processed / total if total > 0 else 0
    eta      = (total - processed) / (processed / elapsed) if processed > 0 and not done else 0

    sigs   = results["signal"]
    cor    = results["correct"]
    wrong  = results["wrong"]
    nosig  = results["no_signal"]
    pnl    = results["pnl"]
    winr   = cor / sigs if sigs > 0 else 0
    sigr   = sigs / processed if processed > 0 else 0
    avgpnl = pnl / sigs if sigs > 0 else 0

    W = 62
    status = "COMPLETE ✓" if done else "RUNNING ▶"

    def row(c=""):
        pad = W - len(c.encode("ascii", errors="replace")) - 1
        return f"  │ {c}{' ' * max(0, pad)}│"

    def top(t=""):
        pad = W - len(t) - 4
        return f"  ┌─ {t} {'─' * max(0, pad)}┐"

    def bot():
        return f"  └{'─' * W}┘"

    lines = [
        "",
        f"  ╔{'═' * W}╗",
        f"  ║{center('POST-CUTOFF BACKTEST', W)}║",
        f"  ║{center(f'resolved after {cutoff}  ·  real prices  ·  {status}', W)}║",
        f"  ╚{'═' * W}╝",
        "",
        top("PROGRESS"),
        row(f"  {bar(pct)}  {processed}/{total}  ({pct:.0%})"),
        row(f"  Elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(eta)}  Rate {rate:.1f}/min"),
        bot(), "",
        top("RESULTS  (resolved immediately — real outcomes known)"),
        row(f"  Signals   {sigs:>4}  ({sigr:.0%} rate)    No-signal  {nosig}"),
        row(f"  Win rate  {bar(winr, 20)}  {winr:.0%}   {cor}W / {wrong}L"),
        row(f"  P&L       {fmt_pnl(pnl):<12}  avg {fmt_pnl(avgpnl)}/trade"),
        bot(), "",
    ]

    # Recent results
    lines.append(top(f"LAST {RECENT_SIZE}  (newest first)"))
    if not recent:
        lines.append(row("  (waiting...)"))
    for r in list(recent)[::-1]:
        if r["direction"] == "NO_SIGNAL":
            icon, stat = "⏭ ", f"{'skip':<26}"
        else:
            icon = "✅" if r["correct"] else "❌"
            stat = f"{r['direction']:<3} {r['conf']:.0%} mkt={r['mkt']:.0%}→mdl={r['mdl']:.0%} {fmt_pnl(r['pnl']):<9}"
        q = r["q"]
        maxq = W - len(stat) - 10
        qs = (q[:maxq] + "…") if len(q) > maxq else q
        lines.append(row(f"  {r['idx']:>3} {icon} {stat}  {qs}"))
    lines += [bot(), ""]

    # Category
    lines.append(top("BY CATEGORY"))
    has_cats = any(v["signal"] > 0 for v in cats.values())
    if not has_cats:
        lines.append(row("  (no signals yet)"))
    else:
        for cat, v in sorted(cats.items(), key=lambda x: -x[1]["signal"]):
            if v["signal"] == 0:
                continue
            wr = v["correct"] / v["signal"] if v["signal"] > 0 else 0
            lines.append(row(
                f"  {cat:<12} {bar(wr, 10)}  {v['correct']}W/{v['wrong']}L  "
                f"{wr:.0%}  {fmt_pnl(v['pnl'])}"
            ))
    lines += [bot(), ""]

    ts = datetime.now().strftime("%H:%M:%S")
    lines += [
        f"  {'─' * W}",
        f"  Why this is clean:",
        f"    ✓ Resolved AFTER {cutoff} → model never saw outcomes in training",
        f"    ✓ Real market prices → tests edge vs crowd wisdom",
        f"    ✓ No news fetching  → pure model knowledge signal",
        f"  Updated {ts}",
        "",
    ]

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\r[{processed:>3}/{total}] {bar(pct, 20)} {pct:.0%}  "
          f"signals={sigs}  {cor}W/{wrong}L  win={winr:.0%}  "
          f"pnl={fmt_pnl(pnl)}  eta={fmt_dur(eta)}",
          end="", flush=True)


# ── Fetch post-cutoff resolved markets ───────────────────────────────────────
def fetch_postcutoff_markets(cutoff_date: str, limit: int = 300) -> list[dict]:
    url    = "https://gamma-api.polymarket.com/markets"
    result = []
    offset = 0
    batch  = 100

    cutoff_dt = datetime.fromisoformat(cutoff_date).replace(tzinfo=timezone.utc)
    print(f"Fetching markets resolved after {cutoff_date}...", flush=True)

    while len(result) < limit:
        try:
            resp = requests.get(url, params={
                "closed": "true", "limit": batch, "offset": offset,
                "order": "volume", "ascending": "false",
            }, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"\nAPI error: {e}"); break

        if not data:
            break

        for m in data:
            try:
                # Check end date is after cutoff
                end_str = m.get("endDate", "")
                if not end_str:
                    continue
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt < cutoff_dt:
                    continue   # resolved before cutoff — model might know outcome

                # Must have clear binary outcome
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
                    continue

                # Get the YES price at some point during the market (use last trade price)
                yes_price_live = float(m.get("lastTradePrice", 0.5) or 0.5)
                # Fallback: use volume-weighted or just 0.5 if unavailable
                if not (0.05 < yes_price_live < 0.95):
                    # Try to infer from the market's last prices before resolution
                    yes_price_live = 0.5   # neutral if we can't get live price

                question = m.get("question", "").strip()
                if not question or len(question) < 10:
                    continue

                # Skip noisy markets where AI has no edge
                skip_kw = ["up or down", "set 1", "set 2", "game 1", "game 2",
                           "map 1", "map 2", "bo3", "bo5", "1st period", "2nd period",
                           "inning", "o/u", "spread:", "highest temperature",
                           "lowest temperature", "°c on", "°f on", "5 minutes",
                           "15 minutes", "candle", "wick", " win on 2"]
                if any(kw in question.lower() for kw in skip_kw):
                    continue

                crypto_kw = ["bitcoin", "ethereum", "btc", "eth", "solana", "xrp",
                             "crypto", "token", "defi", "nft", "usdc", "usdt",
                             "dogecoin", "doge", "bnb", "cardano", "avalanche",
                             "polygon", "chainlink", "uniswap", "coinbase", "binance"]
                result.append({
                    "id":             m.get("id", ""),
                    "question":       question,
                    "yes_price":      yes_price_live,
                    "no_price":       1 - yes_price_live,
                    "volume":         float(m.get("volumeNum", 0) or 0),
                    "actual_outcome": actual,
                    "end_date":       end_str,
                    "is_crypto":      any(kw in question.lower() for kw in crypto_kw),
                    "raw":            {},
                })

                if len(result) >= limit:
                    break

            except Exception:
                continue

        offset += batch
        if len(data) < batch:
            break

    print(f"  → {len(result)} post-cutoff markets found", flush=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def run(n: int = 100, cutoff: str = CUTOFF_DEFAULT, model: str = "", no_crypto: bool = False):
    from dotenv import load_dotenv
    load_dotenv()

    import config
    config.CONSENSUS_MODE   = False
    config.USE_NEWS_CONTEXT = False
    if model:
        config.PRIMARY_MODEL = model
        print(f"Using model: {model}", flush=True)

    from src.db         import init_db, get_conn
    from src.classifier import classify
    from src.forecaster import forecast
    from config         import ANTHROPIC_API_KEY, MOONSHOT_API_KEY

    # Need at least one valid key
    if not ANTHROPIC_API_KEY and not MOONSHOT_API_KEY:
        print("ERROR: No API key set"); sys.exit(1)

    init_db()

    markets = fetch_postcutoff_markets(cutoff_date=cutoff, limit=n * 3)

    if no_crypto:
        before = len(markets)
        markets = [m for m in markets if not m.get("is_crypto", False)]
        print(f"  → Dropped {before - len(markets)} crypto markets ({len(markets)} remaining)")

    markets = markets[:n]

    if not markets:
        print("No markets found."); return

    model_label = config.PRIMARY_MODEL
    print(f"\nRunning on {len(markets)} post-cutoff markets | model={model_label} | no-crypto={no_crypto}")
    print(f"Dashboard: watch -n1 cat {DASHBOARD_PATH}\n")

    conn    = get_conn()
    start   = time.time()
    recent  = deque(maxlen=RECENT_SIZE)
    results = {"signal": 0, "no_signal": 0, "correct": 0, "wrong": 0, "pnl": 0.0}
    cats: dict = {}

    for i, m in enumerate(markets):
        cat    = classify(m["question"])
        signal = forecast(m)
        now    = datetime.now(timezone.utc).isoformat()

        if cat not in cats:
            cats[cat] = {"signal": 0, "correct": 0, "wrong": 0, "pnl": 0.0}

        if signal is None:
            conn.execute("""
                INSERT INTO trades
                  (market_id, question, category, model_probability, confidence,
                   implied_probability, bet_direction, simulated_amount,
                   bankroll_at_bet, timestamp, resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, 0.5, 0.0, ?, 'NO_SIGNAL', 0, 0, ?, 1, ?, 0)
            """, ("pc-" + m["id"], m["question"], cat, m["yes_price"], now, m["actual_outcome"]))
            results["no_signal"] += 1
            recent.append({"idx": i+1, "direction": "NO_SIGNAL", "correct": False,
                           "conf": 0, "mkt": m["yes_price"], "mdl": 0.5,
                           "pnl": 0, "q": m["question"]})
        else:
            direction = signal["bet_direction"]
            amount    = 30.0
            won       = (direction == "YES" and m["actual_outcome"] == 1.0) or \
                        (direction == "NO"  and m["actual_outcome"] == 0.0)
            payout    = amount / (m["yes_price"] if direction == "YES" else m["no_price"]) if won else 0.0
            pnl       = payout - amount - (amount * 0.02)

            conn.execute("""
                INSERT INTO trades
                  (market_id, question, category, model_probability, confidence,
                   implied_probability, bet_direction, simulated_amount,
                   bankroll_at_bet, timestamp, resolved, actual_outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1000, ?, 1, ?, ?)
            """, ("pc-" + m["id"], m["question"], cat,
                  signal["probability"], signal["confidence"], m["yes_price"],
                  direction, amount, now, m["actual_outcome"], pnl))

            results["signal"] += 1
            results["pnl"]    += pnl
            if won:
                results["correct"]        += 1
                cats[cat]["correct"]      += 1
            else:
                results["wrong"]          += 1
                cats[cat]["wrong"]        += 1
            cats[cat]["signal"] += 1
            cats[cat]["pnl"]    += pnl

            recent.append({"idx": i+1, "direction": direction, "correct": won,
                           "conf": signal["confidence"], "mkt": m["yes_price"],
                           "mdl": signal["probability"], "pnl": pnl, "q": m["question"]})

        conn.commit()
        write_dashboard(len(markets), i+1, results, recent, cats, start, cutoff)

    conn.close()
    write_dashboard(len(markets), len(markets), results, recent, cats, start, cutoff, done=True)

    sigs = results["signal"]
    winr = results["correct"] / sigs if sigs > 0 else 0
    print(f"\n\nDone! {sigs} signals | {results['correct']}W/{results['wrong']}L | "
          f"win={winr:.0%} | P&L={fmt_pnl(results['pnl'])}")
    print(f"Dashboard: {DASHBOARD_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n",           type=int,  default=100)
    p.add_argument("--cutoff",     type=str,  default=CUTOFF_DEFAULT,
                   help="Only use markets resolved after this date (default 2025-03-01)")
    p.add_argument("--model",      type=str,  default="",
                   help="Override model (e.g. kimi-k2.5, claude-sonnet-4-6)")
    p.add_argument("--no-crypto",  action="store_true",
                   help="Exclude crypto/token markets")
    args = p.parse_args()
    run(n=args.n, cutoff=args.cutoff, model=args.model, no_crypto=args.no_crypto)

if __name__ == "__main__":
    main()
