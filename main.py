#!/usr/bin/env python3
"""
Polymarket AI Agent — CLI entrypoint.

Commands:
  run         Fetch markets → forecast → paper trade
  status      Show bankroll, open positions, P&L summary
  resolve     Check resolved markets and update trade outcomes
  calibration Show Phase 1 calibration metrics
"""

import argparse
import json
import logging
import os
import sys

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── Rich UI ────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich         import print as rprint
    RICH = True
except ImportError:
    RICH = False
    Console = None  # type: ignore

console = Console() if RICH else None


def _print(msg: str) -> None:
    if RICH:
        console.print(msg)  # type: ignore
    else:
        print(msg)


def cmd_run(args: argparse.Namespace) -> None:
    """Fetch markets, forecast, and paper trade."""
    from config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        _print("[red]ERROR[/red]: ANTHROPIC_API_KEY not set. Export it and retry." if RICH else
               "ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    from src.db          import init_db
    from src.markets     import fetch_active_markets
    from src.classifier  import classify
    from src.forecaster  import forecast
    from src.paper_trader import place_paper_bet

    init_db()
    _print("\n[bold cyan]── Polymarket Agent: RUN ──[/bold cyan]\n" if RICH else
           "\n── Polymarket Agent: RUN ──\n")

    markets = fetch_active_markets()
    if not markets:
        _print("[yellow]No qualifying markets found.[/yellow]" if RICH else
               "No qualifying markets found.")
        return

    _print(f"[green]Fetched {len(markets)} markets[/green]" if RICH else
           f"Fetched {len(markets)} markets")

    signal_records: list[dict] = []
    for market in markets:
        category = classify(market["question"])
        _print(f"\n[dim]→ [{category}] {market['question'][:80]}[/dim]" if RICH else
               f"\n→ [{category}] {market['question'][:80]}")

        signal = forecast(market)
        if signal is None:
            _print("  [dim]No signal[/dim]" if RICH else "  No signal")
            continue

        trade_id = place_paper_bet(market, signal, category)
        if trade_id:
            direction = signal["bet_direction"]
            colour    = "green" if direction == "YES" else "red"
            _print(
                f"  [bold {colour}]{direction}[/bold {colour}] "
                f"prob={signal['probability']:.2f} "
                f"conf={signal['confidence']:.2f} "
                f"edge={signal['edge']:.3f} "
                f"[trade #{trade_id}]"
                if RICH else
                f"  {direction} prob={signal['probability']:.2f} "
                f"conf={signal['confidence']:.2f} edge={signal['edge']:.3f} "
                f"[trade #{trade_id}]"
            )
            # Collect for notifier summary
            from src.paper_trader import kelly_size, _get_bankroll, get_conn  # type: ignore
            conn = get_conn()
            bk = _get_bankroll(conn)
            conn.close()
            signal_records.append({
                "trade_id":        trade_id,
                "question":        market["question"],
                "bet_direction":   direction,
                "probability":     signal["probability"],
                "confidence":      signal["confidence"],
                "edge":            signal["edge"],
                "simulated_amount": kelly_size(bk, signal["confidence"]),
            })

    _print(f"\n[bold]Done. {len(signal_records)}/{len(markets)} markets generated signals.[/bold]\n"
           if RICH else f"\nDone. {len(signal_records)}/{len(markets)} markets generated signals.\n")

    # Phase 2: structured run summary via notifier
    try:
        from src.notifier        import run_summary
        from src.paper_trader    import get_portfolio_summary
        summary = get_portfolio_summary()
        run_summary(markets, signal_records, summary["bankroll"])
    except Exception:
        pass  # notifier is best-effort


def cmd_status(args: argparse.Namespace) -> None:
    """Print portfolio summary and open positions."""
    from src.db           import init_db
    from src.paper_trader import get_portfolio_summary, get_open_positions

    init_db()
    summary   = get_portfolio_summary()
    positions = get_open_positions()

    _print("\n[bold cyan]── Portfolio Status ──[/bold cyan]\n" if RICH else
           "\n── Portfolio Status ──\n")

    if RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column("", style="dim")
        t.add_column("", style="bold")
        t.add_row("Bankroll",     f"${summary['bankroll']:.2f}")
        t.add_row("Open bets",    str(summary['open_bets']))
        t.add_row("Closed bets",  str(summary['closed_bets']))
        t.add_row("Total P&L",    f"${summary['total_pnl']:+.2f}")
        t.add_row("ROI",          f"{summary['roi_pct']:.2f}%")
        console.print(t)  # type: ignore
    else:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    if positions:
        _print("\n[bold]Open positions:[/bold]" if RICH else "\nOpen positions:")
        if RICH:
            pt = Table("ID", "#Question", "Direction", "Amount", "Confidence", "Timestamp")
            for p in positions:
                pt.add_row(
                    str(p["id"]),
                    p["question"][:50],
                    p["bet_direction"],
                    f"${p['simulated_amount']:.2f}",
                    f"{p['confidence']:.2f}",
                    p["timestamp"][:19],
                )
            console.print(pt)  # type: ignore
        else:
            for p in positions:
                print(f"  #{p['id']} {p['bet_direction']} ${p['simulated_amount']:.2f} "
                      f"| {p['question'][:60]}")


def cmd_resolve(args: argparse.Namespace) -> None:
    """Poll Polymarket for resolved markets and update trade records."""
    from src.db      import init_db
    from src.tracker import resolve_open_trades

    init_db()
    _print("\n[bold cyan]── Resolving open trades ──[/bold cyan]\n" if RICH else
           "\n── Resolving open trades ──\n")
    count = resolve_open_trades()
    _print(f"[green]Resolved {count} trade(s).[/green]" if RICH else
           f"Resolved {count} trade(s).")


def cmd_calibration(args: argparse.Namespace) -> None:
    """Print Phase 1 calibration report."""
    from src.db          import init_db
    from src.calibration import generate_report

    init_db()
    _print("\n[bold cyan]── Calibration Report (Phase 1) ──[/bold cyan]\n" if RICH else
           "\n── Calibration Report (Phase 1) ──\n")

    report = generate_report()

    if "error" in report:
        _print(f"[yellow]{report['error']}[/yellow]" if RICH else report["error"])
        return

    # Summary table
    if RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column("", style="dim")
        t.add_column("", style="bold")
        t.add_row("Total trades",  str(report["total_trades"]))
        t.add_row("Total wagered", f"${report['total_wagered']:.2f}")
        t.add_row("Total P&L",     f"${report['total_pnl']:+.2f}")
        t.add_row("ROI",           f"{report['roi_pct']:.2f}%")
        t.add_row("Brier score",   f"{report['brier_score']:.4f}  (random=0.25, perfect=0.00)")
        t.add_row("Max drawdown",  f"${report['max_drawdown']:.2f}")
        console.print(t)  # type: ignore
    else:
        for k, v in report.items():
            if k not in ("calibration_curve", "roi_by_category", "edge_vs_market"):
                print(f"  {k}: {v}")

    # Edge vs market
    edge = report.get("edge_vs_market", {})
    if edge:
        delta_colour = "green" if edge.get("delta", 0) > 0 else "red"
        _print(
            f"\n[bold]Edge vs market:[/bold] "
            f"win_rate={edge.get('win_rate',0):.2f} "
            f"implied={edge.get('avg_implied_prob',0):.2f} "
            f"delta=[{delta_colour}]{edge.get('delta',0):+.3f}[/{delta_colour}]"
            if RICH else
            f"\nEdge vs market: win_rate={edge.get('win_rate',0):.2f} "
            f"implied={edge.get('avg_implied_prob',0):.2f} "
            f"delta={edge.get('delta',0):+.3f}"
        )

    # ROI by category
    roi_cat = report.get("roi_by_category", {})
    if roi_cat:
        _print("\n[bold]ROI by category:[/bold]" if RICH else "\nROI by category:")
        if RICH:
            ct = Table("Category", "Trades", "Wagered", "P&L", "ROI%", "Win%")
            for cat, stats in sorted(roi_cat.items(), key=lambda x: -x[1]["roi_pct"]):
                colour = "green" if stats["roi_pct"] > 0 else "red"
                ct.add_row(
                    cat,
                    str(stats["count"]),
                    f"${stats['wagered']:.2f}",
                    f"${stats['pnl']:+.2f}",
                    f"[{colour}]{stats['roi_pct']:.1f}%[/{colour}]",
                    f"{stats['win_rate']*100:.1f}%",
                )
            console.print(ct)  # type: ignore
        else:
            for cat, stats in sorted(roi_cat.items(), key=lambda x: -x[1]["roi_pct"]):
                print(f"  {cat}: ROI={stats['roi_pct']:.1f}% win={stats['win_rate']*100:.1f}%")

    # Calibration curve
    curve = report.get("calibration_curve", [])
    if curve:
        _print("\n[bold]Calibration curve (predicted vs actual):[/bold]" if RICH else
               "\nCalibration curve (predicted vs actual):")
        if RICH:
            curve_t = Table("Bucket", "Predicted", "Actual", "Count", "Delta")
            for b in curve:
                delta  = b["actual_mean"] - b["predicted_mean"]
                colour = "green" if abs(delta) < 0.05 else "yellow" if abs(delta) < 0.10 else "red"
                curve_t.add_row(
                    f"{b['bucket_low']:.0%}–{b['bucket_high']:.0%}",
                    f"{b['predicted_mean']:.2f}",
                    f"{b['actual_mean']:.2f}",
                    str(b["count"]),
                    f"[{colour}]{delta:+.3f}[/{colour}]",
                )
            console.print(curve_t)  # type: ignore
        else:
            for b in curve:
                delta = b["actual_mean"] - b["predicted_mean"]
                print(f"  {b['bucket_low']:.0%}–{b['bucket_high']:.0%}: "
                      f"pred={b['predicted_mean']:.2f} actual={b['actual_mean']:.2f} "
                      f"Δ={delta:+.3f} (n={b['count']})")

    # Save report
    with open("calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    _print("\n[dim]Report saved to calibration_report.json[/dim]" if RICH else
           "\nReport saved to calibration_report.json")


def cmd_history(args: argparse.Namespace) -> None:
    """Show the last N resolved trades with outcomes and P&L."""
    from src.db import init_db, get_conn

    init_db()
    n = getattr(args, "n", 20)

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, question, category, bet_direction,
               model_probability, implied_probability,
               confidence, simulated_amount,
               actual_outcome, profit_loss, timestamp
        FROM trades
        WHERE resolved = 1
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()

    _print(f"\n[bold cyan]── Trade History (last {n} resolved) ──[/bold cyan]\n"
           if RICH else f"\n── Trade History (last {n} resolved) ──\n")

    if not rows:
        _print("[yellow]No resolved trades yet.[/yellow]" if RICH else "No resolved trades yet.")
        return

    # Aggregate stats
    wins     = sum(1 for r in rows if r["profit_loss"] and r["profit_loss"] > 0)
    total_pl = sum((r["profit_loss"] or 0) for r in rows)
    wagered  = sum((r["simulated_amount"] or 0) for r in rows)
    roi      = (total_pl / wagered * 100) if wagered > 0 else 0.0

    if RICH:
        t = Table(
            "ID", "Dir", "Question", "Prob", "Imp", "Outcome", "P&L",
            box=None, show_lines=False, header_style="bold",
        )
        for r in rows:
            direction = r["bet_direction"]
            outcome   = r["actual_outcome"]
            pnl       = r["profit_loss"] or 0.0
            won       = pnl > 0
            dir_col   = "green" if direction == "YES" else "red"
            pnl_col   = "green" if won else "red"
            out_str   = ("YES" if outcome == 1.0 else "NO") if outcome is not None else "?"
            t.add_row(
                str(r["id"]),
                f"[{dir_col}]{direction}[/{dir_col}]",
                (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"],
                f"{r['model_probability']:.2f}",
                f"{r['implied_probability']:.2f}",
                out_str,
                f"[{pnl_col}]${pnl:+.2f}[/{pnl_col}]",
            )
        console.print(t)  # type: ignore
        console.print(  # type: ignore
            f"\n[bold]Summary:[/bold] "
            f"win_rate={wins}/{len(rows)} ({wins/len(rows)*100:.1f}%)  "
            f"total_P&L=[{'green' if total_pl >= 0 else 'red'}]${total_pl:+.2f}[/{'green' if total_pl >= 0 else 'red'}]  "
            f"ROI=[{'green' if roi >= 0 else 'red'}]{roi:.2f}%[/{'green' if roi >= 0 else 'red'}]\n"
        )
    else:
        for r in rows:
            pnl     = r["profit_loss"] or 0.0
            outcome = ("YES" if r["actual_outcome"] == 1.0 else "NO") if r["actual_outcome"] is not None else "?"
            print(
                f"  #{r['id']:4d} {r['bet_direction']:3s} → {outcome:3s} "
                f"P&L=${pnl:+.2f}  prob={r['model_probability']:.2f} "
                f"| {r['question'][:55]}"
            )
        print(f"\n  Win rate: {wins}/{len(rows)} ({wins/len(rows)*100:.1f}%)  "
              f"Total P&L: ${total_pl:+.2f}  ROI: {roi:.2f}%\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket AI Agent — paper trading + calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("run",         help="Fetch markets → forecast → paper trade")
    sub.add_parser("status",      help="Show bankroll and open positions")
    sub.add_parser("resolve",     help="Resolve completed markets and update P&L")
    sub.add_parser("calibration", help="Show Phase 1 calibration metrics")

    # Phase 2 commands
    hist_p = sub.add_parser("history", help="Show last N resolved trades with outcomes and P&L")
    hist_p.add_argument("-n", type=int, default=20, help="Number of trades to show (default: 20)")

    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "resolve":
        cmd_resolve(args)
    elif args.cmd == "calibration":
        cmd_calibration(args)
    elif args.cmd == "history":
        cmd_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
