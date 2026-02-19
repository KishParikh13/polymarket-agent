"""
Notifier (Phase 2) — structured run summary output using Rich when available.

After each `python main.py run`, prints a formatted summary table showing:
  - All markets evaluated
  - Which ones generated signals (and in which direction)
  - Kelly-sized bet amounts
  - Aggregate stats for the run
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import box as rbox
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore


# ── Public API ─────────────────────────────────────────────────────────────────

def run_summary(
    markets: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    bankroll: float,
) -> None:
    """
    Print a formatted summary of a completed `run` pass.

    Args:
        markets:  All markets that were evaluated this run.
        signals:  List of signal dicts that resulted in trades. Each dict should
                  contain: question, bet_direction, probability, confidence,
                  edge, simulated_amount, trade_id.
        bankroll: Current bankroll after this run.
    """
    total    = len(markets)
    n_trades = len(signals)
    deployed = sum(s.get("simulated_amount", 0) for s in signals)

    if _RICH and _console:
        _rich_summary(total, n_trades, deployed, bankroll, signals)
    else:
        _plain_summary(total, n_trades, deployed, bankroll, signals)


# ── Rich renderer ──────────────────────────────────────────────────────────────

def _rich_summary(
    total: int,
    n_trades: int,
    deployed: float,
    bankroll: float,
    signals: list[dict[str, Any]],
) -> None:
    assert _console is not None

    header = (
        f"[bold cyan]Run Complete[/bold cyan]  "
        f"markets={total}  trades=[bold]{n_trades}[/bold]  "
        f"deployed=[green]${deployed:.2f}[/green]  "
        f"bankroll=[bold]${bankroll:.2f}[/bold]"
    )
    _console.print(Panel(header, expand=False))

    if not signals:
        _console.print("[dim]  No signals generated this run.[/dim]\n")
        return

    t = Table(
        "ID", "Direction", "Question", "Prob", "Conf", "Edge", "Bet ($)",
        box=rbox.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold",
    )
    for s in signals:
        direction = s.get("bet_direction", "?")
        colour    = "green" if direction == "YES" else "red"
        t.add_row(
            str(s.get("trade_id", "—")),
            f"[bold {colour}]{direction}[/bold {colour}]",
            (s.get("question", "")[:55] + "…") if len(s.get("question", "")) > 55 else s.get("question", ""),
            f"{s.get('probability', 0):.2f}",
            f"{s.get('confidence', 0):.2f}",
            f"{s.get('edge', 0):.3f}",
            f"${s.get('simulated_amount', 0):.2f}",
        )
    _console.print(t)
    _console.print()


# ── Plain-text renderer ────────────────────────────────────────────────────────

def _plain_summary(
    total: int,
    n_trades: int,
    deployed: float,
    bankroll: float,
    signals: list[dict[str, Any]],
) -> None:
    print(f"\n── Run Complete ──")
    print(f"  Markets evaluated : {total}")
    print(f"  Trades placed     : {n_trades}")
    print(f"  Capital deployed  : ${deployed:.2f}")
    print(f"  Bankroll          : ${bankroll:.2f}")
    if signals:
        print("\n  Signals:")
        for s in signals:
            direction = s.get("bet_direction", "?")
            print(
                f"    #{s.get('trade_id', '?')} {direction:3s} "
                f"prob={s.get('probability', 0):.2f} "
                f"conf={s.get('confidence', 0):.2f} "
                f"edge={s.get('edge', 0):.3f} "
                f"bet=${s.get('simulated_amount', 0):.2f} "
                f"| {s.get('question', '')[:55]}"
            )
    else:
        print("  No signals generated this run.")
    print()
