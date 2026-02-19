"""
Polymarket live order execution via CLOB client.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config import (
    GAMMA_API_BASE,
    POLY_API_KEY,
    POLY_API_PASSPHRASE,
    POLY_API_SECRET,
    POLY_CHAIN_ID,
    POLY_HOST,
    POLY_PRIVATE_KEY,
)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    try:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
    except Exception:
        AssetType = None
        BalanceAllowanceParams = None
    try:
        from py_clob_client.order_builder.constants import BUY
    except Exception:
        BUY = "BUY"
    _CLOB_IMPORT_ERROR = None
except Exception as import_err:
    ClobClient = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]
    OrderType = None  # type: ignore[assignment]
    AssetType = None
    BalanceAllowanceParams = None
    BUY = "BUY"
    _CLOB_IMPORT_ERROR = import_err

logger = logging.getLogger(__name__)

LIVE_ORDERS_PATH = Path("data/live_orders.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _loads_if_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


class PolymarketExecutor:
    def __init__(self):
        if not POLY_PRIVATE_KEY:
            raise ValueError(
                "POLY_PRIVATE_KEY is not set. Add it to your environment to enable live trading."
            )
        if ClobClient is None or OrderArgs is None or OrderType is None:
            raise RuntimeError(
                f"py-clob-client is not available: {_CLOB_IMPORT_ERROR}. "
                "Install it with: pip install py-clob-client"
            )

        self.host = POLY_HOST
        self.chain_id = POLY_CHAIN_ID
        self.private_key = POLY_PRIVATE_KEY

        self.client = ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
        )

        if POLY_API_KEY and POLY_API_SECRET and POLY_API_PASSPHRASE:
            self.client.set_api_creds(
                {
                    "key": POLY_API_KEY,
                    "secret": POLY_API_SECRET,
                    "passphrase": POLY_API_PASSPHRASE,
                }
            )
        else:
            logger.warning(
                "POLY_API_* credentials are not fully set. Live order placement may fail "
                "if API headers are required."
            )

        LIVE_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _append_order_log(self, payload: dict[str, Any]) -> None:
        try:
            with LIVE_ORDERS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as e:
            logger.error(f"Failed to append live order log: {e}")

    def _extract_order_id(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("order_id", "orderID", "id"):
                value = response.get(key)
                if value:
                    return str(value)
            for nested in ("order", "data", "result"):
                payload = response.get(nested)
                if isinstance(payload, dict):
                    for key in ("order_id", "orderID", "id"):
                        value = payload.get(key)
                        if value:
                            return str(value)
        return ""

    def _extract_price(self, payload: Any) -> float | None:
        if payload is None:
            return None
        if isinstance(payload, (float, int)):
            return float(payload)
        if isinstance(payload, str):
            return _to_float(payload, default=0.0) or None
        if isinstance(payload, dict):
            for key in ("price", "mid", "midpoint", "last_price", "last"):
                if key in payload:
                    v = _to_float(payload.get(key), default=0.0)
                    if v > 0:
                        return v
        return None

    def _extract_token_ids(self, source: Any) -> tuple[str | None, str | None]:
        if not isinstance(source, dict):
            return None, None

        yes_token = source.get("yes_token_id")
        no_token = source.get("no_token_id")
        if yes_token and no_token:
            return str(yes_token), str(no_token)

        token_ids = _loads_if_json(source.get("token_ids") or source.get("tokenIds"))
        if isinstance(token_ids, dict):
            yes_token = token_ids.get("YES") or token_ids.get("yes")
            no_token = token_ids.get("NO") or token_ids.get("no")
            if yes_token and no_token:
                return str(yes_token), str(no_token)
        if isinstance(token_ids, list) and len(token_ids) >= 2:
            return str(token_ids[0]), str(token_ids[1])

        clob_token_ids = _loads_if_json(
            source.get("clobTokenIds") or source.get("outcomeTokenIds")
        )
        if isinstance(clob_token_ids, list) and len(clob_token_ids) >= 2:
            return str(clob_token_ids[0]), str(clob_token_ids[1])

        tokens = _loads_if_json(source.get("tokens"))
        if isinstance(tokens, list) and len(tokens) >= 2:
            def _token_id(tok: Any) -> str | None:
                if isinstance(tok, dict):
                    return str(
                        tok.get("token_id")
                        or tok.get("tokenId")
                        or tok.get("id")
                        or tok.get("asset_id")
                        or ""
                    ) or None
                if isinstance(tok, str):
                    return tok
                return None

            yes = _token_id(tokens[0])
            no = _token_id(tokens[1])
            if yes and no:
                return yes, no

        return None, None

    def _fetch_market_detail(self, market_id: str) -> dict[str, Any] | None:
        try:
            market = self.client.get_market(market_id)
            if isinstance(market, dict):
                return market
        except Exception:
            pass

        candidates = [
            (f"{GAMMA_API_BASE}/markets/{market_id}", {}),
            (f"{GAMMA_API_BASE}/markets", {"condition_ids": market_id, "limit": 1}),
            (f"{GAMMA_API_BASE}/markets", {"conditionId": market_id, "limit": 1}),
        ]
        for url, params in candidates:
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if isinstance(data, list):
                    if data and isinstance(data[0], dict):
                        return data[0]
                elif isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def _resolve_token_id(self, market: dict[str, Any], direction: str) -> str | None:
        yes_token, no_token = self._extract_token_ids(market)
        if yes_token and no_token:
            return yes_token if direction == "YES" else no_token

        market_id = str(
            market.get("condition_id")
            or market.get("conditionId")
            or market.get("market_id")
            or market.get("id")
            or ""
        )
        if not market_id:
            return None

        details = self._fetch_market_detail(market_id)
        if details:
            yes_token, no_token = self._extract_token_ids(details)
            if yes_token and no_token:
                return yes_token if direction == "YES" else no_token

        # Last-resort fallback when caller already passed a token_id as market["id"].
        if market_id.isdigit():
            logger.warning(
                "Using market id as token id fallback for '%s': %s",
                market.get("question", "<unknown>"),
                market_id,
            )
            return market_id
        return None

    def _read_placed_orders(self) -> list[dict[str, Any]]:
        if not LIVE_ORDERS_PATH.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with LIVE_ORDERS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("status") == "placed":
                        rows.append(obj)
        except Exception as e:
            logger.error(f"Failed to read live order log: {e}")
        return rows

    def _market_price(self, token_id: str, fallback: float) -> float:
        for method_name in ("get_midpoint", "get_last_trade_price", "get_price"):
            method = getattr(self.client, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "get_price":
                    payload = method(token_id=token_id, side=BUY)
                else:
                    payload = method(token_id=token_id)
                price = self._extract_price(payload)
                if price is not None and price > 0:
                    return price
            except Exception:
                continue
        return fallback

    def _usdc_balance(self) -> float:
        if BalanceAllowanceParams is not None and AssetType is not None:
            method = getattr(self.client, "get_balance_allowance", None)
            if method is not None:
                try:
                    params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    payload = method(params=params)
                    if isinstance(payload, dict):
                        for key in (
                            "balance",
                            "available",
                            "available_balance",
                            "freeCollateral",
                        ):
                            amount = _to_float(payload.get(key), default=-1.0)
                            if amount >= 0:
                                return amount
                except Exception:
                    pass

        for method_name in ("get_collateral_balance", "get_balance"):
            method = getattr(self.client, method_name, None)
            if method is None:
                continue
            try:
                payload = method()
                amount = self._extract_price(payload)
                if amount is not None and amount >= 0:
                    return amount
            except Exception:
                continue
        return 0.0

    def place_order(self, market: dict, signal: dict, amount_usdc: float) -> dict:
        """
        Place a limit order on Polymarket CLOB.
        """
        question = str(market.get("question", ""))
        direction = str(signal.get("bet_direction", "")).upper()
        market_id = str(market.get("id", ""))
        now = _now_iso()

        result: dict[str, Any] = {
            "order_id": "",
            "market_id": market_id,
            "question": question,
            "direction": direction,
            "price": 0.0,
            "size": 0.0,
            "amount_usdc": float(amount_usdc),
            "status": "error",
            "error": None,
            "timestamp": now,
        }

        try:
            if direction not in {"YES", "NO"}:
                raise ValueError(f"Invalid bet_direction: {direction!r}")
            if amount_usdc <= 0:
                raise ValueError(f"amount_usdc must be > 0 (got {amount_usdc})")

            yes_price = _to_float(market.get("yes_price"), default=0.0)
            no_price = _to_float(market.get("no_price"), default=max(0.0, 1.0 - yes_price))
            price = yes_price if direction == "YES" else no_price
            if not (0 < price < 1):
                raise ValueError(f"Invalid limit price for {direction}: {price}")

            token_id = self._resolve_token_id(market, direction)
            if not token_id:
                raise ValueError(f"Could not resolve {direction} token_id for market")

            size = round(float(amount_usdc) / price, 6)
            if size <= 0:
                raise ValueError("Computed order size is zero")

            args = OrderArgs(
                token_id=str(token_id),
                price=float(price),
                size=float(size),
                side=BUY,
            )
            signed_order = self.client.create_order(args)
            response = self.client.post_order(signed_order, OrderType.GTC)

            result.update(
                {
                    "order_id": self._extract_order_id(response),
                    "price": float(price),
                    "size": float(size),
                    "status": "placed",
                    "error": None,
                    "token_id": str(token_id),
                }
            )
            logger.info(
                "Live order placed | %s | %s %.4f @ %.4f",
                question[:80],
                direction,
                size,
                price,
            )
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error("Live order failed for '%s': %s", question[:120], e)
        finally:
            self._append_order_log(result)

        return result

    def get_position(self, market_id: str) -> dict:
        """Get current position and PnL for a market. Returns {} if no position."""
        orders = [
            o
            for o in self._read_placed_orders()
            if str(o.get("market_id", "")) == str(market_id)
        ]
        if not orders:
            return {}

        latest = orders[-1]
        direction = str(latest.get("direction", ""))
        same_side = [o for o in orders if str(o.get("direction", "")) == direction]
        if not same_side:
            return {}

        total_size = sum(_to_float(o.get("size")) for o in same_side)
        if total_size <= 0:
            return {}
        total_cost = sum(_to_float(o.get("price")) * _to_float(o.get("size")) for o in same_side)
        entry_price = total_cost / total_size

        token_id = str(latest.get("token_id") or "") or self._resolve_token_id(
            {"id": market_id, "question": latest.get("question", "")},
            direction,
        )
        current_price = self._market_price(token_id, entry_price) if token_id else entry_price
        unrealized_pnl = (current_price - entry_price) * total_size

        return {
            "market_id": str(market_id),
            "question": str(latest.get("question", "")),
            "direction": direction,
            "entry_price": round(entry_price, 6),
            "current_price": round(current_price, 6),
            "size": round(total_size, 6),
            "unrealized_pnl": round(unrealized_pnl, 6),
        }

    def get_portfolio(self) -> dict:
        """
        Return portfolio summary.
        """
        orders = self._read_placed_orders()
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for o in orders:
            key = (str(o.get("market_id", "")), str(o.get("direction", "")))
            if key not in grouped:
                grouped[key] = {
                    "market_id": key[0],
                    "question": str(o.get("question", "")),
                    "direction": key[1],
                    "size": 0.0,
                    "cost": 0.0,
                    "token_id": str(o.get("token_id") or ""),
                }
            size = _to_float(o.get("size"))
            price = _to_float(o.get("price"))
            grouped[key]["size"] += size
            grouped[key]["cost"] += price * size
            if not grouped[key]["token_id"]:
                grouped[key]["token_id"] = str(o.get("token_id") or "")

        positions: list[dict[str, Any]] = []
        total_unrealized = 0.0
        for _, pos in grouped.items():
            if pos["size"] <= 0:
                continue
            entry_price = pos["cost"] / pos["size"]
            token_id = pos["token_id"] or self._resolve_token_id(
                {"id": pos["market_id"], "question": pos["question"]},
                pos["direction"],
            )
            current_price = self._market_price(token_id, entry_price) if token_id else entry_price
            unrealized = (current_price - entry_price) * pos["size"]
            total_unrealized += unrealized
            positions.append(
                {
                    "market_id": pos["market_id"],
                    "question": pos["question"],
                    "direction": pos["direction"],
                    "entry_price": round(entry_price, 6),
                    "current_price": round(current_price, 6),
                    "size": round(pos["size"], 6),
                    "unrealized_pnl": round(unrealized, 6),
                }
            )

        return {
            "usdc_balance": round(self._usdc_balance(), 6),
            "positions": positions,
            "total_unrealized_pnl": round(total_unrealized, 6),
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        try:
            response = self.client.cancel(order_id)
            if isinstance(response, bool):
                return response
            if isinstance(response, dict):
                if response.get("success") is True:
                    return True
                if response.get("status") in {"cancelled", "canceled"}:
                    return True
                cancelled = response.get("canceled") or response.get("cancelled")
                if isinstance(cancelled, list) and order_id in cancelled:
                    return True
            return bool(response)
        except Exception as e:
            logger.error("Cancel order failed for %s: %s", order_id, e)
            return False


def execute_signal(market: dict, signal: dict, amount_usdc: float = 30.0) -> dict:
    """Convenience wrapper — creates executor and places one order."""
    try:
        executor = PolymarketExecutor()
        return executor.place_order(market=market, signal=signal, amount_usdc=amount_usdc)
    except Exception as e:
        question = str(market.get("question", ""))
        logger.error("execute_signal failed for '%s': %s", question[:120], e)
        return {
            "order_id": "",
            "market_id": str(market.get("id", "")),
            "question": question,
            "direction": str(signal.get("bet_direction", "")).upper(),
            "price": 0.0,
            "size": 0.0,
            "amount_usdc": float(amount_usdc),
            "status": "error",
            "error": str(e),
            "timestamp": _now_iso(),
        }
