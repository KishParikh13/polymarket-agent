"""
LLM Forecaster — asks Claude to estimate market probabilities.
Supports single-model and two-model consensus modes.
"""

import json
import logging
import re
from typing import Any

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    CONSENSUS_MODE,
    CONSENSUS_TOLERANCE,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are an expert prediction market analyst.
Given a market question and current market price, estimate the true probability that the answer is YES.

Rules:
- Be concise but rigorous in your reasoning
- Account for current market price as a signal (but not the only one)
- Output ONLY valid JSON with these fields:
  {
    "probability": <float 0.0–1.0>,
    "confidence": <float 0.0–1.0, how sure you are in your estimate>,
    "reasoning": "<one paragraph>"
  }
- If you have very low information, set confidence < 0.4
- Do NOT output anything outside the JSON block"""


def _ask_model(model: str, question: str, yes_price: float) -> dict[str, Any] | None:
    """Call one model and parse its JSON response."""
    prompt = (
        f"Market question: {question}\n"
        f"Current YES price (market-implied probability): {yes_price:.2%}\n\n"
        "Estimate the true probability this resolves YES. Output JSON only."
    )
    try:
        resp = _get_client().messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()

        # Extract JSON block (handles ```json ... ``` wrappers)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning(f"No JSON in model response: {text[:200]}")
            return None

        result = json.loads(match.group())
        prob = float(result["probability"])
        conf = float(result["confidence"])

        if not (0.0 <= prob <= 1.0 and 0.0 <= conf <= 1.0):
            logger.warning(f"Out-of-range values: prob={prob}, conf={conf}")
            return None

        return {
            "probability": prob,
            "confidence":  conf,
            "reasoning":   result.get("reasoning", ""),
            "model":       model,
        }

    except Exception as e:
        logger.error(f"Model {model} failed: {e}")
        return None


def forecast(market: dict[str, Any]) -> dict[str, Any] | None:
    """
    Forecast a market. Returns a signal dict or None if no trade should be made.

    Signal dict keys:
      probability, confidence, reasoning, bet_direction, model(s)
    """
    question  = market["question"]
    yes_price = market["yes_price"]

    primary = _ask_model(PRIMARY_MODEL, question, yes_price)
    if primary is None:
        return None

    if CONSENSUS_MODE:
        secondary = _ask_model(SECONDARY_MODEL, question, yes_price)
        if secondary is None:
            logger.info(f"Secondary model failed — skipping for safety: {question[:60]}")
            return None
        if abs(primary["probability"] - secondary["probability"]) > CONSENSUS_TOLERANCE:
            logger.info(
                f"Models disagree ({primary['probability']:.2f} vs "
                f"{secondary['probability']:.2f}) — no bet: {question[:60]}"
            )
            return None
        # Average the two estimates
        probability = (primary["probability"] + secondary["probability"]) / 2
        confidence  = min(primary["confidence"], secondary["confidence"])
    else:
        probability = primary["probability"]
        confidence  = primary["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            f"Low confidence ({confidence:.2f}) — skipping: {question[:60]}"
        )
        return None

    # Determine bet direction: bet YES if model > market, NO if model < market
    if probability > yes_price:
        bet_direction   = "YES"
        edge            = probability - yes_price
        implied_prob    = yes_price
    else:
        bet_direction   = "NO"
        edge            = (1 - probability) - market["no_price"]
        implied_prob    = yes_price

    if edge <= 0:
        logger.info(f"No edge ({edge:.3f}) — skipping: {question[:60]}")
        return None

    return {
        "probability":      probability,
        "confidence":       confidence,
        "reasoning":        primary["reasoning"],
        "bet_direction":    bet_direction,
        "implied_prob":     implied_prob,
        "edge":             edge,
        "models":           [PRIMARY_MODEL] + ([SECONDARY_MODEL] if CONSENSUS_MODE else []),
    }
