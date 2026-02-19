"""
LLM Forecaster — asks Claude to estimate market probabilities.
Supports single-model and two-model consensus modes.
"""

import json
import logging
import os
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
    USE_NEWS_CONTEXT,
    NEWS_MAX_HEADLINES,
)

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_openai_client: Any | None = None

# Models that route through Moonshot (OpenAI-compatible) instead of Anthropic
MOONSHOT_MODELS = {"kimi-k2.5", "kimi-k2-thinking", "moonshot/kimi-k2.5",
                   "kimi-k2-turbo-preview", "kimi-k2-0711-preview", "kimi-latest"}


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    """OpenAI-compatible client for Moonshot/Kimi models."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                api_key=os.getenv("MOONSHOT_API_KEY", ""),
                base_url="https://api.moonshot.ai/v1",
            )
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
    return _openai_client


def _is_moonshot(model: str) -> bool:
    return model in MOONSHOT_MODELS or model.startswith("moonshot/") or model.startswith("kimi-")


SYSTEM_PROMPT = """You are a JSON-only prediction market API endpoint.
Input: a market question + current market price.
Output: a single JSON object, nothing else.

Required format (copy exactly, fill in values):
{"probability":0.00,"confidence":0.00,"reasoning":"one sentence"}

- probability: float 0.0–1.0, your true probability YES resolves
- confidence: float 0.0–1.0, your certainty (low info = below 0.4)
- reasoning: one short sentence, no newlines
- DO NOT write anything before or after the JSON object
- DO NOT use markdown code blocks"""


def _build_news_context(question: str) -> str:
    """Return a formatted news block to append to the prompt, or empty string."""
    if not USE_NEWS_CONTEXT:
        return ""
    try:
        from src.news import fetch_headlines
        headlines = fetch_headlines(question, max_results=NEWS_MAX_HEADLINES)
        if not headlines:
            return ""
        lines = "\n".join(f"  • {h}" for h in headlines)
        return f"\nRecent news headlines:\n{lines}\n"
    except Exception as e:
        logger.debug(f"News context skipped: {e}")
        return ""


def _parse_response(text: str) -> dict[str, Any] | None:
    """
    Extract probability + confidence from model response.
    Handles: clean JSON, JSON in prose, and plain prose with percentages.
    """
    # 1. Try direct JSON parse
    try:
        result = json.loads(text)
        return {"probability": float(result["probability"]),
                "confidence":  float(result["confidence"]),
                "reasoning":   result.get("reasoning", "")}
    except Exception:
        pass

    # 2. Try extracting JSON block from mixed text
    match = re.search(r"\{[^{}]*\"probability\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return {"probability": float(result["probability"]),
                    "confidence":  float(result["confidence"]),
                    "reasoning":   result.get("reasoning", "")}
        except Exception:
            pass

    # 3. Prose fallback: extract bare numbers for probability + confidence
    # Handles patterns like: "probability of 0.72" or "72%" or "confidence: 0.8"
    prob_match = re.search(
        r"(?:probability[:\s]+|\"probability\"[:\s]+)(0\.\d+|\d{1,2}%)", text, re.IGNORECASE
    )
    conf_match = re.search(
        r"(?:confidence[:\s]+|\"confidence\"[:\s]+)(0\.\d+|\d{1,2}%)", text, re.IGNORECASE
    )
    if prob_match and conf_match:
        def to_float(s: str) -> float:
            return float(s.strip("%")) / 100 if "%" in s else float(s)
        try:
            prob = to_float(prob_match.group(1))
            conf = to_float(conf_match.group(1))
            if 0.0 <= prob <= 1.0 and 0.0 <= conf <= 1.0:
                return {"probability": prob, "confidence": conf, "reasoning": "extracted from prose"}
        except Exception:
            pass

    return None


def _ask_model(model: str, question: str, yes_price: float, news_context: str = "") -> dict[str, Any] | None:
    """Call one model and parse its JSON response. Routes to Anthropic or Moonshot."""
    prompt = (
        f"Market: {question}\n"
        f"Market-implied YES probability: {yes_price:.2%}\n"
        f"{news_context}"
        f'Return JSON only: {{"probability":X.XX,"confidence":X.XX,"reasoning":"..."}}'
    )
    try:
        if _is_moonshot(model):
            # Kimi / Moonshot — OpenAI-compatible API
            model_id = model.replace("moonshot/", "")   # strip prefix if present
            client   = _get_openai_client()
            resp     = client.chat.completions.create(
                model=model_id,
                max_tokens=2048,  # Kimi uses reasoning tokens internally — needs headroom
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=1,    # Kimi K2.5 only accepts temperature=1
            )
            text = resp.choices[0].message.content.strip()
        else:
            # Claude — Anthropic API
            resp = _get_anthropic_client().messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

        result = _parse_response(text)
        if result is None:
            # One retry with a more explicit prompt
            logger.warning(f"No JSON from {model}, retrying...")
            retry_prompt = f"You must respond with ONLY this JSON object and nothing else:\n{{\"probability\":0.50,\"confidence\":0.30,\"reasoning\":\"brief\"}}\n\nFor this market: {question}"
            try:
                if _is_moonshot(model):
                    retry_resp = client.chat.completions.create(
                        model=model_id, max_tokens=200, temperature=1,
                        messages=[{"role":"user","content":retry_prompt}]
                    )
                    text = retry_resp.choices[0].message.content.strip()
                else:
                    retry_resp = _get_anthropic_client().messages.create(
                        model=model, max_tokens=200,
                        messages=[{"role":"user","content":retry_prompt}]
                    )
                    text = retry_resp.content[0].text.strip()
                result = _parse_response(text)
            except Exception:
                pass
        if result is None:
            logger.warning(f"No JSON from {model} after retry: {text[:150]}")
            return None

        prob = result["probability"]
        conf = result["confidence"]
        if not (0.0 <= prob <= 1.0 and 0.0 <= conf <= 1.0):
            logger.warning(f"Out-of-range: prob={prob} conf={conf}")
            return None

        return {"probability": prob, "confidence": conf,
                "reasoning": result.get("reasoning", ""), "model": model}

    except Exception as e:
        logger.error(f"Model {model} failed: {e}")
        return None


def forecast(market: dict[str, Any]) -> dict[str, Any] | None:
    """
    Forecast a market. Returns a signal dict or None if no trade should be made.

    Signal dict keys:
      probability, confidence, reasoning, bet_direction, model(s)
    """
    question    = market["question"]
    yes_price   = market["yes_price"]
    news_ctx    = _build_news_context(question)

    primary = _ask_model(PRIMARY_MODEL, question, yes_price, news_ctx)
    if primary is None:
        return None

    if CONSENSUS_MODE:
        secondary = _ask_model(SECONDARY_MODEL, question, yes_price, news_ctx)
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
