"""
Category Classifier — keyword-based market categorisation.
"""

CATEGORIES: dict[str, list[str]] = {
    "politics":   ["election", "president", "congress", "senate", "vote", "democrat",
                   "republican", "biden", "trump", "party", "governor", "mayor",
                   "legislation", "bill", "policy", "referendum", "impeach"],
    "sports":     ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
                   "baseball", "hockey", "tennis", "golf", "superbowl", "championship",
                   "playoff", "world cup", "league", "team", "match", "game",
                   "win", "beat", "title", "tournament"],
    "crypto":     ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
                   "defi", "nft", "token", "solana", "usdc", "binance", "coinbase",
                   "price", "ath", "halving"],
    "tech":       ["apple", "google", "microsoft", "amazon", "meta", "openai",
                   "gpt", "ai ", "artificial intelligence", "model", "release",
                   "iphone", "chip", "ipo", "acquisition"],
    "economics":  ["fed", "federal reserve", "interest rate", "inflation", "gdp",
                   "recession", "unemployment", "cpi", "fomc", "rate hike",
                   "rate cut", "treasury", "bond", "market crash", "stock"],
}


def classify(question: str) -> str:
    """Return the best-matching category for a market question."""
    q = question.lower()
    scores: dict[str, int] = {}
    for cat, keywords in CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "other"
