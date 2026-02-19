#!/usr/bin/env python3
"""Quick realtime signal test — fetch live markets, run Kimi, print signals."""
from dotenv import load_dotenv; load_dotenv()
import config, json, requests, sys
config.CONSENSUS_MODE   = False
config.USE_NEWS_CONTEXT = False
config.PRIMARY_MODEL    = 'kimi-k2.5'

from src.forecaster import forecast

SKIP   = ["up or down","5 min","15 min","°c on","°f on","set 1","set 2",
          "game 1","game 2","bo3","bo5","inning","o/u","spread:","candle","wick"," win on 2"]
CRYPTO = ["bitcoin","ethereum","btc","eth","solana","xrp","dogecoin","doge",
          "token","defi","nft","usdc","usdt","binance","coinbase"]

# Fetch 300 markets sorted by volume, skip top 50 (over-traded) — target mid-tier
all_markets = []
for offset in [0, 100, 200]:
    batch = requests.get('https://gamma-api.polymarket.com/markets', params={
        'closed':'false','active':'true','limit':100,'offset':offset,
        'order':'volume','ascending':'false'
    }, timeout=15).json()
    all_markets.extend(batch)

markets = []
for m in all_markets[50:]:   # skip top 50 most-traded (over-efficient)
    try:
        prices = m.get('outcomePrices', [])
        if isinstance(prices, str): prices = json.loads(prices)
        if len(prices) != 2: continue
        yp = float(prices[0])
        if not (0.12 < yp < 0.88): continue
        q = m.get('question','').strip()
        if not q or len(q) < 10: continue
        if any(k in q.lower() for k in SKIP + CRYPTO): continue
        vol = float(m.get('volumeNum', 0) or 0)
        if not (2000 < vol < 500000): continue  # mid-tier volume sweet spot
        markets.append({'id': m.get('id',''), 'question': q,
                        'yes_price': yp, 'no_price': 1-yp,
                        'volume': vol, 'end_date':'', 'raw':{}})
    except: continue

N = 20
print(f'\n🔍 Testing {N} live markets | Kimi K2.5 | no crypto | vol>$5k\n{"─"*60}')
signals = []

for i, m in enumerate(markets[:N]):
    sys.stdout.write(f'[{i+1:>2}/{N}] {m["question"][:58]:<58} ')
    sys.stdout.flush()
    sig = forecast(m)
    if sig:
        signals.append((m, sig))
        d    = sig['bet_direction']
        edge = sig.get('edge', sig['probability'] - m['yes_price'])
        icon = '🟢' if d == 'YES' else '🔴'
        print(f"{icon} {d:<3} conf={sig['confidence']:.0%} edge={edge:+.0%}")
    else:
        print('⏭  pass')

print(f'\n{"═"*60}')
print(f'  SIGNALS FIRED: {len(signals)} / {N}  ({len(signals)/N:.0%} rate)')
print(f'{"═"*60}')

for m, s in signals:
    d    = s['bet_direction']
    edge = s.get('edge', s['probability'] - m['yes_price'])
    icon = '🟢 BET YES' if d == 'YES' else '🔴 BET NO'
    print(f'\n  {icon}')
    print(f'  Market:  {m["yes_price"]:.0%} YES  |  ${m["volume"]:>10,.0f} vol')
    print(f'  Model:   {s["probability"]:.0%}  |  Edge: {edge:+.0%}  |  Conf: {s["confidence"]:.0%}')
    print(f'  Q:       {m["question"]}')
    print(f'  Reason:  {s["reasoning"][:130]}')
