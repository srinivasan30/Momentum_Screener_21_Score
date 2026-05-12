# ⚡ Momentum Analyzer

A single-page Streamlit app that scores any stock on a **21-point momentum system** combining Mark Minervini's SEPA criteria, Prashant Shah's momentum pillars, and technical/fundamental quality checks.

Enter a ticker → get a score, action (Buy / Wait / Watch / Skip), trade levels, and full signal breakdown.

---

## Quick Start

```bash
# 1. Clone or copy files
mkdir momentum-analyzer && cd momentum-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run momentum_analyzer.py
```

Opens at `http://localhost:8501`

---

## Ticker Format

| Market | Format | Examples |
|--------|--------|---------|
| India (NSE) | `SYMBOL.NS` | `HFCL.NS`, `BBOX.NS`, `ATHERENERG.NS`, `TATACONSUM.NS`, `KIMS.NS` |
| India (BSE) | `SYMBOL.BO` | `HFCL.BO` |
| US | `SYMBOL` | `NVDA`, `PLTR`, `AAPL`, `TSLA` |

---

## What It Scores (21 Points)

### Tier 1 — Fundamentals (7 pts)
1. EPS Growth (>30% YoY)
2. Revenue Growth OR OPM (>15%)
3. Return on Equity (>18%)
4. Return on Capital Employed (>15%)
5. Cash Flow Quality — OCF/PAT (>0.7)
6. Debt/Equity (<0.3)
7. Institutional Holding (>30%)

### Tier 2 — Momentum Triggers (8 pts)
8. Near 52-Week High (<10% away)
9. Above 52-Week Low (>30%)
10. Relative Strength (3M + 6M weighted)
11. Volume Spike (>1.5x avg on up day)
12. EMA Stack (Price > 21 > 50 > 200)
13. Minervini SEPA (150MA > 200MA, 200MA rising)
14. Earnings Acceleration (Forward EPS > Trailing)
15. RSI Sweet Spot (55–70)

### Tier 3 — Multibagger DNA (6 pts)
16. Growth Market Cap ($300M–$10B)
17. Revenue Leader (>25% growth)
18. Gross Margin (>40%)
19. PAT Margin (>15%)
20. Consolidation Coil (<8% 10-day range)
21. Entry Zone (0–5% above EMA21)

---

## Momentum Cases Covered

| Case | What It Detects |
|------|----------------|
| Classic Breakout | Volume surge + daily price spike |
| Pullback to EMA | Healthy retrace to rising EMA21 |
| Earnings Gap-Up | Forward EPS accelerating vs trailing |
| Margin Expansion | OPM/PAT growing faster than revenue (BBOX case) |
| Sector Rotation | Relative strength outperformance 3M/6M |
| Volume Dry-Up Coil | Tight 10-day range = spring loading |
| Stage 2 Uptrend | All MAs stacked and trending up |
| Re-Rating Play | Low PE + accelerating earnings |
| Turnaround | Debt reducing, margins inflecting |
| Relative Strength | Multi-period return comparison |

---

## Score Guide

| Score | Rating | Action |
|-------|--------|--------|
| 16–21 | Elite | Buy Now / Buy 50% |
| 12–15 | Strong | Buy on Dip / Watchlist |
| 8–11 | Moderate | Watchlist |
| <8 | Weak | Skip |

---

## Deploy Free on Streamlit Cloud

1. Push code to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `momentum_analyzer.py`
4. Deploy — access from any device

---

## File Structure

```
momentum-analyzer/
├── momentum_analyzer.py    # Main app (single file)
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Theme config
└── README.md               # This file
```

---

**Disclaimer:** Educational tool only. Not financial advice. Always do your own research.
