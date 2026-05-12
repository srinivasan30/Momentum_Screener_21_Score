"""
Momentum Analyzer — Streamlit App
Enter any ticker → get a 21-point momentum score with full breakdown.

Run:  streamlit run momentum_analyzer.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global */
.stApp { background: #fafafa; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 900px; }

/* Title area */
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #111;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.2;
}
.app-sub {
    font-size: 0.82rem;
    color: #888;
    margin-top: 2px;
    margin-bottom: 1.5rem;
}

/* Score circle */
.score-wrap {
    display: flex;
    align-items: center;
    gap: 20px;
    margin: 1rem 0;
}
.score-circle {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    flex-shrink: 0;
}
.score-circle.elite   { background: #dcfce7; border: 3px solid #16a34a; }
.score-circle.strong  { background: #dbeafe; border: 3px solid #2563eb; }
.score-circle.moderate { background: #fef3c7; border: 3px solid #d97706; }
.score-circle.weak    { background: #f3f4f6; border: 3px solid #9ca3af; }
.score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 700; line-height: 1;
}
.score-circle.elite .score-num   { color: #166534; }
.score-circle.strong .score-num  { color: #1e40af; }
.score-circle.moderate .score-num { color: #92400e; }
.score-circle.weak .score-num    { color: #6b7280; }
.score-max {
    font-size: 0.65rem; color: #999;
    font-weight: 600; margin-top: 2px;
}

/* Action badge */
.action-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.action-badge.buy    { background: #dcfce7; color: #166534; }
.action-badge.wait   { background: #fef3c7; color: #92400e; }
.action-badge.watch  { background: #f3f4f6; color: #4b5563; }
.action-badge.skip   { background: #fee2e2; color: #991b1b; }

/* Meta text next to score */
.score-meta h2 {
    font-size: 1.15rem; font-weight: 700; color: #111;
    margin: 0 0 4px 0;
}
.score-meta p {
    font-size: 0.8rem; color: #666;
    margin: 0 0 8px 0; line-height: 1.4;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 8px;
    margin: 1rem 0;
}
.metric-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 14px;
}
.metric-card .m-label {
    font-size: 0.65rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.metric-card .m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #111;
    margin-top: 2px;
}
.metric-card .m-value.green { color: #16a34a; }
.metric-card .m-value.red   { color: #dc2626; }
.metric-card .m-value.amber { color: #d97706; }

/* Signal pills */
.signal-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0.5rem 0;
}
.signal-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}
.signal-pill.pass { background: #dcfce7; color: #166534; }
.signal-pill.partial { background: #fef3c7; color: #92400e; }
.signal-pill.fail { background: #fee2e2; color: #991b1b; }

/* Section headers */
.section-hdr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 2px solid #e5e7eb;
}

/* Trade levels */
.trade-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 0.5rem 0 1rem;
}
.trade-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.trade-card .t-label {
    font-size: 0.6rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.trade-card .t-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 2px;
}
.trade-card .t-value.red   { color: #dc2626; }
.trade-card .t-value.green { color: #16a34a; }

/* Disclaimer */
.disclaimer {
    margin-top: 2rem;
    padding: 12px 16px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    font-size: 0.7rem;
    color: #92400e;
}

@media (max-width: 640px) {
    .score-wrap { flex-direction: column; align-items: flex-start; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .trade-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────

def compute_technicals(df):
    d = df.copy()

    for span in [10, 21, 50]:
        d[f"EMA_{span}"] = d["Close"].ewm(span=span, adjust=False).mean()
    for span in [100, 150, 200]:
        if len(d) >= span:
            d[f"SMA_{span}"] = d["Close"].rolling(span).mean()

    d["Vol_20_Avg"] = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"] = d["Volume"] / d["Vol_20_Avg"]

    rsi = RSIIndicator(d["Close"], window=14)
    d["RSI"] = rsi.rsi()

    atr = AverageTrueRange(d["High"], d["Low"], d["Close"], window=14)
    d["ATR"] = atr.average_true_range()
    d["ATR_Pct"] = (d["ATR"] / d["Close"]) * 100

    d["Daily_Chg"] = d["Close"].pct_change() * 100

    lookback = min(252, len(d))
    d["H52"] = d["High"].rolling(lookback).max()
    d["L52"] = d["Low"].rolling(lookback).min()
    d["Pct_From_52H"] = ((d["H52"] - d["Close"]) / d["H52"]) * 100
    d["Pct_From_52L"] = ((d["Close"] - d["L52"]) / d["L52"]) * 100

    d["EMA21_Slope"] = d["EMA_21"].diff(5)
    if "SMA_200" in d.columns:
        d["SMA200_Slope"] = d["SMA_200"].diff(20)

    for p, label in [(21, "1M"), (63, "3M"), (126, "6M")]:
        d[f"Ret_{label}"] = d["Close"].pct_change(p) * 100 if len(d) >= p else 0

    d["Range_10D"] = ((d["High"].rolling(10).max() - d["Low"].rolling(10).min()) / d["Close"]) * 100

    return d


def analyze_stock(ticker_input):
    """Run full 21-point analysis on a single ticker."""

    # Normalize ticker
    ticker = ticker_input.strip().upper()
    if not ticker.endswith((".NS", ".BO")) and not any(c in ticker for c in [".", "-"]):
        # Could be either market — try as-is first (US), user can add .NS for India
        pass

    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info

    if hist.empty or len(hist) < 50:
        return None, "Not enough price data (need 50+ trading days)"

    if not info or not info.get("regularMarketPrice"):
        return None, "Could not fetch fundamental data"

    d = compute_technicals(hist)
    L = d.iloc[-1]

    price = L["Close"]
    market = "INDIA" if ".NS" in ticker or ".BO" in ticker else "US"
    currency = "₹" if market == "INDIA" else "$"

    def pct(key, default=0):
        v = info.get(key, default) or default
        return round(v * 100, 1) if abs(v) < 10 else round(v, 1)

    # ================================================================
    # SCORING — 21 POINTS (3 TIERS)
    # ================================================================
    score = 0
    signals = []  # (label, points_earned, max_points, status)

    # ── TIER 1: FUNDAMENTALS (7 pts) ──

    # 1. EPS Growth
    eg = pct("earningsGrowth")
    if eg > 30:
        score += 1; signals.append(("EPS Growth", f"+{eg}%", 1, "pass"))
    elif eg > 15:
        score += 0.5; signals.append(("EPS Growth", f"+{eg}%", 0.5, "partial"))
    else:
        signals.append(("EPS Growth", f"{eg}%", 0, "fail"))

    # 2. Revenue Growth OR OPM
    rg = pct("revenueGrowth")
    opm = pct("operatingMargins")
    if rg > 15 or opm > 15:
        score += 1; signals.append(("Rev/OPM", f"Rev {rg}% OPM {opm}%", 1, "pass"))
    elif rg > 5 or opm > 8:
        score += 0.5; signals.append(("Rev/OPM", f"Rev {rg}% OPM {opm}%", 0.5, "partial"))
    else:
        signals.append(("Rev/OPM", f"Rev {rg}% OPM {opm}%", 0, "fail"))

    # 3. RoE
    roe = pct("returnOnEquity")
    if roe > 18:
        score += 1; signals.append(("RoE", f"{roe}%", 1, "pass"))
    elif roe > 12:
        score += 0.5; signals.append(("RoE", f"{roe}%", 0.5, "partial"))
    else:
        signals.append(("RoE", f"{roe}%", 0, "fail"))

    # 4. RoCE proxy
    roa = pct("returnOnAssets")
    roce = round(roa * 2, 1)
    if roce > 15:
        score += 1; signals.append(("RoCE (est)", f"{roce}%", 1, "pass"))
    elif roce > 10:
        score += 0.5; signals.append(("RoCE (est)", f"{roce}%", 0.5, "partial"))
    else:
        signals.append(("RoCE (est)", f"{roce}%", 0, "fail"))

    # 5. Cash flow quality
    ocf = info.get("operatingCashflow", 0) or 0
    ni = info.get("netIncomeToCommon", 0) or 0
    ocf_pat = round(ocf / ni, 2) if ni > 0 else 0
    if ocf_pat > 0.7:
        score += 1; signals.append(("OCF/PAT", f"{ocf_pat}", 1, "pass"))
    elif ocf_pat > 0.4:
        score += 0.5; signals.append(("OCF/PAT", f"{ocf_pat}", 0.5, "partial"))
    else:
        signals.append(("OCF/PAT", f"{ocf_pat}", 0, "fail"))

    # 6. Debt
    de_raw = info.get("debtToEquity", 0) or 0
    de = round(de_raw / 100, 2)
    if de < 0.3:
        score += 1; signals.append(("Debt/Equity", f"{de}", 1, "pass"))
    elif de < 0.5:
        score += 0.5; signals.append(("Debt/Equity", f"{de}", 0.5, "partial"))
    else:
        signals.append(("Debt/Equity", f"{de}", 0, "fail"))

    # 7. Institutional
    inst = pct("heldPercentInstitutions")
    if inst > 30:
        score += 1; signals.append(("Institutional", f"{inst}%", 1, "pass"))
    elif inst > 15:
        score += 0.5; signals.append(("Institutional", f"{inst}%", 0.5, "partial"))
    else:
        signals.append(("Institutional", f"{inst}%", 0, "fail"))

    # ── TIER 2: MOMENTUM TRIGGERS (8 pts) ──

    # 8. Near 52W high
    pct_from_high = round(L.get("Pct_From_52H", 99), 1)
    if pct_from_high <= 10:
        score += 1; signals.append(("52W High Proximity", f"{pct_from_high}% away", 1, "pass"))
    elif pct_from_high <= 25:
        score += 0.5; signals.append(("52W High Proximity", f"{pct_from_high}% away", 0.5, "partial"))
    else:
        signals.append(("52W High Proximity", f"{pct_from_high}% away", 0, "fail"))

    # 9. Above 52W low
    pct_from_low = round(L.get("Pct_From_52L", 0), 1)
    if pct_from_low >= 30:
        score += 0.5; signals.append(("52W Low Distance", f"+{pct_from_low}%", 0.5, "pass"))
    else:
        signals.append(("52W Low Distance", f"+{pct_from_low}%", 0, "fail"))

    # 10. Relative strength
    r3m = round(L.get("Ret_3M", 0), 1)
    r6m = round(L.get("Ret_6M", 0), 1)
    rs = round(r3m * 0.4 + r6m * 0.6, 1)
    if rs > 40:
        score += 1; signals.append(("Relative Strength", f"3M:{r3m}% 6M:{r6m}%", 1, "pass"))
    elif rs > 20:
        score += 0.5; signals.append(("Relative Strength", f"3M:{r3m}% 6M:{r6m}%", 0.5, "partial"))
    else:
        signals.append(("Relative Strength", f"3M:{r3m}% 6M:{r6m}%", 0, "fail"))

    # 11. Volume confirmation
    vol_r = round(L.get("Vol_Ratio", 0), 2)
    daily_chg = round(L.get("Daily_Chg", 0), 2)
    if vol_r > 1.5 and daily_chg > 2:
        score += 1; signals.append(("Volume Spike", f"{vol_r}x on +{daily_chg}%", 1, "pass"))
    elif vol_r > 1.2 and daily_chg > 0:
        score += 0.5; signals.append(("Volume Spike", f"{vol_r}x on {daily_chg:+.1f}%", 0.5, "partial"))
    else:
        signals.append(("Volume Spike", f"{vol_r}x on {daily_chg:+.1f}%", 0, "fail"))

    # 12. EMA alignment
    has_200 = "SMA_200" in d.columns and not pd.isna(L.get("SMA_200"))
    has_50 = not pd.isna(L.get("EMA_50"))
    ema_aligned = False
    if has_200 and has_50:
        ema_aligned = price > L["EMA_21"] > L["EMA_50"] > L["SMA_200"]
    if ema_aligned:
        score += 1; signals.append(("EMA Stack", "Price>21>50>200", 1, "pass"))
    elif has_50 and price > L["EMA_50"]:
        score += 0.5; signals.append(("EMA Stack", "Price>50 only", 0.5, "partial"))
    else:
        signals.append(("EMA Stack", "Not aligned", 0, "fail"))

    # 13. SEPA
    sepa = False
    has_150 = "SMA_150" in d.columns and not pd.isna(L.get("SMA_150"))
    if has_150 and has_200:
        sepa = (
            L["SMA_150"] > L["SMA_200"]
            and L.get("SMA200_Slope", 0) > 0
            and price > L["SMA_150"]
        )
    if sepa:
        score += 1; signals.append(("Minervini SEPA", "150>200↑, Price above", 1, "pass"))
    elif has_200 and price > L["SMA_200"]:
        score += 0.5; signals.append(("Minervini SEPA", "Price>200 only", 0.5, "partial"))
    else:
        signals.append(("Minervini SEPA", "Not met", 0, "fail"))

    # 14. Earnings acceleration
    t_eps = info.get("trailingEps", 0) or 0
    f_eps = info.get("forwardEps", 0) or 0
    if t_eps > 0 and f_eps > t_eps:
        accel = round(((f_eps - t_eps) / t_eps) * 100, 0)
        if accel > 10:
            score += 1; signals.append(("Earnings Accel", f"Fwd +{accel}% vs Trail", 1, "pass"))
        else:
            score += 0.5; signals.append(("Earnings Accel", f"Fwd +{accel}% vs Trail", 0.5, "partial"))
    else:
        signals.append(("Earnings Accel", "No acceleration", 0, "fail"))

    # 15. RSI sweet spot
    rsi_val = round(L.get("RSI", 50), 1)
    if 55 <= rsi_val <= 70:
        score += 1; signals.append(("RSI Zone", f"{rsi_val}", 1, "pass"))
    elif 45 <= rsi_val <= 80:
        score += 0.5; signals.append(("RSI Zone", f"{rsi_val}", 0.5, "partial"))
    else:
        signals.append(("RSI Zone", f"{rsi_val}", 0, "fail"))

    # ── TIER 3: MULTIBAGGER DNA (6 pts) ──

    # 16. Growth mcap
    mcap_b = round((info.get("marketCap", 0) or 0) / 1e9, 1)
    if 0.3 <= mcap_b <= 10:
        score += 1; signals.append(("MCap Sweet Spot", f"${mcap_b}B", 1, "pass"))
    elif 10 < mcap_b <= 50:
        score += 0.5; signals.append(("MCap Sweet Spot", f"${mcap_b}B", 0.5, "partial"))
    else:
        signals.append(("MCap Sweet Spot", f"${mcap_b}B", 0, "fail"))

    # 17. Revenue leader
    if rg > 25:
        score += 1; signals.append(("Revenue Leader", f"+{rg}%", 1, "pass"))
    elif rg > 15:
        score += 0.5; signals.append(("Revenue Leader", f"+{rg}%", 0.5, "partial"))
    else:
        signals.append(("Revenue Leader", f"{rg}%", 0, "fail"))

    # 18. Gross margin
    gm = pct("grossMargins")
    if gm > 40:
        score += 1; signals.append(("Gross Margin", f"{gm}%", 1, "pass"))
    elif gm > 25:
        score += 0.5; signals.append(("Gross Margin", f"{gm}%", 0.5, "partial"))
    else:
        signals.append(("Gross Margin", f"{gm}%", 0, "fail"))

    # 19. PAT margin
    pm = pct("profitMargins")
    if pm > 15:
        score += 1; signals.append(("PAT Margin", f"{pm}%", 1, "pass"))
    elif pm > 8:
        score += 0.5; signals.append(("PAT Margin", f"{pm}%", 0.5, "partial"))
    else:
        signals.append(("PAT Margin", f"{pm}%", 0, "fail"))

    # 20. Consolidation coil
    r10 = round(L.get("Range_10D", 99), 1)
    if r10 < 8:
        score += 1; signals.append(("Consolidation", f"{r10}% range", 1, "pass"))
    elif r10 < 15:
        score += 0.5; signals.append(("Consolidation", f"{r10}% range", 0.5, "partial"))
    else:
        signals.append(("Consolidation", f"{r10}% range", 0, "fail"))

    # 21. Entry zone
    dist_ema21 = round(((price - L["EMA_21"]) / L["EMA_21"]) * 100, 1) if L["EMA_21"] > 0 else 99
    if 0 < dist_ema21 <= 5:
        score += 1; signals.append(("Entry Zone", f"{dist_ema21}% above EMA21", 1, "pass"))
    elif -3 < dist_ema21 <= 10:
        score += 0.5; signals.append(("Entry Zone", f"{dist_ema21}% from EMA21", 0.5, "partial"))
    else:
        signals.append(("Entry Zone", f"{dist_ema21}% from EMA21", 0, "fail"))

    # ================================================================
    # ACTION
    # ================================================================
    score = round(score, 1)
    atr_val = L.get("ATR", 0) or 0
    stop_loss = round(price - 2 * atr_val, 2)
    target1 = round(price * 1.15, 2)
    target2 = round(price * 1.30, 2)
    rr = round((target1 - price) / max(price - stop_loss, 0.01), 1)

    if score >= 16:
        if dist_ema21 <= 5:
            action, action_class = "BUY NOW", "buy"
        elif dist_ema21 <= 10:
            action, action_class = "BUY 50%", "buy"
        else:
            action, action_class = "WAIT — PULLBACK", "wait"
    elif score >= 12:
        if dist_ema21 <= 5 or r10 < 8:
            action, action_class = "BUY ON DIP", "buy"
        else:
            action, action_class = "WATCHLIST", "watch"
    elif score >= 8:
        action, action_class = "WATCHLIST", "watch"
    else:
        action, action_class = "SKIP", "skip"

    # Package results
    result = {
        "ticker": ticker,
        "name": (info.get("shortName") or info.get("longName") or ticker),
        "market": market,
        "currency": currency,
        "sector": info.get("sector", "—"),
        "industry": info.get("industry", "—"),
        "price": round(price, 2),
        "score": score,
        "action": action,
        "action_class": action_class,
        "signals": signals,
        # Fundamentals
        "eps_growth": eg,
        "rev_growth": rg,
        "opm": opm,
        "gm": gm,
        "pm": pm,
        "roe": roe,
        "roce": roce,
        "de": de,
        "ocf_pat": ocf_pat,
        "inst": inst,
        "mcap_b": mcap_b,
        # Technicals
        "daily_chg": daily_chg,
        "vol_ratio": vol_r,
        "rsi": rsi_val,
        "pct_from_52h": pct_from_high,
        "pct_from_52l": pct_from_low,
        "ret_3m": r3m,
        "ret_6m": r6m,
        "ema_aligned": ema_aligned,
        "sepa": sepa,
        "dist_ema21": dist_ema21,
        "range_10d": r10,
        # Trade
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "atr_pct": round(L.get("ATR_Pct", 0), 2),
    }

    return result, d


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown('<div class="app-title">⚡ Momentum Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Enter a ticker → get a 21-point momentum score instantly</div>', unsafe_allow_html=True)

# Input row
col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker_input = st.text_input(
        "Ticker",
        placeholder="e.g. HFCL.NS, BBOX.NS, NVDA, PLTR",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

st.markdown(
    '<p style="font-size:0.7rem;color:#aaa;margin-top:-10px;">'
    'India: add .NS (e.g. HFCL.NS, TATACONSUM.NS, ATHERENERG.NS) · US: just the ticker (e.g. NVDA, PLTR)'
    '</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if analyze_btn and ticker_input:
    with st.spinner(f"Analyzing {ticker_input.strip().upper()}..."):
        result, chart_data = analyze_stock(ticker_input)

    if result is None:
        st.error(f"Could not analyze: {chart_data}")
    else:
        r = result
        cur = r["currency"]

        # ── SCORE + ACTION ──
        sc_class = "elite" if r["score"] >= 16 else ("strong" if r["score"] >= 12 else ("moderate" if r["score"] >= 8 else "weak"))

        st.markdown(f"""
        <div class="score-wrap">
            <div class="score-circle {sc_class}">
                <div class="score-num">{r['score']}</div>
                <div class="score-max">/ 21</div>
            </div>
            <div class="score-meta">
                <h2>{r['name']}</h2>
                <p>{r['ticker']} · {r['sector']} · {r['industry']}</p>
                <span class="action-badge {r['action_class']}">{r['action']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KEY METRICS ──
        st.markdown('<div class="section-hdr">Key Metrics</div>', unsafe_allow_html=True)

        def color_cls(val, good_thresh, bad_thresh=None, invert=False):
            if invert:
                if val < good_thresh: return "green"
                if bad_thresh and val > bad_thresh: return "red"
                return "amber"
            if val > good_thresh: return "green"
            if bad_thresh and val < bad_thresh: return "red"
            return ""

        metrics_html = '<div class="metric-grid">'
        metrics = [
            ("Price", f"{cur}{r['price']}", ""),
            ("Day Chg", f"{r['daily_chg']:+.1f}%", "green" if r["daily_chg"] > 0 else "red"),
            ("52W High", f"{r['pct_from_52h']}% away", color_cls(r["pct_from_52h"], 10, 25, invert=True)),
            ("Volume", f"{r['vol_ratio']}x avg", color_cls(r["vol_ratio"], 1.5)),
            ("RSI", f"{r['rsi']}", "green" if 55 <= r["rsi"] <= 70 else ("red" if r["rsi"] > 80 or r["rsi"] < 30 else "")),
            ("3M Return", f"{r['ret_3m']:+.1f}%", "green" if r["ret_3m"] > 20 else ("red" if r["ret_3m"] < 0 else "")),
            ("6M Return", f"{r['ret_6m']:+.1f}%", "green" if r["ret_6m"] > 30 else ("red" if r["ret_6m"] < 0 else "")),
            ("MCap", f"${r['mcap_b']}B", ""),
        ]
        for label, val, cls in metrics:
            metrics_html += f'<div class="metric-card"><div class="m-label">{label}</div><div class="m-value {cls}">{val}</div></div>'
        metrics_html += '</div>'
        st.markdown(metrics_html, unsafe_allow_html=True)

        # ── FUNDAMENTALS ──
        st.markdown('<div class="section-hdr">Fundamentals</div>', unsafe_allow_html=True)

        fund_html = '<div class="metric-grid">'
        fund_metrics = [
            ("EPS Growth", f"{r['eps_growth']}%", color_cls(r["eps_growth"], 25, 0)),
            ("Rev Growth", f"{r['rev_growth']}%", color_cls(r["rev_growth"], 15, 0)),
            ("OPM", f"{r['opm']}%", color_cls(r["opm"], 15, 5)),
            ("Gross Margin", f"{r['gm']}%", color_cls(r["gm"], 40, 15)),
            ("PAT Margin", f"{r['pm']}%", color_cls(r["pm"], 15, 0)),
            ("RoE", f"{r['roe']}%", color_cls(r["roe"], 18, 10)),
            ("RoCE (est)", f"{r['roce']}%", color_cls(r["roce"], 15, 8)),
            ("D/E", f"{r['de']}", color_cls(r["de"], 0.3, 0.5, invert=True)),
            ("OCF/PAT", f"{r['ocf_pat']}", color_cls(r["ocf_pat"], 0.7, 0.3)),
            ("Institutional", f"{r['inst']}%", color_cls(r["inst"], 30, 10)),
        ]
        for label, val, cls in fund_metrics:
            fund_html += f'<div class="metric-card"><div class="m-label">{label}</div><div class="m-value {cls}">{val}</div></div>'
        fund_html += '</div>'
        st.markdown(fund_html, unsafe_allow_html=True)

        # ── MOMENTUM SIGNALS ──
        st.markdown('<div class="section-hdr">Momentum Checks (21 Criteria)</div>', unsafe_allow_html=True)

        signals_html = '<div class="signal-row">'
        for label, detail, pts, status in r["signals"]:
            signals_html += f'<span class="signal-pill {status}" title="{detail}">{label}: {detail} ({pts}pt)</span>'
        signals_html += '</div>'
        st.markdown(signals_html, unsafe_allow_html=True)

        # ── TRADE LEVELS ──
        st.markdown('<div class="section-hdr">Trade Levels</div>', unsafe_allow_html=True)

        trade_html = f"""
        <div class="trade-grid">
            <div class="trade-card">
                <div class="t-label">Stop Loss</div>
                <div class="t-value red">{cur}{r['stop_loss']}</div>
            </div>
            <div class="trade-card">
                <div class="t-label">Target 1 (15%)</div>
                <div class="t-value green">{cur}{r['target1']}</div>
            </div>
            <div class="trade-card">
                <div class="t-label">Target 2 (30%)</div>
                <div class="t-value green">{cur}{r['target2']}</div>
            </div>
            <div class="trade-card">
                <div class="t-label">Risk : Reward</div>
                <div class="t-value">{r['rr']}</div>
            </div>
        </div>
        """
        st.markdown(trade_html, unsafe_allow_html=True)

        # ── Additional signals summary ──
        ema_str = "✅ Aligned" if r["ema_aligned"] else "❌ Not aligned"
        sepa_str = "✅ Pass" if r["sepa"] else "❌ Fail"
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="m-label">EMA Stack</div><div class="m-value">{ema_str}</div></div>
            <div class="metric-card"><div class="m-label">SEPA</div><div class="m-value">{sepa_str}</div></div>
            <div class="metric-card"><div class="m-label">ATR %</div><div class="m-value">{r['atr_pct']}%</div></div>
            <div class="metric-card"><div class="m-label">10D Range</div><div class="m-value">{r['range_10d']}%</div></div>
            <div class="metric-card"><div class="m-label">EMA21 Dist</div><div class="m-value">{r['dist_ema21']}%</div></div>
            <div class="metric-card"><div class="m-label">52W Low Dist</div><div class="m-value">+{r['pct_from_52l']}%</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── PRICE CHART ──
        st.markdown('<div class="section-hdr">Price Chart (1Y)</div>', unsafe_allow_html=True)

        chart_df = chart_data[["Close", "EMA_21", "EMA_50"]].copy()
        if "SMA_200" in chart_data.columns:
            chart_df["SMA_200"] = chart_data["SMA_200"]
        chart_df.columns = [c.replace("_", " ") for c in chart_df.columns]
        st.line_chart(chart_df, height=320)

        # ── VOLUME CHART ──
        st.markdown('<div class="section-hdr">Volume</div>', unsafe_allow_html=True)
        vol_df = chart_data[["Volume"]].copy()
        st.bar_chart(vol_df, height=150)

        # ── SCORE GUIDE ──
        st.markdown("""
        <div class="section-hdr">Score Guide</div>
        <div class="metric-grid">
            <div class="metric-card"><div class="m-label">16 – 21</div><div class="m-value green">Elite</div></div>
            <div class="metric-card"><div class="m-label">12 – 15</div><div class="m-value" style="color:#2563eb;">Strong</div></div>
            <div class="metric-card"><div class="m-label">8 – 11</div><div class="m-value amber">Watch</div></div>
            <div class="metric-card"><div class="m-label">Below 8</div><div class="m-value red">Skip</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── DISCLAIMER ──
        st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This is a screening tool for educational purposes only. Not financial advice.
            Always do your own research before trading. Past momentum does not guarantee future returns.
        </div>
        """, unsafe_allow_html=True)


elif not ticker_input and not analyze_btn:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</p>
        <p style="font-size: 1rem; color: #666; max-width: 400px; margin: 0 auto; line-height: 1.6;">
            Type a ticker above and hit Analyze to get a full momentum breakdown with score, trade levels, and signals.
        </p>
        <div style="margin-top: 2rem; display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;">
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">HFCL.NS</span>
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">BBOX.NS</span>
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">ATHERENERG.NS</span>
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">NVDA</span>
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">PLTR</span>
            <span style="background:#f3f4f6; padding:4px 12px; border-radius:4px; font-size:0.8rem; color:#666; font-family:'JetBrains Mono',monospace;">TATACONSUM.NS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
