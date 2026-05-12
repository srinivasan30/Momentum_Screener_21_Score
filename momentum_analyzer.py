"""
Momentum Analyzer v3 — Hybrid Data Source
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
India:  yfinance (price/technicals) + Screener.in (fundamentals)
US:     yfinance (everything)

Features:
- 21-point momentum score
- Upside multiple estimate (bear / conservative / base / bull)
- PAT growth timeline with bar chart
- Trade levels + risk management
- Full 21-signal breakdown by tier

Run: streamlit run momentum_analyzer.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp { background: #f8f9fa; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 960px; }

.app-title { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#111; letter-spacing:-0.5px; }
.app-sub   { font-size:0.76rem; color:#999; margin-top:3px; margin-bottom:1.2rem; }

/* Hero */
.hero { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:20px 22px; display:flex; gap:20px; align-items:flex-start; margin-bottom:12px; }
.score-ring { width:86px; height:86px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0; font-family:'JetBrains Mono',monospace; }
.score-ring.elite    { background:#dcfce7; border:3px solid #16a34a; }
.score-ring.strong   { background:#dbeafe; border:3px solid #2563eb; }
.score-ring.moderate { background:#fef3c7; border:3px solid #d97706; }
.score-ring.weak     { background:#fef2f2; border:3px solid #dc2626; }
.score-ring .snum    { font-size:1.7rem; font-weight:700; line-height:1; }
.score-ring.elite    .snum { color:#166534; }
.score-ring.strong   .snum { color:#1e40af; }
.score-ring.moderate .snum { color:#92400e; }
.score-ring.weak     .snum { color:#dc2626; }
.score-ring .sden    { font-size:0.58rem; color:#999; font-weight:600; margin-top:1px; }
.hero-body  { flex:1; min-width:0; }
.hero-name  { font-size:1.05rem; font-weight:700; color:#111; margin:0; line-height:1.3; }
.hero-meta  { font-size:0.73rem; color:#999; margin:2px 0 8px; }

/* Badges */
.badge { display:inline-block; padding:4px 12px; border-radius:5px; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; margin-right:5px; margin-bottom:3px; }
.badge.buy    { background:#dcfce7; color:#166534; }
.badge.wait   { background:#fef3c7; color:#92400e; }
.badge.watch  { background:#f1f5f9; color:#475569; }
.badge.skip   { background:#fef2f2; color:#991b1b; }
.badge.india  { background:#eff6ff; color:#1d4ed8; }
.badge.us     { background:#f0fdf4; color:#15803d; }

/* Multiples box */
.mbox { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; margin-bottom:12px; }
.mbox-title { font-family:'JetBrains Mono',monospace; font-size:0.67rem; font-weight:700; color:#999; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 12px; }
.mgrid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.mc { border-radius:8px; padding:12px 10px; text-align:center; border:1px solid transparent; }
.mc.bear { background:#fef2f2; border-color:#fecaca; }
.mc.cons { background:#f0fdf4; border-color:#bbf7d0; }
.mc.base { background:#eff6ff; border-color:#bfdbfe; }
.mc.bull { background:#fefce8; border-color:#fde68a; }
.mc .ml  { font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; color:#888; }
.mc .mx  { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; line-height:1.2; margin:3px 0 1px; }
.mc.bear .mx { color:#dc2626; }
.mc.cons .mx { color:#166534; }
.mc.base .mx { color:#1e40af; }
.mc.bull .mx { color:#92400e; }
.mc .mtp { font-size:0.7rem; color:#555; font-weight:600; }
.mc .mlo { font-size:0.62rem; color:#999; margin-top:3px; line-height:1.3; }
.mbox-footer { font-size:0.67rem; color:#9ca3af; margin-top:10px; line-height:1.6; }

/* Section header */
.sec { font-family:'JetBrains Mono',monospace; font-size:0.66rem; font-weight:700; color:#9ca3af; text-transform:uppercase; letter-spacing:1.5px; margin:16px 0 8px; padding-bottom:5px; border-bottom:1.5px solid #e5e7eb; }

/* Metric grid */
.mg  { display:grid; grid-template-columns:repeat(auto-fill,minmax(118px,1fr)); gap:7px; margin-bottom:8px; }
.mce { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:9px 12px; }
.mce .ml { font-size:0.58rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.4px; font-weight:600; }
.mce .mv { font-family:'JetBrains Mono',monospace; font-size:0.88rem; font-weight:700; color:#111; margin-top:2px; }
.mce .mv.g { color:#16a34a; } .mce .mv.r { color:#dc2626; }
.mce .mv.a { color:#d97706; } .mce .mv.b { color:#2563eb; }

/* Source tag */
.stag { display:inline-block; font-size:0.56rem; padding:1px 5px; border-radius:3px; font-weight:600; letter-spacing:0.3px; vertical-align:middle; margin-left:3px; }
.stag.sc { background:#eff6ff; color:#1d4ed8; }
.stag.yf { background:#f0fdf4; color:#15803d; }
.stag.ca { background:#fefce8; color:#854d0e; }

/* Trade grid */
.tg { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin-bottom:6px; }
.tc { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:9px 12px; text-align:center; }
.tc .tl { font-size:0.58rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.4px; font-weight:600; }
.tc .tv { font-family:'JetBrains Mono',monospace; font-size:0.92rem; font-weight:700; margin-top:2px; }
.tc .tv.r { color:#dc2626; } .tc .tv.g { color:#16a34a; }

/* Signal pills */
.sp { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:6px; }
.sp .pi { display:inline-block; padding:3px 8px; border-radius:4px; font-size:0.67rem; font-weight:600; cursor:default; }
.sp .pi.pass    { background:#dcfce7; color:#166534; }
.sp .pi.partial { background:#fef3c7; color:#92400e; }
.sp .pi.fail    { background:#fef2f2; color:#991b1b; }
.tier-lbl { width:100%; font-size:0.6rem; font-weight:700; color:#9ca3af; text-transform:uppercase; letter-spacing:1px; margin:6px 0 3px; }

/* PAT timeline */
.timeline { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:14px 16px; overflow-x:auto; }
.tl-row   { display:flex; gap:0; align-items:stretch; min-width:480px; }
.tl-yr-col{ flex:1; text-align:center; padding:6px 4px; border-right:1px solid #f0f0f0; }
.tl-yr-col:last-child { border-right:none; }
.tl-y   { font-size:0.62rem; color:#9ca3af; font-weight:600; }
.tl-s   { font-size:0.68rem; color:#374151; font-weight:600; margin:2px 0; }
.tl-o   { font-size:0.65rem; color:#2563eb; font-weight:600; }
.tl-bar-w { height:6px; background:#f3f4f6; border-radius:3px; margin:4px 6px; overflow:hidden; }
.tl-bar   { height:100%; border-radius:3px; background:#16a34a; }
.tl-p   { font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:700; color:#111; margin:2px 0; }
.tl-p.hi { color:#16a34a; }

/* Data notice */
.dn { background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:8px 12px; font-size:0.7rem; color:#92400e; margin:6px 0 10px; }

/* Disclaimer */
.disc { background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:10px 14px; font-size:0.67rem; color:#9ca3af; margin-top:1.5rem; }

@media(max-width:640px){
  .hero { flex-direction:column; }
  .mg { grid-template-columns:repeat(2,1fr); }
  .tg { grid-template-columns:repeat(2,1fr); }
  .mgrid4 { grid-template-columns:repeat(2,1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SCREENER.IN SCRAPER
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener(symbol: str) -> dict:
    symbol = symbol.replace(".NS","").replace(".BO","").upper()
    result = {}

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })

    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{symbol}{suffix}"
        try:
            r = session.get(url, timeout=12)
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # ── Ratios section ──
            ratios_sec = soup.find("section", id="ratios")
            if ratios_sec:
                for li in ratios_sec.find_all("li"):
                    n_tag = li.find("span", class_="name")
                    v_tag = li.find("span", class_="nowrap") or li.find("span", class_="number")
                    if n_tag and v_tag:
                        k = n_tag.get_text(strip=True)
                        v = v_tag.get_text(strip=True).replace(",","").replace("₹","").replace("%","").strip()
                        try:    result[k] = float(v)
                        except: result[k] = v

            # ── Table parser ──
            def parse_table(sec_id):
                sec = soup.find("section", id=sec_id)
                if not sec: return {}
                tbl = sec.find("table")
                if not tbl: return {}
                hdrs = [th.get_text(strip=True) for th in tbl.find_all("th")]
                rows = {}
                for row in tbl.find_all("tr")[1:]:
                    cells = row.find_all(["td","th"])
                    if not cells: continue
                    rname = cells[0].get_text(strip=True)
                    vals  = []
                    for c in cells[1:]:
                        t = c.get_text(strip=True).replace(",","")
                        try:    vals.append(float(t))
                        except: vals.append(None)
                    if rname and vals:
                        rows[rname] = vals
                return {"headers": hdrs[1:], "data": rows}

            pl = parse_table("profit-loss")
            bs = parse_table("balance-sheet")
            cf = parse_table("cash-flow")
            sh = parse_table("shareholding")

            if pl: result["_pl"] = pl
            if bs: result["_bs"] = bs
            if cf: result["_cf"] = cf
            if sh: result["_sh"] = sh

            # ── Derived P&L metrics ──
            if pl and "data" in pl:
                d = pl["data"]

                def last_n(key, n=2):
                    for k in d:
                        if key.lower() in k.lower():
                            vals = [v for v in d[k] if v is not None]
                            return vals[-n:] if len(vals)>=n else vals
                    return []

                def yoy(key):
                    v = last_n(key, 2)
                    if len(v)==2 and v[0] and v[0]!=0:
                        return round(((v[1]-v[0])/abs(v[0]))*100, 1)
                    return None

                def latest(key):
                    v = last_n(key, 1)
                    return v[-1] if v else None

                result["rev_growth"]   = yoy("Sales")
                result["pat_growth"]   = yoy("Net Profit") or yoy("Profit after tax")
                result["opm_latest"]   = latest("OPM %") or latest("OPM")
                result["pat_latest"]   = latest("Net Profit") or latest("Profit after tax")
                result["sales_latest"] = latest("Sales")
                result["eps_latest"]   = latest("EPS")

                for k in d:
                    if k.strip() == "Sales":
                        result["_sales_series"] = [v for v in d[k] if v is not None]
                    if ("Net Profit" in k or "Profit after tax" in k) and "_pat_series" not in result:
                        result["_pat_series"] = [v for v in d[k] if v is not None]
                    if "OPM" in k and "_opm_series" not in result:
                        result["_opm_series"] = [v for v in d[k] if v is not None]

            # ── Derived balance sheet ──
            if bs and "data" in bs:
                d = bs["data"]
                def last_bs(key):
                    for k in d:
                        if key.lower() in k.lower():
                            vals = [v for v in d[k] if v is not None]
                            return vals[-1] if vals else None
                    return None
                debt   = last_bs("Total Debt") or last_bs("Borrowings")
                equity = last_bs("Equity") or last_bs("Shareholders")
                if debt is not None and equity and equity != 0:
                    result["debt_equity"] = round(debt/equity, 2)

            # ── Derived cashflow ──
            if cf and "data" in cf:
                d = cf["data"]
                def last_cf(key):
                    for k in d:
                        if key.lower() in k.lower():
                            vals = [v for v in d[k] if v is not None]
                            return vals[-1] if vals else None
                    return None
                ocf = last_cf("Operating") or last_cf("Cash from Operations")
                pat = result.get("pat_latest")
                if ocf is not None and pat and pat != 0:
                    result["ocf_pat"] = round(ocf/pat, 2)

            # ── Shareholding ──
            if sh and "data" in sh:
                d = sh["data"]
                for k in d:
                    vals = [v for v in d[k] if v is not None]
                    if "Promoter" in k:
                        result["promoter_pct"] = vals[-1] if vals else None
                    if ("FII" in k or "Foreign" in k) and "fii_pct" not in result:
                        result["fii_pct"] = vals[-1] if vals else None
                        if len(vals)>=2 and vals[-2]:
                            result["fii_change"] = round(vals[-1]-vals[-2], 2)
                    if ("DII" in k or "Domestic" in k) and "dii_pct" not in result:
                        result["dii_pct"] = vals[-1] if vals else None

            if "_pl" in result:
                result["_years"]  = pl.get("headers", [])
                result["_source"] = "screener.in"
                return result

        except Exception as e:
            result["_error"] = str(e)
            continue

    return result


# ─────────────────────────────────────────────────────────────
# TECHNICALS
# ─────────────────────────────────────────────────────────────
def compute_technicals(df):
    d = df.copy()
    for span in [10, 21, 50]:
        d[f"EMA_{span}"] = d["Close"].ewm(span=span, adjust=False).mean()
    for span in [100, 150, 200]:
        if len(d) >= span:
            d[f"SMA_{span}"] = d["Close"].rolling(span).mean()
    d["Vol_20_Avg"] = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"]  = d["Volume"] / d["Vol_20_Avg"]
    d["RSI"]        = RSIIndicator(d["Close"], window=14).rsi()
    atr             = AverageTrueRange(d["High"], d["Low"], d["Close"], window=14)
    d["ATR"]        = atr.average_true_range()
    d["ATR_Pct"]    = (d["ATR"] / d["Close"]) * 100
    d["Daily_Chg"]  = d["Close"].pct_change() * 100
    lb = min(252, len(d))
    d["H52"]           = d["High"].rolling(lb).max()
    d["L52"]           = d["Low"].rolling(lb).min()
    d["Pct_From_52H"]  = ((d["H52"] - d["Close"]) / d["H52"]) * 100
    d["Pct_From_52L"]  = ((d["Close"] - d["L52"]) / d["L52"]) * 100
    d["EMA21_Slope"]   = d["EMA_21"].diff(5)
    if "SMA_200" in d.columns:
        d["SMA200_Slope"] = d["SMA_200"].diff(20)
    for p, lbl in [(21,"1M"),(63,"3M"),(126,"6M"),(252,"12M")]:
        d[f"Ret_{lbl}"] = d["Close"].pct_change(p)*100 if len(d)>=p else 0
    d["Range_10D"] = ((d["High"].rolling(10).max() - d["Low"].rolling(10).min()) / d["Close"]) * 100
    return d


# ─────────────────────────────────────────────────────────────
# MULTIPLES ENGINE
# ─────────────────────────────────────────────────────────────
def estimate_multiples(price, info, sc, is_india):
    res = {"bear":None,"conservative":None,"base":None,"bull":None,
           "logic":"","pat_cagr":None,"pe_current":None,"pe_fair":None,
           "pe_bull":None,"margin_bonus":0}
    try:
        # PE
        pe = None
        if is_india: pe = sc.get("Stock P/E") or sc.get("P/E")
        if not pe:   pe = info.get("trailingPE") or info.get("forwardPE")
        pe = round(float(pe), 1) if pe else None

        # PAT growth
        pg = None
        if is_india: pg = sc.get("pat_growth")
        if pg is None:
            eg = info.get("earningsGrowth", 0) or 0
            pg = eg*100 if abs(eg)<10 else eg

        # OPM
        opm = None
        if is_india: opm = sc.get("opm_latest") or sc.get("OPM")
        if not opm:
            opm = (info.get("operatingMargins",0) or 0)*100
            if abs(opm)>100: opm = None

        # CAGR estimate (moderate from last growth)
        if pg and pg > 0:
            cagr = 35 if pg>100 else (30 if pg>50 else (25 if pg>30 else (18 if pg>15 else max(pg*0.8,5))))
        else:
            cagr = 12

        # Margin expansion bonus
        mb = 5 if opm and opm<8 else (3 if opm and opm<15 else (1 if opm and opm<25 else 0))

        # PE re-rating
        if pe:
            if   pe < 15: pf,pb,note = pe*2.0, pe*3.0, "Deeply undervalued → PE expansion potential"
            elif pe < 25: pf,pb,note = pe*1.5, pe*2.2, "Moderate PE → re-rating possible on growth"
            elif pe < 40: pf,pb,note = pe*1.2, pe*1.6, "Fair PE → growth must justify valuation"
            elif pe < 60: pf,pb,note = pe*1.0, pe*1.2, "Premium PE → pure growth play, no re-rating"
            else:         pf,pb,note = pe*0.85,pe*1.0, "High PE → de-rating risk if growth slows"
        else:
            pf,pb,pe,note = 25, 35, 20, "PE unavailable → using sector average estimate"

        def mult(c, pe_end, pe_start):
            eps_m = (1+c/100)**3
            pe_m  = (pe_end/pe_start) if pe_start and pe_start>0 else 1
            return round(min(eps_m*pe_m, 25), 1)

        bear_c = max(cagr*0.3, -10)
        cons_c = cagr*0.6
        base_c = cagr + mb*0.5
        bull_c = min(cagr*1.3 + mb, 60)

        pe_bear = (pe*0.7) if pe else pf*0.7

        res.update({
            "bear":         mult(bear_c, pe_bear, pe or pf),
            "conservative": mult(cons_c, pf*0.9,  pe or pf),
            "base":         mult(base_c, pf,       pe or pf),
            "bull":         min(mult(bull_c, pb,   pe or pf), 20),
            "logic":        note,
            "pat_cagr":     cagr,
            "margin_bonus": mb,
            "pe_current":   pe,
            "pe_fair":      round(pf, 0),
            "pe_bull":      round(pb, 0),
        })
    except Exception as e:
        res["error"] = str(e)
    return res


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze(ticker_raw: str):
    ticker   = ticker_raw.strip().upper()
    is_india = ticker.endswith(".NS") or ticker.endswith(".BO")

    # Price data
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1y")
    info  = stock.info or {}

    if hist.empty or len(hist) < 50:
        return None, "Not enough price data (need 50+ trading days)."
    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
        return None, "Could not fetch quote. Check the ticker (India: add .NS, e.g. HFCL.NS)."

    # Screener.in for India
    sc = {}
    if is_india:
        with st.spinner("Fetching Screener.in fundamentals …"):
            sc = fetch_screener(ticker)

    # Technicals
    d  = compute_technicals(hist)
    L  = d.iloc[-1]
    price    = float(L["Close"])
    currency = "₹" if is_india else "$"

    # ── Fundamental getter: Screener.in preferred, yfinance fallback ──
    def gf(sc_key, yf_key, yf_mult=1.0, default=0.0):
        if is_india and sc:
            v = sc.get(sc_key)
            if v is not None:
                try:
                    return round(float(v), 2), "sc"
                except: pass
        v = info.get(yf_key, default) or default
        try:
            v = float(v)
            if yf_mult == 100 and abs(v) < 10:
                v = v * 100
            return round(v, 2), "yf"
        except:
            return default, "missing"

    eps_g,  src_eg   = gf("pat_growth",   "earningsGrowth",        100)
    rev_g,  src_rg   = gf("rev_growth",   "revenueGrowth",         100)
    opm,    src_opm  = gf("opm_latest",   "operatingMargins",      100)
    gm,     _        = gf("OPM",          "grossMargins",          100)
    pm,     src_pm   = gf("Net Profit %", "profitMargins",         100)
    roe,    src_roe  = gf("ROE",          "returnOnEquity",        100)
    roce,   src_roce = gf("ROCE",         "returnOnAssets",        200)
    de,     src_de   = gf("debt_equity",  "debtToEquity",          0.01)
    ocf_p,  src_cf   = gf("ocf_pat",      "operatingCashflow",     1.0)
    pe_c,   _        = gf("Stock P/E",    "trailingPE",            1.0)

    # OCF/PAT from yfinance for US
    if not is_india or src_cf == "yf":
        ocf = info.get("operatingCashflow", 0) or 0
        ni  = info.get("netIncomeToCommon",  0) or 0
        if ni and ni != 0:
            ocf_p   = round(ocf/ni, 2)
            src_cf  = "yf"

    # Institutional
    if is_india and sc:
        fii  = sc.get("fii_pct", 0) or 0
        dii  = sc.get("dii_pct", 0) or 0
        inst = round(fii+dii, 1)
        src_inst = "sc"
        fii_chg  = sc.get("fii_change")
        promoter = sc.get("promoter_pct")
    else:
        ir = info.get("heldPercentInstitutions", 0) or 0
        inst = round(float(ir)*100, 1) if abs(float(ir))<=1 else round(float(ir), 1)
        src_inst, fii_chg, promoter = "yf", None, None

    # Technicals
    pct_hi   = round(float(L.get("Pct_From_52H", 99)), 1)
    pct_lo   = round(float(L.get("Pct_From_52L",  0)), 1)
    r3m      = round(float(L.get("Ret_3M",  0)), 1)
    r6m      = round(float(L.get("Ret_6M",  0)), 1)
    r12m     = round(float(L.get("Ret_12M", 0)), 1)
    vol_r    = round(float(L.get("Vol_Ratio", 0)), 2)
    dchg     = round(float(L.get("Daily_Chg", 0)), 2)
    rsi_v    = round(float(L.get("RSI", 50)), 1)
    r10d     = round(float(L.get("Range_10D", 99)), 1)
    atr_v    = float(L.get("ATR", 0)) or 0
    atr_pct  = round(float(L.get("ATR_Pct", 0)), 2)

    ema21  = float(L.get("EMA_21",  price))
    ema50  = float(L.get("EMA_50",  price))
    sma200 = float(L.get("SMA_200", 0)) if "SMA_200" in d.columns else 0
    sma150 = float(L.get("SMA_150", 0)) if "SMA_150" in d.columns else 0
    sma200s= float(L.get("SMA200_Slope", 0)) if "SMA200_Slope" in d.columns else 0

    ema_ok = bool(price > ema21 > ema50 > sma200) if sma200>0 else bool(price > ema21 > ema50)
    sepa   = bool(sma150>sma200>0 and sma200s>0 and price>sma150)
    d21    = round(((price-ema21)/ema21)*100, 1) if ema21>0 else 0
    mcap_b = round((info.get("marketCap",0) or 0)/1e9, 1)
    t_eps  = info.get("trailingEps", 0) or 0
    f_eps  = info.get("forwardEps",  0) or 0

    # ═══════════════════════════════════════
    # SCORING — 21 POINTS
    # ═══════════════════════════════════════
    score = 0.0
    sigs  = []   # (label, detail, pts, status)

    def add(lbl, detail, pts):
        nonlocal score
        score += pts
        st = "pass" if pts==1 else ("partial" if pts==0.5 else "fail")
        sigs.append((lbl, detail, pts, st))

    # TIER 1 — FUNDAMENTALS
    add("EPS/PAT Growth", f"{eps_g:+.0f}%",
        1.0 if eps_g>30 else (0.5 if eps_g>15 else 0.0))
    add("Rev/OPM", f"Rev {rev_g:.0f}% | OPM {opm:.0f}%",
        1.0 if rev_g>15 or opm>15 else (0.5 if rev_g>5 or opm>8 else 0.0))
    add("RoE", f"{roe:.1f}%",
        1.0 if roe>18 else (0.5 if roe>12 else 0.0))
    add("RoCE", f"{roce:.1f}%",
        1.0 if roce>15 else (0.5 if roce>10 else 0.0))
    add("OCF/PAT", f"{ocf_p:.2f}",
        1.0 if ocf_p>0.7 else (0.5 if ocf_p>0.4 else 0.0))
    add("Debt/Equity", f"{de:.2f}",
        1.0 if de<0.3 else (0.5 if de<0.5 else 0.0))
    add("Institutional", f"{inst:.1f}%",
        1.0 if inst>30 else (0.5 if inst>15 else 0.0))

    # TIER 2 — MOMENTUM
    add("52W High", f"{pct_hi:.1f}% away",
        1.0 if pct_hi<=10 else (0.5 if pct_hi<=25 else 0.0))
    add("52W Low Gap", f"+{pct_lo:.0f}%",
        0.5 if pct_lo>=30 else 0.0)
    rs = r3m*0.4 + r6m*0.6
    add("Rel. Strength", f"3M:{r3m:.0f}% 6M:{r6m:.0f}%",
        1.0 if rs>40 else (0.5 if rs>20 else 0.0))
    add("Volume Spike", f"{vol_r:.1f}x on {dchg:+.1f}%",
        1.0 if vol_r>1.5 and dchg>2 else (0.5 if vol_r>1.2 and dchg>0 else 0.0))
    add("EMA Stack", "P>21>50>200 ✓" if ema_ok else ("P>21>50" if price>ema21>ema50 else "Misaligned"),
        1.0 if ema_ok else (0.5 if price>ema21>ema50 else 0.0))
    add("SEPA", "Pass ✓" if sepa else ("P>200" if sma200>0 and price>sma200 else "Fail"),
        1.0 if sepa else (0.5 if sma200>0 and price>sma200 else 0.0))
    if t_eps>0 and f_eps>t_eps:
        accel = round(((f_eps-t_eps)/abs(t_eps))*100, 0)
        add("Earnings Accel", f"Fwd +{accel:.0f}% vs Trail",
            1.0 if accel>10 else 0.5)
    elif eps_g>30:
        add("Earnings Accel", f"Strong EPS trend {eps_g:.0f}%", 0.5)
    else:
        add("Earnings Accel", "No visible acceleration", 0.0)
    add("RSI Zone", f"{rsi_v:.0f}",
        1.0 if 55<=rsi_v<=70 else (0.5 if 45<=rsi_v<=80 else 0.0))

    # TIER 3 — MULTIBAGGER DNA
    add("MCap Zone", f"${mcap_b}B",
        1.0 if 0.3<=mcap_b<=10 else (0.5 if 10<mcap_b<=50 else 0.0))
    add("Rev Leader", f"+{rev_g:.0f}%",
        1.0 if rev_g>25 else (0.5 if rev_g>15 else 0.0))
    mv = opm if opm else gm
    add("Margin Quality", f"{mv:.0f}%",
        1.0 if mv>40 else (0.5 if mv>20 else 0.0))
    add("PAT Margin", f"{pm:.1f}%",
        1.0 if pm>15 else (0.5 if pm>8 else 0.0))
    add("Coil/Consolidation", f"{r10d:.1f}% range",
        1.0 if r10d<8 else (0.5 if r10d<15 else 0.0))
    add("Entry Zone", f"{d21:+.1f}% from EMA21",
        1.0 if 0<d21<=5 else (0.5 if -3<d21<=10 else 0.0))

    score = round(score, 1)

    # Action
    if score>=16:
        act,acls = ("BUY NOW","buy") if d21<=5 else (("BUY 50%","buy") if d21<=10 else ("WAIT — PULLBACK","wait"))
    elif score>=12:
        act,acls = ("BUY ON DIP","buy") if (d21<=5 or r10d<8) else ("WATCHLIST","watch")
    elif score>=8:
        act,acls = "WATCHLIST","watch"
    else:
        act,acls = "SKIP","skip"

    # Trade levels
    sl  = round(price - 2*atr_v, 2)
    t1  = round(price*1.15, 2)
    t2  = round(price*1.30, 2)
    rr  = round((t1-price)/max(price-sl, 0.01), 1)

    mults = estimate_multiples(price, info, sc, is_india)

    return {
        "ticker":ticker, "name":info.get("shortName") or info.get("longName") or ticker,
        "market":"INDIA" if is_india else "US", "currency":currency, "is_india":is_india,
        "sector":info.get("sector","—"), "industry":info.get("industry","—"),
        "score":score, "sigs":sigs, "action":act, "acls":acls,
        "price":round(price,2), "dchg":dchg, "vol_ratio":vol_r, "rsi":rsi_v,
        "pct_hi":pct_hi, "pct_lo":pct_lo, "r3m":r3m, "r6m":r6m, "r12m":r12m,
        "ema_ok":ema_ok, "sepa":sepa, "d21":d21, "r10d":r10d,
        "atr_pct":atr_pct, "ema21":round(ema21,2),
        "eps_g":eps_g, "rev_g":rev_g, "opm":opm, "gm":gm, "pm":pm,
        "roe":roe, "roce":roce, "de":de, "ocf_p":ocf_p, "inst":inst,
        "mcap_b":mcap_b, "pe_c":pe_c, "fii_chg":fii_chg, "promoter":promoter,
        "src_eg":src_eg, "src_roe":src_roe, "src_de":src_de,
        "src_cf":src_cf, "src_inst":src_inst,
        "sc":sc, "sl":sl, "t1":t1, "t2":t2, "rr":rr,
        "mults":mults, "_chart":d,
    }, None


# ─────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────
def stag(src):
    if src=="sc":  return '<span class="stag sc">screener.in</span>'
    if src=="yf":  return '<span class="stag yf">yfinance</span>'
    return '<span class="stag ca">estimated</span>'

def mc(lbl, val, cls=""):
    return f'<div class="mce"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>'

def cc(val, g, r=None, inv=False):
    if val is None: return ""
    if inv: return "g" if val<g else ("r" if r and val>r else "a")
    return "g" if val>g else ("r" if r is not None and val<r else "")


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-title">⚡ Momentum Analyzer</div>
<div class="app-sub">
  India: Screener.in fundamentals + yfinance technicals &nbsp;·&nbsp;
  US: yfinance &nbsp;·&nbsp; 21-point score · upside multiples · PAT timeline
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([5,1])
with c1:
    ticker_input = st.text_input("t", label_visibility="collapsed",
        placeholder="BBOX.NS  ·  HFCL.NS  ·  ATHERENERG.NS  ·  NVDA  ·  PLTR")
with c2:
    go = st.button("Analyze ⚡", type="primary", use_container_width=True)

st.markdown('<p style="font-size:0.67rem;color:#bbb;margin-top:-8px;">India NSE → .NS suffix &nbsp;|&nbsp; US → ticker only</p>', unsafe_allow_html=True)

if go and ticker_input:
    with st.spinner(f"Analyzing {ticker_input.strip().upper()} …"):
        r, err = analyze(ticker_input)
    if err:
        st.error(f"⚠ {err}"); st.stop()

    cur = r["currency"]
    sc  = r["sc"]
    sc_ok = bool(sc and "_pl" in sc)

    # ── HERO ──
    sc_cls = "elite" if r["score"]>=16 else ("strong" if r["score"]>=12 else ("moderate" if r["score"]>=8 else "weak"))
    mkt_badge = '<span class="badge india">🇮🇳 NSE</span>' if r["is_india"] else '<span class="badge us">🇺🇸 US</span>'
    act_badge = f'<span class="badge {r["acls"]}">{r["action"]}</span>'

    st.markdown(f"""
    <div class="hero">
      <div class="score-ring {sc_cls}">
        <span class="snum">{r['score']}</span>
        <span class="sden">/ 21</span>
      </div>
      <div class="hero-body">
        <div class="hero-name">{r['name']}</div>
        <div class="hero-meta">{r['ticker']} · {r['sector']} · {r['industry']}</div>
        {act_badge} {mkt_badge}
        <div style="font-size:0.7rem;color:#888;margin-top:6px;">
          {cur}{r['price']} &nbsp;
          <span style="color:{'#16a34a' if r['dchg']>0 else '#dc2626'}">{r['dchg']:+.1f}%</span>
          &nbsp;·&nbsp; RSI {r['rsi']} &nbsp;·&nbsp; Vol {r['vol_ratio']:.1f}x &nbsp;·&nbsp;
          {r['pct_hi']:.1f}% below 52W high
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if r["is_india"] and not sc_ok:
        st.markdown("""<div class="dn">⚠ Screener.in data unavailable for this ticker.
        Showing yfinance values — may be incomplete for Indian mid/smallcaps.
        Screener.in works when running the app locally on your machine.</div>""", unsafe_allow_html=True)

    # ── MULTIPLES ──
    m = r["mults"]
    st.markdown('<div class="sec">Upside Multiple Estimate — 3-Year Horizon <span class="stag ca">model</span></div>', unsafe_allow_html=True)

    if m.get("bear") is not None:
        def tp(x): return f"{cur}{round(r['price']*x)}"
        pe_line   = (f"Current PE: {m['pe_current']}x → Fair PE: {m['pe_fair']}x → Bull PE: {m['pe_bull']}x") if m.get("pe_current") else ""
        cagr_line = f"Assumed PAT CAGR: ~{m['pat_cagr']}%/yr" + (f" + {m['margin_bonus']}% margin expansion bonus" if m.get('margin_bonus') else "")

        st.markdown(f"""
        <div class="mbox">
          <div class="mbox-title">Return Scenarios (current price = 1x)</div>
          <div class="mgrid4">
            <div class="mc bear">
              <div class="ml">Bear</div>
              <div class="mx">{m['bear']:.1f}x</div>
              <div class="mtp">{tp(m['bear'])}</div>
              <div class="mlo">Growth disappoints + PE compression</div>
            </div>
            <div class="mc cons">
              <div class="ml">Conservative</div>
              <div class="mx">{m['conservative']:.1f}x</div>
              <div class="mtp">{tp(m['conservative'])}</div>
              <div class="mlo">Moderate growth, no re-rating</div>
            </div>
            <div class="mc base">
              <div class="ml">Base Case</div>
              <div class="mx">{m['base']:.1f}x</div>
              <div class="mtp">{tp(m['base'])}</div>
              <div class="mlo">Sustained growth + mild PE re-rating</div>
            </div>
            <div class="mc bull">
              <div class="ml">Bull Case</div>
              <div class="mx">{m['bull']:.1f}x</div>
              <div class="mtp">{tp(m['bull'])}</div>
              <div class="mlo">Growth accelerates + full PE expansion</div>
            </div>
          </div>
          <div class="mbox-footer">
            {cagr_line}<br>{pe_line}<br><em>{m['logic']}</em>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAT TIMELINE ──
    if sc_ok and sc.get("_pat_series") and sc.get("_sales_series"):
        st.markdown(f'<div class="sec">Financial History — Revenue · OPM · PAT {stag("sc")}</div>', unsafe_allow_html=True)
        years   = sc.get("_years", [])
        sales   = sc.get("_sales_series", [])
        pat     = sc.get("_pat_series",   [])
        opm_s   = sc.get("_opm_series",   [])
        n       = min(len(years), len(sales), len(pat))
        years   = years[-n:]; sales = sales[-n:]; pat = pat[-n:]
        opm_s   = opm_s[-n:] if opm_s else [None]*n
        max_pat = max([p for p in pat if p], default=1) or 1
        cells   = ""
        for i, yr in enumerate(years):
            p    = pat[i]   if i<len(pat)   else None
            s    = sales[i] if i<len(sales) else None
            o    = opm_s[i] if i<len(opm_s) else None
            bar  = int((p/max_pat)*100) if p and max_pat else 0
            pcls = " hi" if i==len(years)-1 else ""
            cells += f"""<div class="tl-yr-col">
              <div class="tl-y">{yr}</div>
              <div class="tl-s">{'₹'+str(int(s))+'Cr' if s else '—'}</div>
              <div class="tl-o">{'OPM '+str(int(o))+'%' if o else '—'}</div>
              <div class="tl-bar-w"><div class="tl-bar" style="width:{bar}%"></div></div>
              <div class="tl-p{pcls}">{'₹'+str(round(p,0))+'Cr' if p else '—'}</div>
            </div>"""
        pat_mult = ""
        if len(pat)>=2 and pat[0] and pat[-1] and pat[0]!=0:
            pm_v = round(pat[-1]/pat[0], 1)
            pat_mult = f"&nbsp;·&nbsp; PAT grew <strong>{pm_v}x</strong> in {n} years"
        st.markdown(f"""
        <div class="timeline">
          <div style="font-size:0.67rem;color:#9ca3af;margin-bottom:8px;">
            Annual data{pat_mult}
          </div>
          <div class="tl-row">{cells}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── TECHNICALS ──
    st.markdown(f'<div class="sec">Technical Snapshot {stag("yf")}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="mg">
      {mc("Price",        f"{cur}{r['price']}")}
      {mc("Day Change",   f"{r['dchg']:+.1f}%",     "g" if r['dchg']>0 else "r")}
      {mc("Volume",       f"{r['vol_ratio']:.1f}x",  "g" if r['vol_ratio']>1.5 else "")}
      {mc("RSI",          f"{r['rsi']}",             "g" if 55<=r['rsi']<=70 else ("r" if r['rsi']>80 else ""))}
      {mc("52W High",     f"{r['pct_hi']}% away",    cc(r['pct_hi'],0,25,True))}
      {mc("3M Return",    f"{r['r3m']:+.0f}%",       "g" if r['r3m']>20 else ("r" if r['r3m']<0 else ""))}
      {mc("6M Return",    f"{r['r6m']:+.0f}%",       "g" if r['r6m']>30 else ("r" if r['r6m']<0 else ""))}
      {mc("12M Return",   f"{r['r12m']:+.0f}%",      "g" if r['r12m']>40 else "")}
      {mc("EMA21 Dist",   f"{r['d21']:+.1f}%",       "g" if 0<r['d21']<=5 else ("r" if r['d21']>10 else ""))}
      {mc("10D Range",    f"{r['r10d']:.1f}%",       "g" if r['r10d']<8 else "")}
      {mc("EMA Stack",    "✓ Aligned" if r['ema_ok'] else "✗ No", "g" if r['ema_ok'] else "r")}
      {mc("SEPA",         "✓ Pass" if r['sepa'] else "✗ Fail",    "g" if r['sepa'] else "r")}
    </div>""", unsafe_allow_html=True)

    # ── FUNDAMENTALS ──
    fund_src = stag(r["src_roe"] if r["src_roe"]!="missing" else "yf")
    st.markdown(f'<div class="sec">Fundamentals {fund_src}</div>', unsafe_allow_html=True)
    extra = ""
    if r["is_india"] and r.get("fii_chg") is not None:
        extra += mc("FII Chg QoQ", f"{r['fii_chg']:+.1f}%", "g" if r['fii_chg']>0 else "r")
    if r.get("promoter") is not None:
        extra += mc("Promoter %", f"{r['promoter']:.1f}%", "g" if r['promoter']>50 else "")
    st.markdown(f"""<div class="mg">
      {mc("EPS/PAT Growth", f"{r['eps_g']:+.0f}%",  "g" if r['eps_g']>25 else ("r" if r['eps_g']<0 else ""))}
      {mc("Rev Growth",     f"{r['rev_g']:+.0f}%",  "g" if r['rev_g']>15 else "")}
      {mc("OPM",            f"{r['opm']:.1f}%",     "g" if r['opm']>15 else "")}
      {mc("PAT Margin",     f"{r['pm']:.1f}%",      "g" if r['pm']>15 else ("r" if r['pm']<0 else ""))}
      {mc("RoE",            f"{r['roe']:.1f}%",     "g" if r['roe']>18 else ("r" if r['roe']<10 else ""))}
      {mc("RoCE",           f"{r['roce']:.1f}%",    "g" if r['roce']>15 else "")}
      {mc("Debt/Equity",    f"{r['de']:.2f}",       cc(r['de'], 0.3, 0.5, True))}
      {mc("OCF/PAT",        f"{r['ocf_p']:.2f}",    "g" if r['ocf_p']>0.7 else ("r" if r['ocf_p']<0.4 else ""))}
      {mc("Institutional",  f"{r['inst']:.1f}%",    "g" if r['inst']>30 else "")}
      {mc("Current PE",     f"{r['pe_c']:.0f}x" if r['pe_c'] else "—", "b")}
      {mc("MCap",           f"${r['mcap_b']}B")}
      {extra}
    </div>""", unsafe_allow_html=True)

    # ── TRADE LEVELS ──
    st.markdown(f'<div class="sec">Trade Levels {stag("ca")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tg">
      <div class="tc"><div class="tl">Stop Loss (2×ATR)</div><div class="tv r">{cur}{r['sl']}</div></div>
      <div class="tc"><div class="tl">Target 1 (+15%)</div><div class="tv g">{cur}{r['t1']}</div></div>
      <div class="tc"><div class="tl">Target 2 (+30%)</div><div class="tv g">{cur}{r['t2']}</div></div>
      <div class="tc"><div class="tl">Risk : Reward</div><div class="tv {'g' if r['rr']>=2 else ''}">{r['rr']:.1f}</div></div>
    </div>
    <p style="font-size:0.67rem;color:#9ca3af;margin-top:-3px;">
      EMA21 support: {cur}{r['ema21']} &nbsp;·&nbsp; ATR: {r['atr_pct']:.2f}%
    </p>
    """, unsafe_allow_html=True)

    # ── 21 SIGNALS ──
    st.markdown('<div class="sec">21-Point Signal Breakdown</div>', unsafe_allow_html=True)
    tier_map = {0:"Tier 1 — Fundamentals (7pt)", 7:"Tier 2 — Momentum Triggers (8pt)", 15:"Tier 3 — Multibagger DNA (6pt)"}
    tier_sc  = [0.0, 0.0, 0.0]
    pills    = ""
    for i, (lbl, detail, pts, status) in enumerate(r["sigs"]):
        if i in tier_map:
            pills += f'<div class="tier-lbl">{tier_map[i]}</div>'
        ti = 0 if i<7 else (1 if i<15 else 2)
        tier_sc[ti] += pts
        pills += f'<span class="pi {status}" title="{detail}">{lbl}: {detail} ({pts}pt)</span>'
    tmx = [7,8,6]
    summary = " &nbsp;·&nbsp; ".join([f"T{i+1}: {tier_sc[i]:.1f}/{tmx[i]}" for i in range(3)])
    st.markdown(f'<div style="font-size:0.68rem;color:#9ca3af;margin-bottom:5px;">{summary}</div><div class="sp">{pills}</div>', unsafe_allow_html=True)

    # ── CHARTS ──
    st.markdown(f'<div class="sec">Price Chart (1Y) {stag("yf")}</div>', unsafe_allow_html=True)
    ch = r["_chart"][["Close","EMA_21","EMA_50"]].copy()
    if "SMA_200" in r["_chart"].columns:
        ch["SMA_200"] = r["_chart"]["SMA_200"]
    ch.columns = [c.replace("_"," ") for c in ch.columns]
    st.line_chart(ch, height=280)
    st.markdown('<div class="sec">Volume</div>', unsafe_allow_html=True)
    st.bar_chart(r["_chart"][["Volume"]], height=110)

    # ── SCORE GUIDE ──
    st.markdown("""<div class="sec">Score Guide</div>
    <div class="mg">
      <div class="mce"><div class="ml">16–21</div><div class="mv g">Elite momentum</div></div>
      <div class="mce"><div class="ml">12–15</div><div class="mv b">Strong setup</div></div>
      <div class="mce"><div class="ml">8–11</div><div class="mv a">Watch only</div></div>
      <div class="mce"><div class="ml">Below 8</div><div class="mv r">Skip</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="disc">Educational tool only. Not financial advice. Multiples are model estimates — actual returns can vary significantly. Always verify data independently before making any trading decision.</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
      <div style="font-size:2.6rem;margin-bottom:12px;">📊</div>
      <p style="font-size:0.88rem;color:#666;max-width:500px;margin:0 auto;line-height:1.7;">
        Score any stock on 21 momentum criteria.<br>
        Indian stocks use <strong>Screener.in</strong> for real fundamentals
        (revenue, PAT growth, OPM, RoE, D/E, OCF).<br>
        US stocks use <strong>yfinance</strong>.<br>
        Includes upside <strong>multiples estimate</strong> and PAT growth timeline.
      </p>
      <div style="margin-top:18px;display:flex;flex-wrap:wrap;justify-content:center;gap:7px;">
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">BBOX.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">HFCL.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">ATHERENERG.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">TATACONSUM.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">KIMS.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">NVDA</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.78rem;color:#374151;">PLTR</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
