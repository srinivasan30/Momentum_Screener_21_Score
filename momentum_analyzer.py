"""
Momentum Analyzer v4 — Fully Corrected Screener.in Parser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
India:  Screener.in (fundamentals) + yfinance (price/technicals)
US:     yfinance (everything)

Bugs fixed vs v3:
  1. OPM:         Was ×100 (8%→800%). Now read as-is (screener stores as integer %)
  2. D/E:         Was looking for non-existent 'Total Debt'. Now: Borrowings/(EquityCap+Reserves)
  3. OCF:         Was looking for 'Cash from Operations'. Now: 'Cash from Operating Activity'
  4. Row names:   Screener uses 'Sales +','Net Profit +'. Now using 'in' substring match
  5. Inst %:      Now correctly latest-quarter FIIs + DIIs only
  6. PAT Margin:  Now computed as PAT/Sales×100 (not from yfinance)
  7. TTM:         Excluded from all FY calculations
  8. Series data: PAT/Sales/OPM series exclude TTM for correct timeline

Run:  streamlit run momentum_analyzer.py
"""

import math
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

.stApp { background:#f8f9fa; }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:1.8rem; padding-bottom:2rem; max-width:960px; }

.app-title { font-family:'JetBrains Mono',monospace; font-size:1.45rem; font-weight:700; color:#111; letter-spacing:-0.4px; }
.app-sub   { font-size:0.74rem; color:#999; margin-top:3px; margin-bottom:1.2rem; }

/* Hero */
.hero { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; display:flex; gap:18px; align-items:flex-start; margin-bottom:10px; }
.sring { width:84px; height:84px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0; font-family:'JetBrains Mono',monospace; }
.sring.elite    { background:#dcfce7; border:3px solid #16a34a; }
.sring.strong   { background:#dbeafe; border:3px solid #2563eb; }
.sring.moderate { background:#fef3c7; border:3px solid #d97706; }
.sring.weak     { background:#fef2f2; border:3px solid #dc2626; }
.sring .sn { font-size:1.65rem; font-weight:700; line-height:1; }
.sring.elite    .sn { color:#166534; }
.sring.strong   .sn { color:#1e40af; }
.sring.moderate .sn { color:#92400e; }
.sring.weak     .sn { color:#dc2626; }
.sring .sd { font-size:0.58rem; color:#aaa; font-weight:600; margin-top:1px; }
.hbody { flex:1; min-width:0; }
.hname { font-size:1.02rem; font-weight:700; color:#111; margin:0; line-height:1.3; }
.hmeta { font-size:0.71rem; color:#aaa; margin:2px 0 8px; }

/* Badges */
.bdg { display:inline-block; padding:4px 11px; border-radius:5px; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; margin-right:4px; margin-bottom:3px; }
.bdg.buy    { background:#dcfce7; color:#166534; }
.bdg.wait   { background:#fef3c7; color:#92400e; }
.bdg.watch  { background:#f1f5f9; color:#475569; }
.bdg.skip   { background:#fef2f2; color:#991b1b; }
.bdg.india  { background:#eff6ff; color:#1d4ed8; }
.bdg.us     { background:#f0fdf4; color:#15803d; }

/* Multiples */
.mbox { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; margin-bottom:10px; }
.mbox-t { font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:700; color:#aaa; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 12px; }
.mg4 { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; }
.mcard { border-radius:8px; padding:11px 8px; text-align:center; border:1px solid transparent; }
.mcard.bear { background:#fef2f2; border-color:#fecaca; }
.mcard.cons { background:#f0fdf4; border-color:#bbf7d0; }
.mcard.base { background:#eff6ff; border-color:#bfdbfe; }
.mcard.bull { background:#fefce8; border-color:#fde68a; }
.mcard .ml  { font-size:0.58rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; color:#888; }
.mcard .mx  { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:700; line-height:1.2; margin:3px 0 2px; }
.mcard.bear .mx { color:#dc2626; }
.mcard.cons .mx { color:#166534; }
.mcard.base .mx { color:#1e40af; }
.mcard.bull .mx { color:#92400e; }
.mcard .mtp { font-size:0.68rem; color:#555; font-weight:600; }
.mcard .mlo { font-size:0.6rem; color:#999; margin-top:3px; line-height:1.3; }
.mfooter { font-size:0.65rem; color:#aaa; margin-top:10px; line-height:1.6; }

/* Section header */
.sec { font-family:'JetBrains Mono',monospace; font-size:0.64rem; font-weight:700; color:#aaa; text-transform:uppercase; letter-spacing:1.5px; margin:14px 0 7px; padding-bottom:5px; border-bottom:1.5px solid #e5e7eb; }

/* Metric cells */
.mg  { display:grid; grid-template-columns:repeat(auto-fill,minmax(115px,1fr)); gap:6px; margin-bottom:7px; }
.mce { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:8px 11px; }
.mce .ml { font-size:0.57rem; color:#aaa; text-transform:uppercase; letter-spacing:0.4px; font-weight:600; }
.mce .mv { font-family:'JetBrains Mono',monospace; font-size:0.86rem; font-weight:700; color:#111; margin-top:2px; }
.mce .mv.g { color:#16a34a; } .mce .mv.r { color:#dc2626; }
.mce .mv.a { color:#d97706; } .mce .mv.b { color:#2563eb; }

/* Source tag */
.stg { display:inline-block; font-size:0.54rem; padding:1px 5px; border-radius:3px; font-weight:700; letter-spacing:0.3px; vertical-align:middle; margin-left:3px; }
.stg.sc { background:#eff6ff; color:#1d4ed8; }
.stg.yf { background:#f0fdf4; color:#15803d; }
.stg.ca { background:#fefce8; color:#854d0e; }

/* Trade grid */
.tg { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:5px; }
.tc { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:8px 11px; text-align:center; }
.tc .tl { font-size:0.57rem; color:#aaa; text-transform:uppercase; letter-spacing:0.4px; font-weight:600; }
.tc .tv { font-family:'JetBrains Mono',monospace; font-size:0.9rem; font-weight:700; margin-top:2px; }
.tc .tv.r { color:#dc2626; } .tc .tv.g { color:#16a34a; }

/* Signal pills */
.srow { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:5px; }
.srow .pi { display:inline-block; padding:3px 8px; border-radius:4px; font-size:0.65rem; font-weight:600; cursor:default; white-space:nowrap; }
.srow .pi.pass    { background:#dcfce7; color:#166534; }
.srow .pi.partial { background:#fef3c7; color:#92400e; }
.srow .pi.fail    { background:#fef2f2; color:#991b1b; }
.tlbl { width:100%; font-size:0.58rem; font-weight:700; color:#aaa; text-transform:uppercase; letter-spacing:1px; margin:6px 0 3px; }

/* Timeline */
.tl { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px; overflow-x:auto; }
.tl-row { display:flex; min-width:460px; }
.tl-col { flex:1; text-align:center; padding:5px 3px; border-right:1px solid #f0f0f0; }
.tl-col:last-child { border-right:none; }
.tl-y  { font-size:0.6rem; color:#aaa; font-weight:600; }
.tl-s  { font-size:0.66rem; color:#374151; font-weight:600; margin:2px 0; }
.tl-o  { font-size:0.62rem; color:#2563eb; font-weight:600; }
.tl-bw { height:5px; background:#f3f4f6; border-radius:3px; margin:4px 5px; overflow:hidden; }
.tl-b  { height:100%; border-radius:3px; background:#16a34a; }
.tl-p  { font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; color:#111; margin:2px 0; }
.tl-p.hi { color:#16a34a; }

/* Notice */
.notice { background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:8px 12px; font-size:0.68rem; color:#92400e; margin:6px 0 10px; }

/* Disclaimer */
.disc { background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:9px 13px; font-size:0.65rem; color:#aaa; margin-top:1.5rem; }

@media(max-width:640px){
  .hero { flex-direction:column; }
  .mg { grid-template-columns:repeat(2,1fr); }
  .tg { grid-template-columns:repeat(2,1fr); }
  .mg4 { grid-template-columns:repeat(2,1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SCREENER.IN — FIXED PARSER
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener(symbol: str) -> dict:
    """
    Scrape Screener.in with fully corrected parsing.
    All 8 bugs from v3 are fixed here.
    """
    symbol = symbol.replace(".NS","").replace(".BO","").upper()
    result = {"_ok": False}

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })

    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{symbol}{suffix}"
        try:
            resp = session.get(url, timeout=14)
            if resp.status_code != 200:
                continue
            if len(resp.text) < 5000:      # too short = error page
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # ── RATIOS ──
            ratios_sec = soup.find("section", id="ratios")
            if ratios_sec:
                for li in ratios_sec.find_all("li"):
                    n_tag = li.find("span", class_="name")
                    v_tag = li.find("span", class_="number")
                    if not n_tag or not v_tag:
                        continue
                    k   = n_tag.get_text(strip=True)
                    raw = v_tag.get_text(strip=True)
                    # Clean: commas, Cr., %, Rs., trailing spaces
                    cleaned = (raw.replace(",","").replace("Cr.","")
                                  .replace("%","").replace("Rs.","").strip())
                    # "153 / 62" → take first number only
                    if "/" in cleaned:
                        cleaned = cleaned.split("/")[0].strip()
                    try:
                        result[k] = float(cleaned)
                    except:
                        result[k] = cleaned

            # ── TABLE PARSER ──
            def parse_table(sec_id):
                """
                Returns (rows_dict, headers_list, has_ttm)
                rows_dict: {clean_row_name: [float_or_None, ...]}
                  - Row names stripped of trailing ' +' and whitespace
                  - Values: commas removed, floats; None if unparseable
                has_ttm: True if last header is 'TTM'
                """
                sec = soup.find("section", id=sec_id)
                if not sec:
                    return {}, [], False
                tbl = sec.find("table")
                if not tbl:
                    return {}, [], False

                hdrs = []
                for th in tbl.find_all("th"):
                    t = th.get_text(strip=True)
                    if t:
                        hdrs.append(t)

                has_ttm = bool(hdrs and hdrs[-1].upper() == "TTM")

                rows = {}
                for row in tbl.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    # Clean row name: remove trailing ' +', strip whitespace
                    raw_name = cells[0].get_text(strip=True)
                    row_name = raw_name.replace(" +", "").strip()
                    if not row_name:
                        continue
                    vals = []
                    for c in cells[1:]:
                        t = (c.get_text(strip=True)
                               .replace(",", "")
                               .replace("%", "")
                               .strip())
                        try:
                            vals.append(float(t))
                        except:
                            vals.append(None)
                    if vals:
                        rows[row_name] = vals
                return rows, hdrs, has_ttm

            pl_rows, pl_hdrs, pl_has_ttm = parse_table("profit-loss")
            bs_rows, bs_hdrs, _           = parse_table("balance-sheet")
            cf_rows, cf_hdrs, _           = parse_table("cash-flow")
            sh_rows, sh_hdrs, _           = parse_table("shareholding")

            if not pl_rows:
                continue     # not a valid page

            # ── HELPERS ──
            def get_fy_vals(rows, key, n=None):
                """
                Find a row by substring match on key.
                Return non-null FY values EXCLUDING TTM (last col if present).
                """
                for k, v in rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        # Exclude TTM column
                        if pl_has_ttm and len(vals) > 1:
                            vals = vals[:-1]
                        if n is not None:
                            return vals[-n:] if len(vals) >= n else vals
                        return vals
                return []

            def latest(rows, key):
                v = get_fy_vals(rows, key)
                return v[-1] if v else None

            def yoy_growth(rows, key):
                v = get_fy_vals(rows, key, 2)
                if len(v) == 2 and v[0] and v[0] != 0:
                    return round(((v[1] - v[0]) / abs(v[0])) * 100, 1)
                return None

            def multiyear_series(rows, key, exclude_ttm=True):
                for k, v in rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        if exclude_ttm and pl_has_ttm and len(vals) > 1:
                            vals = vals[:-1]
                        return vals
                return []

            def fy_headers(exclude_ttm=True):
                hdrs = [h for h in pl_hdrs if h]
                if exclude_ttm and hdrs and hdrs[-1].upper() == "TTM":
                    hdrs = hdrs[:-1]
                return hdrs

            # ── P&L METRICS ──
            # NOTE: OPM from screener is stored as integer percent (15 = 15%), NOT decimal
            sales_latest = latest(pl_rows, "Sales")
            pat_latest   = latest(pl_rows, "Net Profit")
            opm_latest   = latest(pl_rows, "OPM %")         # e.g. 15.0 means 15%
            eps_latest   = latest(pl_rows, "EPS in Rs")

            rev_growth   = yoy_growth(pl_rows, "Sales")
            pat_growth   = yoy_growth(pl_rows, "Net Profit")

            # PAT margin = PAT / Sales × 100 (computed, not from screener)
            pat_margin = (
                round((pat_latest / sales_latest) * 100, 1)
                if pat_latest and sales_latest and sales_latest != 0
                else None
            )

            # Gross margin proxy = Operating Profit / Sales × 100
            op_profit = latest(pl_rows, "Operating Profit")
            gross_margin = (
                round((op_profit / sales_latest) * 100, 1)
                if op_profit and sales_latest and sales_latest != 0
                else opm_latest   # fallback to OPM
            )

            # Series for timeline (FY only, no TTM)
            sales_series = multiyear_series(pl_rows, "Sales")
            pat_series   = multiyear_series(pl_rows, "Net Profit")
            opm_series   = multiyear_series(pl_rows, "OPM %")
            years_list   = fy_headers()

            # ── BALANCE SHEET METRICS ──
            # D/E = Borrowings / (Equity Capital + Reserves)
            def latest_bs(key):
                for k, v in bs_rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        return vals[-1] if vals else None
                return None

            borrowings   = latest_bs("Borrowings")
            equity_cap   = latest_bs("Equity Capital")
            reserves     = latest_bs("Reserves")
            total_equity = (equity_cap or 0) + (reserves or 0)
            debt_equity  = (
                round(borrowings / total_equity, 2)
                if borrowings is not None and total_equity and total_equity != 0
                else None
            )

            # ── CASH FLOW ──
            # Key on screener: "Cash from Operating Activity +"
            def latest_cf(key):
                for k, v in cf_rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        return vals[-1] if vals else None
                return None

            ocf = latest_cf("Operating Activity")

            # OCF/PAT
            ocf_pat = (
                round(ocf / pat_latest, 2)
                if ocf is not None and pat_latest and pat_latest != 0
                else None
            )

            # ── SHAREHOLDING ──
            def latest_sh(key):
                for k, v in sh_rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        return vals[-1] if vals else None
                return None

            def prev_sh(key):
                for k, v in sh_rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        return vals[-2] if len(vals) >= 2 else None
                return None

            promoter = latest_sh("Promoters")
            fii      = latest_sh("FIIs")
            dii      = latest_sh("DIIs")

            # Some pages use "Domestic Institutions" instead of "DIIs"
            if dii is None:
                dii = latest_sh("Domestic Institution")

            fii_prev = prev_sh("FIIs")
            fii_chg  = (
                round(fii - fii_prev, 2)
                if fii is not None and fii_prev is not None
                else None
            )
            inst = round((fii or 0) + (dii or 0), 1)

            # ── ROE / ROCE from ratios section (already parsed above) ──
            roe  = result.get("ROE")
            roce = result.get("ROCE")
            pe   = result.get("Stock P/E")
            mcap = result.get("Market Cap")   # in Cr

            # ── STORE ALL ──
            result.update({
                "_ok":          True,
                "_url":         url,
                "_suffix":      suffix,
                # Ratios (direct from ratios section)
                "roe":          roe,
                "roce":         roce,
                "pe":           pe,
                "mcap_cr":      mcap,
                # P&L
                "sales_latest": sales_latest,
                "pat_latest":   pat_latest,
                "opm":          opm_latest,        # already in % (e.g. 15.0)
                "gross_margin": gross_margin,
                "pat_margin":   pat_margin,
                "eps_latest":   eps_latest,
                "rev_growth":   rev_growth,
                "pat_growth":   pat_growth,
                # Balance sheet
                "debt_equity":  debt_equity,
                "borrowings":   borrowings,
                "total_equity": total_equity,
                # Cash flow
                "ocf":          ocf,
                "ocf_pat":      ocf_pat,
                # Shareholding
                "promoter":     promoter,
                "fii":          fii,
                "dii":          dii,
                "fii_chg":      fii_chg,
                "inst":         inst,
                # Series for timeline
                "_years":       years_list,
                "_sales":       sales_series,
                "_pat":         pat_series,
                "_opm":         opm_series,
            })
            return result

        except Exception as e:
            result["_error"] = str(e)
            continue

    return result


# ─────────────────────────────────────────────
# TECHNICALS
# ─────────────────────────────────────────────

def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
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
    d["H52"] = d["High"].rolling(lb).max()
    d["L52"] = d["Low"].rolling(lb).min()
    d["Pct_From_52H"] = ((d["H52"] - d["Close"]) / d["H52"]) * 100
    d["Pct_From_52L"] = ((d["Close"] - d["L52"]) / d["L52"]) * 100
    d["EMA21_Slope"] = d["EMA_21"].diff(5)
    if "SMA_200" in d.columns:
        d["SMA200_Slope"] = d["SMA_200"].diff(20)
    for p, lbl in [(21,"1M"),(63,"3M"),(126,"6M"),(252,"12M")]:
        d[f"Ret_{lbl}"] = d["Close"].pct_change(p)*100 if len(d)>=p else 0
    d["Range_10D"] = ((d["High"].rolling(10).max() - d["Low"].rolling(10).min()) / d["Close"]) * 100
    return d


# ─────────────────────────────────────────────
# MULTIPLES ENGINE
# ─────────────────────────────────────────────

def estimate_multiples(price, pe, pat_growth, opm, pat_margin):
    """
    4-scenario upside estimate over 3-year horizon.
    Based on: PAT CAGR × PE re-rating
    """
    res = {
        "bear": None, "conservative": None, "base": None, "bull": None,
        "logic": "", "cagr": None, "mb": 0,
        "pe": pe, "pe_fair": None, "pe_bull": None,
    }
    try:
        # ── CAGR from last PAT growth, moderated ──
        if pat_growth and pat_growth > 0:
            cagr = (35 if pat_growth > 100 else
                    30 if pat_growth > 50  else
                    25 if pat_growth > 30  else
                    18 if pat_growth > 15  else
                    max(pat_growth * 0.8, 5))
        else:
            cagr = 12

        # ── Margin expansion bonus ──
        mb = (5 if opm and opm < 8   else
              3 if opm and opm < 15  else
              1 if opm and opm < 25  else 0)

        # ── PE re-rating ──
        if pe and pe > 0:
            if   pe < 15: pf, pb, note = pe*2.0, pe*3.0, "Deeply undervalued — PE expansion potential is high"
            elif pe < 25: pf, pb, note = pe*1.5, pe*2.2, "Moderate PE — re-rating possible on sustained growth"
            elif pe < 40: pf, pb, note = pe*1.2, pe*1.6, "Fair PE — growth must justify current valuation"
            elif pe < 60: pf, pb, note = pe*1.0, pe*1.2, "Premium PE — purely a growth play, limited re-rating"
            else:         pf, pb, note = pe*0.85,pe*1.0, "High PE — de-rating risk if growth decelerates"
        else:
            pf, pb, pe_used, note = 25, 35, 20, "PE unavailable — using market-average estimate"
            pe = pe_used

        def mult(c, pe_end, pe_start):
            eps_m = (1 + c / 100) ** 3
            pe_m  = (pe_end / pe_start) if pe_start and pe_start > 0 else 1
            return round(min(eps_m * pe_m, 25), 1)

        res.update({
            "bear":         mult(max(cagr*0.3,-10), pe*0.7,  pe),
            "conservative": mult(cagr*0.6,          pf*0.9,  pe),
            "base":         mult(cagr + mb*0.5,      pf,      pe),
            "bull":         min(mult(cagr*1.3+mb,   pb,      pe), 20),
            "logic":        note,
            "cagr":         cagr,
            "mb":           mb,
            "pe":           pe,
            "pe_fair":      round(pf, 0),
            "pe_bull":      round(pb, 0),
        })
    except Exception as e:
        res["error"] = str(e)
    return res


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

def analyze(ticker_raw: str):
    ticker   = ticker_raw.strip().upper()
    is_india = ticker.endswith(".NS") or ticker.endswith(".BO")

    # ── Price data ──
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1y")
    info  = stock.info or {}

    if hist.empty or len(hist) < 50:
        return None, "Not enough price data (need 50+ trading days)."
    if not (info.get("regularMarketPrice") or info.get("currentPrice")):
        return None, "Could not fetch quote. Check ticker (India: add .NS e.g. HFCL.NS)."

    # ── Screener.in for India ──
    sc = {}
    if is_india:
        with st.spinner("Fetching Screener.in …"):
            sc = fetch_screener(ticker)

    sc_ok = sc.get("_ok", False)

    # ── Technicals ──
    d     = compute_technicals(hist)
    L     = d.iloc[-1]
    price = float(L["Close"])
    cur   = "₹" if is_india else "$"

    # ── Fundamental values ──
    # Priority: Screener.in (if ok) → yfinance → 0
    def sc_val(key, default=0.0):
        if sc_ok:
            v = sc.get(key)
            if v is not None:
                try: return float(v), "sc"
                except: pass
        return default, "missing"

    def yf_pct(key, default=0.0):
        v = info.get(key, default) or default
        try:
            v = float(v)
            return round(v * 100, 1) if abs(v) <= 1 else round(v, 1)
        except:
            return default

    # Core fundamentals
    if sc_ok:
        eps_g    = sc.get("pat_growth")    or 0.0
        rev_g    = sc.get("rev_growth")    or 0.0
        opm      = sc.get("opm")           or 0.0    # already in % from screener
        gm       = sc.get("gross_margin")  or opm
        pm       = sc.get("pat_margin")    or 0.0
        roe      = sc.get("roe")           or 0.0
        roce     = sc.get("roce")          or 0.0
        de       = sc.get("debt_equity")
        ocf_p    = sc.get("ocf_pat")
        pe_c     = sc.get("pe")
        inst     = sc.get("inst")          or 0.0
        fii_chg  = sc.get("fii_chg")
        promoter = sc.get("promoter")
        src_main = "sc"
    else:
        # yfinance fallback (US stocks and India fallback)
        eps_g    = yf_pct("earningsGrowth")
        rev_g    = yf_pct("revenueGrowth")
        opm      = yf_pct("operatingMargins")
        gm       = yf_pct("grossMargins")
        pm       = yf_pct("profitMargins")
        roe      = yf_pct("returnOnEquity")
        roce     = round(yf_pct("returnOnAssets") * 2, 1)
        de_raw   = info.get("debtToEquity", 0) or 0
        de       = round(float(de_raw) / 100, 2)
        ocf      = info.get("operatingCashflow", 0) or 0
        ni       = info.get("netIncomeToCommon", 1) or 1
        ocf_p    = round(ocf / ni, 2) if ni != 0 else 0.0
        pe_c     = info.get("trailingPE") or info.get("forwardPE")
        ir       = info.get("heldPercentInstitutions", 0) or 0
        inst     = round(float(ir)*100, 1) if abs(float(ir))<=1 else round(float(ir),1)
        fii_chg  = None
        promoter = None
        src_main = "yf"

    # Ensure numeric
    de    = float(de)    if de    is not None else 0.0
    ocf_p = float(ocf_p) if ocf_p is not None else 0.0
    eps_g = float(eps_g) if eps_g is not None else 0.0
    rev_g = float(rev_g) if rev_g is not None else 0.0
    opm   = float(opm)   if opm   is not None else 0.0
    gm    = float(gm)    if gm    is not None else 0.0
    pm    = float(pm)    if pm    is not None else 0.0
    roe   = float(roe)   if roe   is not None else 0.0
    roce  = float(roce)  if roce  is not None else 0.0
    inst  = float(inst)  if inst  is not None else 0.0
    pe_c  = float(pe_c)  if pe_c  is not None else 0.0

    # MCap
    if sc_ok and sc.get("mcap_cr"):
        mcap_b = round(float(sc["mcap_cr"]) * 0.012, 1)  # Cr → USD billion approx
    else:
        mcap_b = round((info.get("marketCap", 0) or 0) / 1e9, 1)

    # Technicals
    pct_hi  = round(float(L.get("Pct_From_52H", 99)), 1)
    pct_lo  = round(float(L.get("Pct_From_52L",  0)), 1)
    r3m     = round(float(L.get("Ret_3M",  0)), 1)
    r6m     = round(float(L.get("Ret_6M",  0)), 1)
    r12m    = round(float(L.get("Ret_12M", 0)), 1)
    vol_r   = round(float(L.get("Vol_Ratio", 0)), 2)
    dchg    = round(float(L.get("Daily_Chg", 0)), 2)
    rsi_v   = round(float(L.get("RSI", 50)), 1)
    r10d    = round(float(L.get("Range_10D", 99)), 1)
    atr_v   = float(L.get("ATR", 0)) or 0
    atr_pct = round(float(L.get("ATR_Pct", 0)), 2)

    ema21  = float(L.get("EMA_21",  price))
    ema50  = float(L.get("EMA_50",  price))
    sma200 = float(L.get("SMA_200", 0)) if "SMA_200" in d.columns else 0
    sma150 = float(L.get("SMA_150", 0)) if "SMA_150" in d.columns else 0
    sl200  = float(L.get("SMA200_Slope", 0)) if "SMA200_Slope" in d.columns else 0

    ema_ok = bool(price > ema21 > ema50 > sma200) if sma200>0 else bool(price>ema21>ema50)
    sepa   = bool(sma150 > sma200 > 0 and sl200 > 0 and price > sma150)
    d21    = round(((price - ema21) / ema21) * 100, 1) if ema21>0 else 0

    t_eps  = info.get("trailingEps", 0) or 0
    f_eps  = info.get("forwardEps",  0) or 0

    # ═════════════════════════════════════════
    # SCORING — 21 POINTS
    # ═════════════════════════════════════════
    score = 0.0
    sigs  = []

    def add(lbl, detail, pts):
        nonlocal score
        score += pts
        st_  = "pass" if pts >= 1 else ("partial" if pts > 0 else "fail")
        sigs.append((lbl, detail, pts, st_))

    # TIER 1 — FUNDAMENTALS (7 pts)
    add("EPS/PAT Growth",
        f"{eps_g:+.0f}%",
        1.0 if eps_g>30 else (0.5 if eps_g>15 else 0.0))

    add("Revenue / OPM",
        f"Rev {rev_g:.0f}% | OPM {opm:.0f}%",
        1.0 if rev_g>15 or opm>15 else (0.5 if rev_g>5 or opm>8 else 0.0))

    add("RoE",
        f"{roe:.1f}%",
        1.0 if roe>18 else (0.5 if roe>12 else 0.0))

    add("RoCE",
        f"{roce:.1f}%",
        1.0 if roce>15 else (0.5 if roce>10 else 0.0))

    add("OCF / PAT",
        f"{ocf_p:.2f}",
        1.0 if ocf_p>0.7 else (0.5 if ocf_p>0.4 else 0.0))

    add("Debt / Equity",
        f"{de:.2f}",
        1.0 if de<0.3 else (0.5 if de<0.5 else 0.0))

    add("Institutional %",
        f"{inst:.1f}%",
        1.0 if inst>30 else (0.5 if inst>15 else 0.0))

    # TIER 2 — MOMENTUM TRIGGERS (8 pts)
    add("52W High Proximity",
        f"{pct_hi:.1f}% away",
        1.0 if pct_hi<=10 else (0.5 if pct_hi<=25 else 0.0))

    add("52W Low Distance",
        f"+{pct_lo:.0f}%",
        0.5 if pct_lo>=30 else 0.0)

    rs = r3m*0.4 + r6m*0.6
    add("Relative Strength",
        f"3M {r3m:+.0f}% / 6M {r6m:+.0f}%",
        1.0 if rs>40 else (0.5 if rs>20 else 0.0))

    add("Volume Spike",
        f"{vol_r:.1f}x on {dchg:+.1f}%",
        1.0 if vol_r>1.5 and dchg>2 else (0.5 if vol_r>1.2 and dchg>0 else 0.0))

    add("EMA Stack",
        "P>21>50>200 ✓" if ema_ok else ("P>21>50" if price>ema21>ema50 else "Misaligned"),
        1.0 if ema_ok else (0.5 if price>ema21>ema50 else 0.0))

    add("SEPA (Minervini)",
        "Pass ✓" if sepa else ("P>200MA" if sma200>0 and price>sma200 else "Fail"),
        1.0 if sepa else (0.5 if sma200>0 and price>sma200 else 0.0))

    if t_eps > 0 and f_eps > t_eps:
        accel = round(((f_eps - t_eps) / abs(t_eps)) * 100, 0)
        add("Earnings Accel",
            f"Fwd +{accel:.0f}% vs Trail",
            1.0 if accel>10 else 0.5)
    elif eps_g > 30:
        add("Earnings Accel",
            f"Strong trend {eps_g:.0f}%",
            0.5)
    else:
        add("Earnings Accel",
            "No acceleration",
            0.0)

    add("RSI Zone",
        f"{rsi_v:.0f}",
        1.0 if 55<=rsi_v<=70 else (0.5 if 45<=rsi_v<=80 else 0.0))

    # TIER 3 — MULTIBAGGER DNA (6 pts)
    add("MCap Sweet Spot",
        f"${mcap_b}B",
        1.0 if 0.3<=mcap_b<=10 else (0.5 if 10<mcap_b<=50 else 0.0))

    add("Revenue Leader",
        f"+{rev_g:.0f}%",
        1.0 if rev_g>25 else (0.5 if rev_g>15 else 0.0))

    add("Margin Quality",
        f"OPM {opm:.0f}% / GM {gm:.0f}%",
        1.0 if opm>25 or gm>40 else (0.5 if opm>15 or gm>25 else 0.0))

    add("PAT Margin",
        f"{pm:.1f}%",
        1.0 if pm>15 else (0.5 if pm>8 else 0.0))

    add("Consolidation / Coil",
        f"{r10d:.1f}% range",
        1.0 if r10d<8 else (0.5 if r10d<15 else 0.0))

    add("Entry Zone",
        f"{d21:+.1f}% from EMA21",
        1.0 if 0<d21<=5 else (0.5 if -3<d21<=10 else 0.0))

    score = round(score, 1)

    # ── Action ──
    if score >= 16:
        act, acls = (("BUY NOW","buy") if d21<=5 else
                     ("BUY 50%","buy") if d21<=10 else
                     ("WAIT — PULLBACK","wait"))
    elif score >= 12:
        act, acls = (("BUY ON DIP","buy") if (d21<=5 or r10d<8) else
                     ("WATCHLIST","watch"))
    elif score >= 8:
        act, acls = "WATCHLIST", "watch"
    else:
        act, acls = "SKIP", "skip"

    # ── Trade levels ──
    sl = round(price - 2*atr_v, 2)
    t1 = round(price * 1.15, 2)
    t2 = round(price * 1.30, 2)
    rr = round((t1 - price) / max(price - sl, 0.01), 1)

    # ── Multiples ──
    mults = estimate_multiples(price, pe_c, eps_g, opm, pm)

    return {
        "ticker": ticker, "name": info.get("shortName") or info.get("longName") or ticker,
        "market": "INDIA" if is_india else "US",
        "currency": cur, "is_india": is_india, "sc_ok": sc_ok, "src": src_main,
        "sector": info.get("sector","—"), "industry": info.get("industry","—"),
        # Score
        "score": score, "sigs": sigs, "action": act, "acls": acls,
        # Technicals
        "price": round(price,2), "dchg": dchg, "vol_ratio": vol_r, "rsi": rsi_v,
        "pct_hi": pct_hi, "pct_lo": pct_lo,
        "r3m": r3m, "r6m": r6m, "r12m": r12m,
        "ema_ok": ema_ok, "sepa": sepa, "d21": d21, "r10d": r10d,
        "atr_pct": atr_pct, "ema21": round(ema21,2),
        # Fundamentals
        "eps_g": round(eps_g,1), "rev_g": round(rev_g,1),
        "opm": round(opm,1), "gm": round(gm,1), "pm": round(pm,1),
        "roe": round(roe,1), "roce": round(roce,1),
        "de": round(de,2), "ocf_p": round(ocf_p,2),
        "inst": round(inst,1), "pe_c": pe_c,
        "fii_chg": fii_chg, "promoter": promoter,
        "mcap_b": mcap_b,
        # Screener extras
        "sc": sc,
        # Trade
        "sl": sl, "t1": t1, "t2": t2, "rr": rr,
        # Multiples
        "mults": mults,
        # Chart
        "_chart": d,
    }, None


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────

def stg(src):
    return (f'<span class="stg sc">screener.in</span>' if src == "sc" else
            f'<span class="stg yf">yfinance</span>'    if src == "yf" else
            f'<span class="stg ca">calculated</span>')

def mc(lbl, val, cls=""):
    return f'<div class="mce"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>'

def cc(val, good, bad=None, inv=False):
    if val is None: return ""
    if inv: return "g" if val<good else ("r" if bad and val>bad else "a")
    return "g" if val>good else ("r" if bad is not None and val<bad else "")


# ─────────────────────────────────────────────
# APP UI
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-title">⚡ Momentum Analyzer</div>
<div class="app-sub">
  India → Screener.in fundamentals + yfinance technicals &nbsp;·&nbsp;
  US → yfinance &nbsp;·&nbsp; 21-pt score · multiples · PAT timeline
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([5, 1])
with c1:
    ticker_input = st.text_input(
        "t", label_visibility="collapsed",
        placeholder="BBOX.NS · HFCL.NS · ATHERENERG.NS · KIMS.NS · NVDA · PLTR"
    )
with c2:
    go = st.button("Analyze ⚡", type="primary", use_container_width=True)

st.markdown(
    '<p style="font-size:0.65rem;color:#bbb;margin-top:-8px;">'
    'India NSE → add .NS &nbsp;|&nbsp; US → ticker only</p>',
    unsafe_allow_html=True
)

if go and ticker_input:
    with st.spinner(f"Analyzing {ticker_input.strip().upper()} …"):
        r, err = analyze(ticker_input)
    if err:
        st.error(f"⚠ {err}")
        st.stop()

    sc    = r["sc"]
    cur   = r["currency"]

    # ── HERO ──
    sc_cls = ("elite"    if r["score"]>=16 else
              "strong"   if r["score"]>=12 else
              "moderate" if r["score"]>=8  else "weak")
    mkt_b = '<span class="bdg india">🇮🇳 NSE</span>' if r["is_india"] else '<span class="bdg us">🇺🇸 US</span>'
    act_b = f'<span class="bdg {r["acls"]}">{r["action"]}</span>'

    st.markdown(f"""
    <div class="hero">
      <div class="sring {sc_cls}">
        <span class="sn">{r['score']}</span>
        <span class="sd">/ 21</span>
      </div>
      <div class="hbody">
        <div class="hname">{r['name']}</div>
        <div class="hmeta">{r['ticker']} · {r['sector']} · {r['industry']}</div>
        {act_b} {mkt_b}
        <div style="font-size:0.68rem;color:#999;margin-top:6px;">
          {cur}{r['price']} &nbsp;
          <span style="color:{'#16a34a' if r['dchg']>0 else '#dc2626'}">{r['dchg']:+.1f}% today</span>
          &nbsp;·&nbsp; RSI {r['rsi']}
          &nbsp;·&nbsp; Vol {r['vol_ratio']:.1f}x
          &nbsp;·&nbsp; {r['pct_hi']:.1f}% from 52W high
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if r["is_india"] and not r["sc_ok"]:
        st.markdown("""
        <div class="notice">
          ⚠ Screener.in unavailable for this ticker (403/timeout).
          Showing yfinance values — may be inaccurate for Indian stocks.
          The scraper works correctly when running on your local machine.
        </div>""", unsafe_allow_html=True)

    # ── MULTIPLES ──
    m = r["mults"]
    st.markdown(f'<div class="sec">Upside Multiple Estimate — 3-Year Horizon {stg("ca")}</div>', unsafe_allow_html=True)

    if m.get("bear") is not None:
        def tp(x): return f"{cur}{round(r['price']*x):,.0f}"
        pe_line   = (f"Current PE: {m['pe']:.0f}x → Fair PE: {m['pe_fair']:.0f}x → Bull PE: {m['pe_bull']:.0f}x"
                     if m.get("pe") and m["pe"] > 0 else "")
        cagr_line = (f"Assumed PAT CAGR: ~{m['cagr']:.0f}%/yr"
                     + (f" + {m['mb']}% margin expansion bonus" if m.get("mb") else ""))

        st.markdown(f"""
        <div class="mbox">
          <div class="mbox-t">Return Scenarios (current price = 1.0×)</div>
          <div class="mg4">
            <div class="mcard bear">
              <div class="ml">Bear</div>
              <div class="mx">{m['bear']:.1f}×</div>
              <div class="mtp">{tp(m['bear'])}</div>
              <div class="mlo">Growth miss + PE compression</div>
            </div>
            <div class="mcard cons">
              <div class="ml">Conservative</div>
              <div class="mx">{m['conservative']:.1f}×</div>
              <div class="mtp">{tp(m['conservative'])}</div>
              <div class="mlo">Moderate growth, no re-rating</div>
            </div>
            <div class="mcard base">
              <div class="ml">Base Case</div>
              <div class="mx">{m['base']:.1f}×</div>
              <div class="mtp">{tp(m['base'])}</div>
              <div class="mlo">Growth sustains + mild re-rating</div>
            </div>
            <div class="mcard bull">
              <div class="ml">Bull Case</div>
              <div class="mx">{m['bull']:.1f}×</div>
              <div class="mtp">{tp(m['bull'])}</div>
              <div class="mlo">Acceleration + full PE expansion</div>
            </div>
          </div>
          <div class="mfooter">{cagr_line}<br>{pe_line}<br><em>{m['logic']}</em></div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAT TIMELINE ──
    if r["sc_ok"] and sc.get("_pat") and sc.get("_sales"):
        years = sc.get("_years", [])
        sales = sc.get("_sales", [])
        pat   = sc.get("_pat",   [])
        opm_s = sc.get("_opm",   [])
        n     = min(len(years), len(sales), len(pat))
        years, sales, pat = years[-n:], sales[-n:], pat[-n:]
        opm_s = opm_s[-n:] if opm_s else [None]*n
        max_p = max([p for p in pat if p], default=1) or 1

        pat_mult_str = ""
        if len(pat) >= 2 and pat[0] and pat[-1] and pat[0] != 0:
            pm_v = round(pat[-1] / pat[0], 1)
            pat_mult_str = f"&nbsp;·&nbsp; PAT <strong>{pm_v}×</strong> in {n} yrs"

        cells = ""
        for i, yr in enumerate(years):
            p   = pat[i]   if i < len(pat)   else None
            s   = sales[i] if i < len(sales) else None
            o   = opm_s[i] if i < len(opm_s) else None
            bar = int((p / max_p)*100) if p and max_p else 0
            pcls = " hi" if i == len(years)-1 else ""
            cells += f"""<div class="tl-col">
              <div class="tl-y">{yr}</div>
              <div class="tl-s">{"₹"+str(int(s))+"Cr" if s else "—"}</div>
              <div class="tl-o">{"OPM "+str(int(o))+"%" if o else "—"}</div>
              <div class="tl-bw"><div class="tl-b" style="width:{bar}%"></div></div>
              <div class="tl-p{pcls}">{"₹"+str(round(p,0))+"Cr" if p else "—"}</div>
            </div>"""

        st.markdown(f'<div class="sec">Financial History — Revenue · OPM · PAT {stg("sc")}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="tl">
          <div style="font-size:0.65rem;color:#aaa;margin-bottom:8px;">
            Annual data (FY){pat_mult_str}
          </div>
          <div class="tl-row">{cells}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── TECHNICALS ──
    st.markdown(f'<div class="sec">Technical Snapshot {stg("yf")}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="mg">
      {mc("Price",       f"{cur}{r['price']:,.2f}")}
      {mc("Day Change",  f"{r['dchg']:+.1f}%",     "g" if r['dchg']>0 else "r")}
      {mc("Volume",      f"{r['vol_ratio']:.1f}×",  "g" if r['vol_ratio']>1.5 else "")}
      {mc("RSI",         f"{r['rsi']}",             "g" if 55<=r['rsi']<=70 else ("r" if r['rsi']>80 else ""))}
      {mc("52W High",    f"{r['pct_hi']}% away",    cc(r['pct_hi'],0,25,True))}
      {mc("3M Return",   f"{r['r3m']:+.0f}%",       "g" if r['r3m']>20 else ("r" if r['r3m']<0 else ""))}
      {mc("6M Return",   f"{r['r6m']:+.0f}%",       "g" if r['r6m']>30 else ("r" if r['r6m']<0 else ""))}
      {mc("12M Return",  f"{r['r12m']:+.0f}%",      "g" if r['r12m']>40 else "")}
      {mc("EMA21 Dist",  f"{r['d21']:+.1f}%",       "g" if 0<r['d21']<=5 else ("r" if r['d21']>10 else ""))}
      {mc("10D Range",   f"{r['r10d']:.1f}%",       "g" if r['r10d']<8 else "")}
      {mc("EMA Stack",   "✓ Aligned" if r['ema_ok'] else "✗ No", "g" if r['ema_ok'] else "r")}
      {mc("SEPA",        "✓ Pass" if r['sepa'] else "✗ Fail",    "g" if r['sepa'] else "r")}
    </div>""", unsafe_allow_html=True)

    # ── FUNDAMENTALS ──
    st.markdown(f'<div class="sec">Fundamentals {stg(r["src"])}</div>', unsafe_allow_html=True)
    extras = ""
    if r["is_india"] and r.get("fii_chg") is not None:
        extras += mc("FII Δ QoQ", f"{r['fii_chg']:+.1f}%", "g" if r["fii_chg"]>0 else "r")
    if r.get("promoter") is not None:
        extras += mc("Promoter", f"{r['promoter']:.1f}%", "g" if r["promoter"]>50 else "")

    st.markdown(f"""<div class="mg">
      {mc("EPS/PAT Growth", f"{r['eps_g']:+.0f}%",  "g" if r['eps_g']>25 else ("r" if r['eps_g']<0 else ""))}
      {mc("Rev Growth",     f"{r['rev_g']:+.0f}%",  "g" if r['rev_g']>15 else "")}
      {mc("OPM",            f"{r['opm']:.0f}%",     "g" if r['opm']>15 else "")}
      {mc("PAT Margin",     f"{r['pm']:.1f}%",      "g" if r['pm']>15 else ("r" if r['pm']<0 else ""))}
      {mc("RoE",            f"{r['roe']:.1f}%",     "g" if r['roe']>18 else ("r" if r['roe']<10 else ""))}
      {mc("RoCE",           f"{r['roce']:.1f}%",    "g" if r['roce']>15 else "")}
      {mc("Debt/Equity",    f"{r['de']:.2f}",       cc(r['de'],0.3,0.5,True))}
      {mc("OCF/PAT",        f"{r['ocf_p']:.2f}",    "g" if r['ocf_p']>0.7 else ("r" if 0<r['ocf_p']<0.4 else ""))}
      {mc("Institutional",  f"{r['inst']:.1f}%",    "g" if r['inst']>30 else "")}
      {mc("PE",             f"{r['pe_c']:.0f}×" if r['pe_c'] else "—", "b")}
      {mc("MCap",           f"${r['mcap_b']:.1f}B")}
      {extras}
    </div>""", unsafe_allow_html=True)

    # ── TRADE LEVELS ──
    st.markdown(f'<div class="sec">Trade Levels {stg("ca")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tg">
      <div class="tc"><div class="tl">Stop Loss (2×ATR)</div><div class="tv r">{cur}{r['sl']:,.2f}</div></div>
      <div class="tc"><div class="tl">Target 1 (+15%)</div><div class="tv g">{cur}{r['t1']:,.2f}</div></div>
      <div class="tc"><div class="tl">Target 2 (+30%)</div><div class="tv g">{cur}{r['t2']:,.2f}</div></div>
      <div class="tc"><div class="tl">Risk : Reward</div><div class="tv {'g' if r['rr']>=2 else ''}">{r['rr']:.1f}</div></div>
    </div>
    <p style="font-size:0.65rem;color:#aaa;margin-top:-2px;">
      EMA21 support: {cur}{r['ema21']:,.2f} &nbsp;·&nbsp; ATR: {r['atr_pct']:.2f}%
    </p>""", unsafe_allow_html=True)

    # ── 21 SIGNALS ──
    st.markdown('<div class="sec">21-Point Signal Breakdown</div>', unsafe_allow_html=True)
    tier_map = {
        0:  "Tier 1 — Fundamentals (7pt)",
        7:  "Tier 2 — Momentum Triggers (8pt)",
        15: "Tier 3 — Multibagger DNA (6pt)",
    }
    tier_sc = [0.0, 0.0, 0.0]
    pills   = ""
    for i, (lbl, detail, pts, status) in enumerate(r["sigs"]):
        if i in tier_map:
            pills += f'<div class="tlbl">{tier_map[i]}</div>'
        ti = 0 if i<7 else (1 if i<15 else 2)
        tier_sc[ti] += pts
        pills += f'<span class="pi {status}" title="{detail}">{lbl}: {detail} ({pts}pt)</span>'

    tmx  = [7, 8, 6]
    summ = " &nbsp;·&nbsp; ".join([f"T{i+1}: {tier_sc[i]:.1f}/{tmx[i]}" for i in range(3)])
    st.markdown(
        f'<div style="font-size:0.66rem;color:#aaa;margin-bottom:5px;">{summ}</div>'
        f'<div class="srow">{pills}</div>',
        unsafe_allow_html=True
    )

    # ── CHARTS ──
    st.markdown(f'<div class="sec">Price Chart (1Y) + EMAs {stg("yf")}</div>', unsafe_allow_html=True)
    ch = r["_chart"][["Close","EMA_21","EMA_50"]].copy()
    if "SMA_200" in r["_chart"].columns:
        ch["SMA_200"] = r["_chart"]["SMA_200"]
    ch.columns = [c.replace("_"," ") for c in ch.columns]
    st.line_chart(ch, height=270)

    st.markdown('<div class="sec">Volume</div>', unsafe_allow_html=True)
    st.bar_chart(r["_chart"][["Volume"]], height=110)

    # ── SCORE GUIDE ──
    st.markdown("""<div class="sec">Score Guide</div><div class="mg">
      <div class="mce"><div class="ml">16 – 21</div><div class="mv g">Elite — act fast</div></div>
      <div class="mce"><div class="ml">12 – 15</div><div class="mv b">Strong — enter on dip</div></div>
      <div class="mce"><div class="ml">8 – 11</div><div class="mv a">Watch — not yet</div></div>
      <div class="mce"><div class="ml">Below 8</div><div class="mv r">Skip</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
      Educational tool only. Not financial advice. Multiples are model projections —
      actual returns depend on execution and market conditions.
      Always cross-verify data on Screener.in before trading.
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
      <div style="font-size:2.5rem;margin-bottom:12px;">📊</div>
      <p style="font-size:0.86rem;color:#666;max-width:500px;margin:0 auto;line-height:1.75;">
        Score any stock on 21 momentum criteria.<br>
        Indian stocks pull <strong>real fundamentals from Screener.in</strong>
        — Revenue, PAT, OPM, RoE, D/E, OCF/PAT, FII/DII shareholding.<br>
        US stocks use <strong>yfinance</strong>.<br>
        Includes <strong>upside multiple scenarios</strong> + PAT growth timeline.
      </p>
      <div style="margin-top:18px;display:flex;flex-wrap:wrap;justify-content:center;gap:7px;">
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">BBOX.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">HFCL.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">ATHERENERG.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">TATACONSUM.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">KIMS.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">DSSL.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">NVDA</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;color:#374151;">PLTR</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
