"""
Momentum Analyzer v5 — All data bugs fixed
Run: streamlit run momentum_analyzer.py
"""
import math, warnings, requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Momentum Analyzer", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
.stApp{background:#f8f9fa;} html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;} .block-container{padding-top:1.8rem;padding-bottom:2rem;max-width:960px;}
.app-title{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:700;color:#111;}
.app-sub{font-size:0.74rem;color:#999;margin-top:3px;margin-bottom:1.2rem;}
.hero{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;display:flex;gap:18px;align-items:flex-start;margin-bottom:10px;}
.sring{width:84px;height:84px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;font-family:'JetBrains Mono',monospace;}
.sring.elite{background:#dcfce7;border:3px solid #16a34a;} .sring.strong{background:#dbeafe;border:3px solid #2563eb;}
.sring.moderate{background:#fef3c7;border:3px solid #d97706;} .sring.weak{background:#fef2f2;border:3px solid #dc2626;}
.sring .sn{font-size:1.65rem;font-weight:700;line-height:1;}
.sring.elite .sn{color:#166534;} .sring.strong .sn{color:#1e40af;} .sring.moderate .sn{color:#92400e;} .sring.weak .sn{color:#dc2626;}
.sring .sd{font-size:0.58rem;color:#aaa;font-weight:600;margin-top:1px;}
.hbody{flex:1;min-width:0;} .hname{font-size:1.02rem;font-weight:700;color:#111;margin:0;}
.hmeta{font-size:0.71rem;color:#aaa;margin:2px 0 8px;}
.bdg{display:inline-block;padding:4px 11px;border-radius:5px;font-size:0.7rem;font-weight:700;text-transform:uppercase;margin-right:4px;margin-bottom:3px;}
.bdg.buy{background:#dcfce7;color:#166534;} .bdg.wait{background:#fef3c7;color:#92400e;}
.bdg.watch{background:#f1f5f9;color:#475569;} .bdg.skip{background:#fef2f2;color:#991b1b;}
.bdg.india{background:#eff6ff;color:#1d4ed8;} .bdg.us{background:#f0fdf4;color:#15803d;}
.mbox{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin-bottom:10px;}
.mbox-t{font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1.5px;margin:0 0 12px;}
.mg4{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;}
.mc{border-radius:8px;padding:11px 8px;text-align:center;border:1px solid transparent;}
.mc.bear{background:#fef2f2;border-color:#fecaca;} .mc.cons{background:#f0fdf4;border-color:#bbf7d0;}
.mc.base{background:#eff6ff;border-color:#bfdbfe;} .mc.bull{background:#fefce8;border-color:#fde68a;}
.mc .ml{font-size:0.58rem;font-weight:700;text-transform:uppercase;color:#888;}
.mc .mx{font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:700;line-height:1.2;margin:3px 0 2px;}
.mc.bear .mx{color:#dc2626;} .mc.cons .mx{color:#166534;} .mc.base .mx{color:#1e40af;} .mc.bull .mx{color:#92400e;}
.mc .mtp{font-size:0.68rem;color:#555;font-weight:600;} .mc .mlo{font-size:0.6rem;color:#999;margin-top:3px;line-height:1.3;}
.mfooter{font-size:0.65rem;color:#aaa;margin-top:10px;line-height:1.6;}
.sec{font-family:'JetBrains Mono',monospace;font-size:0.64rem;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1.5px;margin:14px 0 7px;padding-bottom:5px;border-bottom:1.5px solid #e5e7eb;}
.mg{display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:6px;margin-bottom:7px;}
.mce{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 11px;}
.mce .ml{font-size:0.57rem;color:#aaa;text-transform:uppercase;letter-spacing:0.4px;font-weight:600;}
.mce .mv{font-family:'JetBrains Mono',monospace;font-size:0.86rem;font-weight:700;color:#111;margin-top:2px;}
.mce .mv.g{color:#16a34a;} .mce .mv.r{color:#dc2626;} .mce .mv.a{color:#d97706;} .mce .mv.b{color:#2563eb;} .mce .mv.dim{color:#bbb;}
.stg{display:inline-block;font-size:0.54rem;padding:1px 5px;border-radius:3px;font-weight:700;vertical-align:middle;margin-left:3px;}
.stg.sc{background:#eff6ff;color:#1d4ed8;} .stg.yf{background:#f0fdf4;color:#15803d;} .stg.ca{background:#fefce8;color:#854d0e;}
.tg{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:5px;}
.tc{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 11px;text-align:center;}
.tc .tl{font-size:0.57rem;color:#aaa;text-transform:uppercase;font-weight:600;}
.tc .tv{font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:700;margin-top:2px;}
.tc .tv.r{color:#dc2626;} .tc .tv.g{color:#16a34a;}
.srow{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:5px;}
.srow .pi{display:inline-block;padding:3px 8px;border-radius:4px;font-size:0.65rem;font-weight:600;cursor:default;white-space:nowrap;}
.srow .pi.pass{background:#dcfce7;color:#166534;} .srow .pi.partial{background:#fef3c7;color:#92400e;} .srow .pi.fail{background:#fef2f2;color:#991b1b;}
.tlbl{width:100%;font-size:0.58rem;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px;}
.tl{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;overflow-x:auto;}
.tl-row{display:flex;min-width:460px;}
.tl-col{flex:1;text-align:center;padding:5px 3px;border-right:1px solid #f0f0f0;}
.tl-col:last-child{border-right:none;}
.tl-y{font-size:0.6rem;color:#aaa;font-weight:600;} .tl-s{font-size:0.66rem;color:#374151;font-weight:600;margin:2px 0;}
.tl-o{font-size:0.62rem;color:#2563eb;font-weight:600;}
.tl-bw{height:5px;background:#f3f4f6;border-radius:3px;margin:4px 5px;overflow:hidden;}
.tl-b{height:100%;border-radius:3px;background:#16a34a;}
.tl-p{font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:700;color:#111;margin:2px 0;} .tl-p.hi{color:#16a34a;}
.notice{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:8px 12px;font-size:0.68rem;color:#92400e;margin:6px 0 10px;}
.disc{background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:9px 13px;font-size:0.65rem;color:#aaa;margin-top:1.5rem;}
@media(max-width:640px){.hero{flex-direction:column;}.mg{grid-template-columns:repeat(2,1fr);}.tg{grid-template-columns:repeat(2,1fr);}.mg4{grid-template-columns:repeat(2,1fr);}}
</style>
""", unsafe_allow_html=True)

# ── SAFE HELPERS ──
NaN = float("nan")

def ok(v):
    if v is None: return False
    try: return math.isfinite(float(v)) and float(v) != 0
    except: return False

def ok0(v):
    """ok even if zero (for D/E = 0)"""
    if v is None: return False
    try: return math.isfinite(float(v))
    except: return False

def sf(v, d=NaN):
    if v is None: return d
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except: return d

def fmt(v, suf="", dec=1, pre=""):
    if not ok0(v): return "—"
    try:
        v = float(v)
        if dec == 0: return f"{pre}{v:,.0f}{suf}"
        return f"{pre}{v:,.{dec}f}{suf}"
    except: return "—"

def yf_to_pct(v):
    """Convert yfinance decimal ratio to percent. 0.226 -> 22.6. 25.4 stays 25.4."""
    if v is None: return NaN
    try:
        v = float(v)
        return round(v * 100, 1) if abs(v) <= 2.0 else round(v, 1)
    except: return NaN

def stag(s):
    m = {"sc": ("sc","screener.in"), "yf": ("yf","yfinance"), "ca": ("ca","calc")}
    cls, lbl = m.get(s, ("yf","yfinance"))
    return f'<span class="stg {cls}">{lbl}</span>'

def mc(lbl, val, cls=""):
    return f'<div class="mce"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>'

def cc(v, good, bad=None, inv=False):
    if not ok0(v): return "dim"
    v = float(v)
    if inv: return "g" if v < good else ("r" if bad is not None and v > bad else "a")
    return "g" if v > good else ("r" if bad is not None and v < bad else "")


# ── SCREENER.IN SCRAPER ──
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener(symbol):
    sym = symbol.replace(".NS","").replace(".BO","").upper()
    result = {"_ok": False}
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    for suffix in ["/consolidated/", "/"]:
        url = f"https://www.screener.in/company/{sym}{suffix}"
        try:
            resp = session.get(url, timeout=14)
            if resp.status_code != 200 or len(resp.text) < 5000:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Ratios
            rs = soup.find("section", id="ratios")
            if rs:
                for li in rs.find_all("li"):
                    nt = li.find("span", class_="name")
                    vt = li.find("span", class_="number")
                    if not nt or not vt: continue
                    k = nt.get_text(strip=True)
                    raw = vt.get_text(strip=True)
                    c = raw.replace(",","").replace("Cr.","").replace("%","").replace("Rs.","").strip()
                    if "/" in c: c = c.split("/")[0].strip()
                    try: result[k] = float(c)
                    except: result[k] = c

            def parse_tbl(sid):
                sec = soup.find("section", id=sid)
                if not sec: return {}, [], False
                tbl = sec.find("table")
                if not tbl: return {}, [], False
                hdrs = [th.get_text(strip=True) for th in tbl.find_all("th") if th.get_text(strip=True)]
                has_ttm = bool(hdrs and hdrs[-1].upper() == "TTM")
                rows = {}
                for row in tbl.find_all("tr"):
                    cells = row.find_all(["td","th"])
                    if len(cells) < 2: continue
                    rn = cells[0].get_text(strip=True).replace(" +","").strip()
                    if not rn: continue
                    vals = []
                    for c in cells[1:]:
                        t = c.get_text(strip=True).replace(",","").replace("%","").strip()
                        try: vals.append(float(t))
                        except: vals.append(None)
                    if vals: rows[rn] = vals
                return rows, hdrs, has_ttm

            pl, pl_h, pl_ttm = parse_tbl("profit-loss")
            bs, _,   _       = parse_tbl("balance-sheet")
            cf, _,   _       = parse_tbl("cash-flow")
            sh, _,   _       = parse_tbl("shareholding")
            if not pl: continue

            def fy(rows, key, n=None):
                for k,v in rows.items():
                    if key.lower() in k.lower():
                        vals = [x for x in v if x is not None]
                        if pl_ttm and len(vals) > 1: vals = vals[:-1]
                        return (vals[-n:] if n and len(vals)>=n else vals)
                return []

            def lat_pl(key):
                v = fy(pl, key); return v[-1] if v else NaN
            def lat_bs(key):
                for k,v in bs.items():
                    if key.lower() in k.lower():
                        vals=[x for x in v if x is not None]; return vals[-1] if vals else NaN
                return NaN
            def lat_cf(key):
                for k,v in cf.items():
                    if key.lower() in k.lower():
                        vals=[x for x in v if x is not None]; return vals[-1] if vals else NaN
                return NaN
            def lat_sh(key):
                for k,v in sh.items():
                    if key.lower() in k.lower():
                        vals=[x for x in v if x is not None]; return vals[-1] if vals else NaN
                return NaN
            def prv_sh(key):
                for k,v in sh.items():
                    if key.lower() in k.lower():
                        vals=[x for x in v if x is not None]; return vals[-2] if len(vals)>=2 else NaN
                return NaN
            def yoy(key):
                v=fy(pl,key,2)
                if len(v)==2 and v[0] and v[0]!=0: return round(((v[1]-v[0])/abs(v[0]))*100,1)
                return NaN
            def ser(key):
                for k,v in pl.items():
                    if key.lower() in k.lower():
                        vals=[x for x in v if x is not None]
                        return vals[:-1] if pl_ttm and len(vals)>1 else vals
                return []

            sales     = lat_pl("Sales")
            pat       = lat_pl("Net Profit")
            op_p      = lat_pl("Operating Profit")
            opm       = lat_pl("OPM %")       # already in % (15 = 15%)
            eps       = lat_pl("EPS in Rs")
            rev_g     = yoy("Sales")
            pat_g     = yoy("Net Profit")
            pm        = round((sf(pat)/sf(sales))*100,1)  if ok(pat) and ok(sales) else NaN
            gm        = round((sf(op_p)/sf(sales))*100,1) if ok(op_p) and ok(sales) else NaN

            borrow    = lat_bs("Borrowings")
            eq_cap    = lat_bs("Equity Capital")
            reserves  = lat_bs("Reserves")
            tot_eq    = sf(eq_cap,0)+sf(reserves,0)
            de        = round(sf(borrow)/tot_eq,2) if tot_eq>0 and ok(borrow) else NaN

            ocf       = lat_cf("Operating Activity")
            ocf_pat   = round(sf(ocf)/sf(pat),2) if ok(ocf) and ok(pat) else NaN

            promoter  = lat_sh("Promoters")
            fii       = lat_sh("FIIs")
            dii       = lat_sh("DIIs")
            if not ok0(dii): dii = lat_sh("Domestic Institution")
            fii_prev  = prv_sh("FIIs")
            fii_chg   = round(sf(fii)-sf(fii_prev),2) if ok0(fii) and ok0(fii_prev) else NaN
            inst      = round(sf(fii,0)+sf(dii,0),1)

            roe  = sf(result.get("ROE"))
            roce = sf(result.get("ROCE"))
            pe   = sf(result.get("Stock P/E"))

            yrs = [h for h in pl_h if h]
            if pl_ttm and yrs: yrs = yrs[:-1]

            result.update({
                "_ok":True, "_url":url,
                "roe":roe, "roce":roce, "pe":pe,
                "sales":sales, "pat":pat, "opm":opm, "gm":gm, "pm":pm, "eps":eps,
                "rev_g":rev_g, "pat_g":pat_g,
                "de":de, "ocf_pat":ocf_pat,
                "promoter":promoter, "fii":fii, "dii":dii, "fii_chg":fii_chg, "inst":inst,
                "_years":yrs, "_sales":ser("Sales"), "_pat":ser("Net Profit"), "_opm":ser("OPM %"),
            })
            return result
        except Exception as e:
            result["_error"]=str(e); continue
    return result


# ── TECHNICALS ──
def compute_technicals(df):
    d = df.copy()
    for s in [10,21,50]:
        d[f"EMA_{s}"] = d["Close"].ewm(span=s, adjust=False).mean()
    for s in [100,150,200]:
        if len(d) >= s: d[f"SMA_{s}"] = d["Close"].rolling(s).mean()
    d["Vol_20"] = d["Volume"].rolling(20).mean()
    d["Vol_R"]  = d["Volume"] / d["Vol_20"]
    d["RSI"]    = RSIIndicator(d["Close"], window=14).rsi()
    atr = AverageTrueRange(d["High"], d["Low"], d["Close"], window=14)
    d["ATR"] = atr.average_true_range()
    d["ATR_Pct"] = (d["ATR"] / d["Close"]) * 100
    d["Chg"] = d["Close"].pct_change() * 100
    lb = min(252, len(d))
    d["H52"] = d["High"].rolling(lb).max()
    d["L52"]  = d["Low"].rolling(lb).min()
    d["Hi_Dist"] = ((d["H52"] - d["Close"]) / d["H52"]) * 100
    d["Lo_Dist"] = ((d["Close"] - d["L52"]) / d["L52"]) * 100
    d["E21_Slope"] = d["EMA_21"].diff(5)
    if "SMA_200" in d.columns: d["S200_Slope"] = d["SMA_200"].diff(20)
    for p, lbl in [(21,"1M"),(63,"3M"),(126,"6M"),(252,"12M")]:
        d[f"R_{lbl}"] = d["Close"].pct_change(p)*100 if len(d)>=p else NaN
    d["Rng10"] = ((d["High"].rolling(10).max()-d["Low"].rolling(10).min())/d["Close"])*100
    return d


# ── MULTIPLES ──
def multiples(price, pe, pat_g, opm):
    r = {"bear":NaN,"cons":NaN,"base":NaN,"bull":NaN,"note":"","cagr":NaN,"mb":0,"pe_f":NaN,"pe_b":NaN}
    try:
        pg = sf(pat_g, 0)
        cagr = (35 if pg>100 else 30 if pg>50 else 25 if pg>30 else 18 if pg>15 else max(pg*0.8,5)) if pg>0 else 12
        opm_v = sf(opm, 20)
        mb = 5 if opm_v<8 else (3 if opm_v<15 else (1 if opm_v<25 else 0))
        pe_v = sf(pe, 20)
        if pe_v > 0:
            if   pe_v<15: pf,pb,note = pe_v*2.0,pe_v*3.0,"Deeply undervalued — PE expansion possible"
            elif pe_v<25: pf,pb,note = pe_v*1.5,pe_v*2.2,"Moderate PE — re-rating on growth"
            elif pe_v<40: pf,pb,note = pe_v*1.2,pe_v*1.6,"Fair PE — growth must drive returns"
            elif pe_v<60: pf,pb,note = pe_v*1.0,pe_v*1.2,"Premium PE — pure growth play"
            else:         pf,pb,note = pe_v*0.85,pe_v*1.0,"High PE — de-rating risk"
        else:
            pf,pb,note = 25,35,"PE unavailable — using estimate"
        def m(c, pe_e, pe_s): return round(min((1+c/100)**3*(pe_e/pe_s if pe_s>0 else 1),25),1)
        r.update({"bear":m(max(cagr*0.3,-10),pe_v*0.7,pe_v), "cons":m(cagr*0.6,pf*0.9,pe_v),
                  "base":m(cagr+mb*0.5,pf,pe_v), "bull":min(m(cagr*1.3+mb,pb,pe_v),20),
                  "note":note, "cagr":cagr, "mb":mb, "pe_f":round(pf,0), "pe_b":round(pb,0)})
    except Exception as e: r["err"]=str(e)
    return r


# ── ANALYZE ──
def analyze(raw):
    ticker   = raw.strip().upper()
    is_india = ticker.endswith(".NS") or ticker.endswith(".BO")
    cur      = "\u20b9" if is_india else "$"

    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1y")
    info  = {}
    try: info = stock.info or {}
    except: pass

    if hist.empty or len(hist) < 20:
        return None, "No price history. Check ticker (India: HFCL.NS, US: NVDA)."

    # Price — use info first (more reliable), fall back to hist
    price_raw = (info.get("currentPrice") or info.get("regularMarketPrice")
                 or info.get("previousClose"))
    price = sf(price_raw) if ok(price_raw) else float(hist["Close"].iloc[-1])
    if not ok(price) or price <= 0:
        return None, "Cannot determine price."

    # Screener.in for India
    sc    = {}
    sc_ok = False
    if is_india:
        with st.spinner("Fetching Screener.in …"):
            sc    = fetch_screener(ticker)
            sc_ok = sc.get("_ok", False)

    # Technicals
    d   = compute_technicals(hist)
    L   = d.iloc[-1]
    T   = lambda col: sf(L.get(col))

    hi_dist = T("Hi_Dist"); lo_dist = T("Lo_Dist")
    r3m = T("R_3M"); r6m = T("R_6M"); r12m = T("R_12M")
    vol_r = T("Vol_R"); dchg = T("Chg"); rsi_v = T("RSI")
    rng10 = T("Rng10"); atr_v = sf(T("ATR"), 0); atr_pct = T("ATR_Pct")
    ema21  = sf(T("EMA_21"),  price)
    ema50  = sf(T("EMA_50"),  price)
    sma200 = sf(T("SMA_200"), 0) if "SMA_200" in d.columns else 0
    sma150 = sf(T("SMA_150"), 0) if "SMA_150" in d.columns else 0
    sl200  = sf(T("S200_Slope"), 0) if "S200_Slope" in d.columns else 0
    ema_ok = bool(price>ema21>ema50>sma200) if sma200>0 else bool(price>ema21>ema50)
    sepa   = bool(sma150>sma200>0 and sl200>0 and price>sma150)
    d21    = round(((price-ema21)/ema21)*100,1) if ema21>0 else NaN
    t_eps  = sf(info.get("trailingEps")); f_eps = sf(info.get("forwardEps"))

    # Fundamentals
    if sc_ok:
        eps_g = sf(sc.get("pat_g")); rev_g = sf(sc.get("rev_g"))
        opm   = sf(sc.get("opm"));   gm    = sf(sc.get("gm"))
        pm    = sf(sc.get("pm"));    roe   = sf(sc.get("roe"))
        roce  = sf(sc.get("roce"));  de    = sf(sc.get("de"))
        ocf_p = sf(sc.get("ocf_pat")); pe_c = sf(sc.get("pe"))
        inst  = sf(sc.get("inst"));  fii_chg = sf(sc.get("fii_chg"))
        promoter = sf(sc.get("promoter")); src = "sc"
    else:
        # yfinance: decimals for ratios (0.226 = 22.6%)
        eps_g = yf_to_pct(info.get("earningsGrowth"))
        rev_g = yf_to_pct(info.get("revenueGrowth"))
        opm   = yf_to_pct(info.get("operatingMargins"))
        gm    = yf_to_pct(info.get("grossMargins"))
        pm    = yf_to_pct(info.get("profitMargins"))
        roe   = yf_to_pct(info.get("returnOnEquity"))
        roa   = yf_to_pct(info.get("returnOnAssets"))
        roce  = round(sf(roa)*2,1) if ok(roa) else NaN
        # D/E: yfinance gives 25.4 meaning 0.254 for Indian stocks
        de_raw = sf(info.get("debtToEquity"))
        de = round(de_raw/100,2) if ok(de_raw) and de_raw>1 else de_raw
        # OCF/PAT: both in same currency
        ocf_raw = sf(info.get("operatingCashflow"))
        ni_raw  = sf(info.get("netIncomeToCommon"))
        ocf_p   = round(ocf_raw/ni_raw,2) if ok(ocf_raw) and ok(ni_raw) and ni_raw!=0 else NaN
        pe_c    = sf(info.get("trailingPE") or info.get("forwardPE"))
        # Institutional: yfinance gives 0.416 = 41.6%
        inst_r   = sf(info.get("heldPercentInstitutions"))
        inst     = round(inst_r*100,1) if ok(inst_r) and inst_r<=1 else (round(inst_r,1) if ok(inst_r) else NaN)
        prom_r   = sf(info.get("heldPercentInsiders"))
        promoter = round(prom_r*100,1) if ok(prom_r) and prom_r<=1 else (round(prom_r,1) if ok(prom_r) else NaN)
        fii_chg  = NaN; src = "yf"

    # Market cap
    mcap_raw = sf(info.get("marketCap"))
    if ok(mcap_raw):
        if is_india:
            mcap_cr   = round(mcap_raw/1e7, 0)
            mcap_disp = f"\u20b9{mcap_cr:,.0f} Cr"
            mcap_b    = round(mcap_raw/(84*1e9), 2)
        else:
            mcap_b    = round(mcap_raw/1e9, 2)
            mcap_disp = f"${mcap_b}B"
    else:
        mcap_b = NaN; mcap_disp = "—"

    # SCORING
    score = 0.0; sigs = []
    def add(lbl, detail, pts):
        nonlocal score; score += pts
        sigs.append((lbl, detail, pts, "pass" if pts>=1 else ("partial" if pts>0 else "fail")))
    def v(x): return sf(x, 0)

    add("EPS/PAT Growth",   fmt(eps_g,"%",0,"+") if ok(eps_g) else "—",
        1.0 if v(eps_g)>30 else (0.5 if v(eps_g)>15 else 0.0))
    add("Revenue / OPM",    f"Rev {fmt(rev_g,'%',0)} | OPM {fmt(opm,'%',0)}",
        1.0 if v(rev_g)>15 or v(opm)>15 else (0.5 if v(rev_g)>5 or v(opm)>8 else 0.0))
    add("RoE",              fmt(roe, "%"),
        1.0 if v(roe)>18 else (0.5 if v(roe)>12 else 0.0))
    add("RoCE",             fmt(roce, "%"),
        1.0 if v(roce)>15 else (0.5 if v(roce)>10 else 0.0))
    add("OCF / PAT",        fmt(ocf_p, "", 2),
        1.0 if v(ocf_p)>0.7 else (0.5 if v(ocf_p)>0.4 else 0.0))
    add("Debt / Equity",    fmt(de, "", 2),
        1.0 if ok0(de) and v(de)<0.3 else (0.5 if ok0(de) and v(de)<0.5 else 0.0))
    add("Institutional %",  fmt(inst, "%"),
        1.0 if v(inst)>30 else (0.5 if v(inst)>15 else 0.0))
    add("52W High",         fmt(hi_dist, "% away"),
        1.0 if ok(hi_dist) and hi_dist<=10 else (0.5 if ok(hi_dist) and hi_dist<=25 else 0.0))
    add("52W Low Gap",      f"+{fmt(lo_dist,'%',0)}",
        0.5 if v(lo_dist)>=30 else 0.0)
    rs = v(r3m)*0.4 + v(r6m)*0.6
    add("Rel Strength",     f"3M {fmt(r3m,'%',0)} / 6M {fmt(r6m,'%',0)}",
        1.0 if rs>40 else (0.5 if rs>20 else 0.0))
    add("Volume Spike",     f"{fmt(vol_r,'x',1)} on {fmt(dchg,'%',1,'+')}",
        1.0 if v(vol_r)>1.5 and v(dchg)>2 else (0.5 if v(vol_r)>1.2 and v(dchg)>0 else 0.0))
    add("EMA Stack",        "P>21>50>200 ✓" if ema_ok else ("P>21>50" if price>ema21>ema50 else "Misaligned"),
        1.0 if ema_ok else (0.5 if price>ema21>ema50 else 0.0))
    add("SEPA",             "Pass ✓" if sepa else ("P>200MA" if sma200>0 and price>sma200 else "Fail"),
        1.0 if sepa else (0.5 if sma200>0 and price>sma200 else 0.0))
    if ok(t_eps) and ok(f_eps) and t_eps>0 and f_eps>t_eps:
        accel = round(((f_eps-t_eps)/abs(t_eps))*100, 0)
        add("Earnings Accel", f"Fwd +{accel:.0f}% vs Trail", 1.0 if accel>10 else 0.5)
    elif v(eps_g)>30:
        add("Earnings Accel", f"Strong trend {fmt(eps_g,'%',0)}", 0.5)
    else:
        add("Earnings Accel", "No data", 0.0)
    add("RSI Zone",         fmt(rsi_v, "", 0),
        1.0 if ok(rsi_v) and 55<=rsi_v<=70 else (0.5 if ok(rsi_v) and 45<=rsi_v<=80 else 0.0))
    add("MCap Zone",        mcap_disp,
        1.0 if ok(mcap_b) and 0.3<=mcap_b<=10 else (0.5 if ok(mcap_b) and 10<mcap_b<=50 else 0.0))
    add("Revenue Leader",   fmt(rev_g, "%", 0, "+"),
        1.0 if v(rev_g)>25 else (0.5 if v(rev_g)>15 else 0.0))
    add("Margin Quality",   f"OPM {fmt(opm,'%',0)}",
        1.0 if v(opm)>25 or v(gm)>40 else (0.5 if v(opm)>15 or v(gm)>25 else 0.0))
    add("PAT Margin",       fmt(pm, "%"),
        1.0 if v(pm)>15 else (0.5 if v(pm)>8 else 0.0))
    add("Consolidation",    fmt(rng10, "% rng"),
        1.0 if ok(rng10) and rng10<8 else (0.5 if ok(rng10) and rng10<15 else 0.0))
    add("Entry Zone",       fmt(d21, "% EMA21", 1, "+") if ok(d21) else "—",
        1.0 if ok(d21) and 0<d21<=5 else (0.5 if ok(d21) and -3<d21<=10 else 0.0))

    score = round(score, 1)
    d21v = sf(d21, 99)
    if score>=16:
        act,acls = ("BUY NOW","buy") if d21v<=5 else (("BUY 50%","buy") if d21v<=10 else ("WAIT — PULLBACK","wait"))
    elif score>=12:
        act,acls = ("BUY ON DIP","buy") if d21v<=5 or v(rng10)<8 else ("WATCHLIST","watch")
    elif score>=8:  act,acls = "WATCHLIST","watch"
    else:           act,acls = "SKIP","skip"

    sl = round(price-2*atr_v,2) if atr_v>0 else NaN
    t1 = round(price*1.15,2); t2 = round(price*1.30,2)
    rr = round((t1-price)/max(price-sf(sl,price),0.01),1) if ok(sl) else NaN

    return {
        "ticker":ticker, "name":info.get("shortName") or info.get("longName") or ticker,
        "market":"INDIA" if is_india else "US", "cur":cur,
        "is_india":is_india, "sc_ok":sc_ok, "src":src,
        "sector":info.get("sector","—"), "industry":info.get("industry","—"),
        "score":score, "sigs":sigs, "action":act, "acls":acls,
        "price":round(price,2), "dchg":dchg, "vol_r":vol_r, "rsi":rsi_v,
        "hi_dist":hi_dist, "lo_dist":lo_dist, "r3m":r3m, "r6m":r6m, "r12m":r12m,
        "ema_ok":ema_ok, "sepa":sepa, "d21":d21, "rng10":rng10,
        "atr_pct":atr_pct, "ema21":round(ema21,2),
        "eps_g":eps_g, "rev_g":rev_g, "opm":opm, "gm":gm, "pm":pm,
        "roe":roe, "roce":roce, "de":de, "ocf_p":ocf_p, "inst":inst,
        "pe_c":pe_c, "mcap_disp":mcap_disp, "mcap_b":mcap_b,
        "fii_chg":fii_chg, "promoter":promoter,
        "sc":sc, "sl":sl, "t1":t1, "t2":t2, "rr":rr,
        "mults":multiples(price, pe_c, eps_g, opm),
        "_chart":d,
    }, None


# ── UI ──
st.markdown('<div class="app-title">⚡ Momentum Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">India → Screener.in + yfinance &nbsp;·&nbsp; US → yfinance &nbsp;·&nbsp; 21-pt score · multiples · PAT timeline</div>', unsafe_allow_html=True)

c1,c2 = st.columns([5,1])
with c1:
    ticker_input = st.text_input("t", label_visibility="collapsed",
        placeholder="BBOX.NS · HFCL.NS · ATHERENERG.NS · KIMS.NS · NVDA · PLTR")
with c2:
    go = st.button("Analyze ⚡", type="primary", use_container_width=True)

st.markdown('<p style="font-size:0.65rem;color:#bbb;margin-top:-8px;">India NSE → .NS suffix &nbsp;|&nbsp; US → ticker only</p>', unsafe_allow_html=True)

if go and ticker_input:
    with st.spinner(f"Analyzing {ticker_input.strip().upper()} …"):
        r, err = analyze(ticker_input)
    if err:
        st.error(f"⚠ {err}"); st.stop()

    sc = r["sc"]; cur = r["cur"]

    # Hero
    sc_cls = "elite" if r["score"]>=16 else ("strong" if r["score"]>=12 else ("moderate" if r["score"]>=8 else "weak"))
    mb = '<span class="bdg india">\U0001f1ee\U0001f1f3 NSE</span>' if r["is_india"] else '<span class="bdg us">\U0001f1fa\U0001f1f8 US</span>'
    ab = f'<span class="bdg {r["acls"]}">{r["action"]}</span>'
    dchg_col = "#16a34a" if sf(r["dchg"],0)>0 else "#dc2626"
    st.markdown(f"""
    <div class="hero">
      <div class="sring {sc_cls}"><span class="sn">{r['score']}</span><span class="sd">/ 21</span></div>
      <div class="hbody">
        <div class="hname">{r['name']}</div>
        <div class="hmeta">{r['ticker']} · {r['sector']} · {r['industry']}</div>
        {ab} {mb}
        <div style="font-size:0.68rem;color:#999;margin-top:6px;">
          {cur}{r['price']:,.2f} &nbsp;
          <span style="color:{dchg_col}">{fmt(r['dchg'],'%',1,'+')} today</span>
          &nbsp;·&nbsp; RSI {fmt(r['rsi'],"",0)}
          &nbsp;·&nbsp; Vol {fmt(r['vol_r'],'x',1)}
          &nbsp;·&nbsp; {fmt(r['hi_dist'],'% from 52W high')}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if r["is_india"] and not r["sc_ok"]:
        st.markdown('<div class="notice">⚠ Screener.in unavailable. Showing yfinance values — may be incomplete for Indian stocks. Works on local machine.</div>', unsafe_allow_html=True)

    # Multiples
    m = r["mults"]
    st.markdown(f'<div class="sec">Upside Multiples — 3-Year Horizon {stag("ca")}</div>', unsafe_allow_html=True)
    if ok(m.get("bear")):
        def tp(x): return f"{cur}{round(r['price']*x):,.0f}"
        pe_line = (f"PE: {m['pe']:.0f}x → Fair: {m['pe_f']:.0f}x → Bull: {m['pe_b']:.0f}x") if ok(m.get("pe")) else ""
        cagr_line = f"PAT CAGR: ~{m['cagr']:.0f}%/yr" + (f" + {m['mb']}% margin bonus" if m.get("mb") else "")
        st.markdown(f"""
        <div class="mbox">
          <div class="mbox-t">Return Scenarios (current = 1.0×)</div>
          <div class="mg4">
            <div class="mc bear"><div class="ml">Bear</div><div class="mx">{m['bear']:.1f}×</div><div class="mtp">{tp(m['bear'])}</div><div class="mlo">Growth miss + PE compression</div></div>
            <div class="mc cons"><div class="ml">Conservative</div><div class="mx">{m['cons']:.1f}×</div><div class="mtp">{tp(m['cons'])}</div><div class="mlo">Moderate growth, no re-rating</div></div>
            <div class="mc base"><div class="ml">Base Case</div><div class="mx">{m['base']:.1f}×</div><div class="mtp">{tp(m['base'])}</div><div class="mlo">Growth sustains + mild re-rating</div></div>
            <div class="mc bull"><div class="ml">Bull Case</div><div class="mx">{m['bull']:.1f}×</div><div class="mtp">{tp(m['bull'])}</div><div class="mlo">Acceleration + full PE expansion</div></div>
          </div>
          <div class="mfooter">{cagr_line}<br>{pe_line}<br><em>{m['note']}</em></div>
        </div>""", unsafe_allow_html=True)

    # PAT Timeline
    if r["sc_ok"] and sc.get("_pat") and sc.get("_sales"):
        yrs = sc.get("_years",[]); sal = sc.get("_sales",[]); pat = sc.get("_pat",[]); opms = sc.get("_opm",[])
        n = min(len(yrs),len(sal),len(pat))
        yrs,sal,pat = yrs[-n:],sal[-n:],pat[-n:]
        opms = opms[-n:] if opms else [None]*n
        mx = max([p for p in pat if p], default=1) or 1
        pm_str = ""
        if len(pat)>=2 and pat[0] and pat[-1] and pat[0]!=0:
            pm_str = f"&nbsp;·&nbsp; PAT <strong>{round(pat[-1]/pat[0],1)}×</strong> in {n} yrs"
        cells = ""
        for i,yr in enumerate(yrs):
            p=pat[i] if i<len(pat) else None; s=sal[i] if i<len(sal) else None; o=opms[i] if i<len(opms) else None
            bar=int((p/mx)*100) if p and mx else 0
            pcls=" hi" if i==len(yrs)-1 else ""
            cells += f'<div class="tl-col"><div class="tl-y">{yr}</div><div class="tl-s">{"\u20b9"+str(int(s))+"Cr" if s else "—"}</div><div class="tl-o">{"OPM "+str(int(o))+"%" if o else "—"}</div><div class="tl-bw"><div class="tl-b" style="width:{bar}%"></div></div><div class="tl-p{pcls}">{"\u20b9"+str(round(p,0))+"Cr" if p else "—"}</div></div>'
        st.markdown(f'<div class="sec">Financial History {stag("sc")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tl"><div style="font-size:0.65rem;color:#aaa;margin-bottom:8px;">Annual FY data{pm_str}</div><div class="tl-row">{cells}</div></div>', unsafe_allow_html=True)

    # Technicals
    st.markdown(f'<div class="sec">Technical Snapshot {stag("yf")}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="mg">
      {mc("Price",      f"{cur}{r['price']:,.2f}")}
      {mc("Day Change", fmt(r['dchg'],'%',1,'+'), "g" if sf(r['dchg'],0)>0 else "r")}
      {mc("Volume",     fmt(r['vol_r'],'x',1),      "g" if sf(r['vol_r'],0)>1.5 else "")}
      {mc("RSI",        fmt(r['rsi'],"",0),           "g" if ok(r['rsi']) and 55<=r['rsi']<=70 else ("r" if ok(r['rsi']) and r['rsi']>80 else ""))}
      {mc("52W High",   fmt(r['hi_dist'],'% away'),  cc(r['hi_dist'],0,25,True))}
      {mc("3M Return",  fmt(r['r3m'],'%',0,'+'),  "g" if sf(r['r3m'],0)>20 else ("r" if sf(r['r3m'],0)<0 else ""))}
      {mc("6M Return",  fmt(r['r6m'],'%',0,'+'),  "g" if sf(r['r6m'],0)>30 else ("r" if sf(r['r6m'],0)<0 else ""))}
      {mc("12M Return", fmt(r['r12m'],'%',0,'+'), "g" if sf(r['r12m'],0)>40 else ("" if not ok(r['r12m']) else ""))}
      {mc("EMA21 Dist", fmt(r['d21'],'%',1,'+') if ok(r['d21']) else "—", "g" if ok(r['d21']) and 0<r['d21']<=5 else ("r" if ok(r['d21']) and r['d21']>10 else ""))}
      {mc("10D Range",  fmt(r['rng10'],'%'),         "g" if ok(r['rng10']) and r['rng10']<8 else "")}
      {mc("EMA Stack",  "✓ Aligned" if r['ema_ok'] else "✗ No", "g" if r['ema_ok'] else "r")}
      {mc("SEPA",       "✓ Pass" if r['sepa'] else "✗ Fail",    "g" if r['sepa'] else "r")}
    </div>""", unsafe_allow_html=True)

    # Fundamentals
    extras = ""
    if r["is_india"] and ok(r.get("fii_chg")):
        extras += mc("FII Δ QoQ", fmt(r["fii_chg"],"%",1,"+"), "g" if sf(r["fii_chg"],0)>0 else "r")
    if ok(r.get("promoter")):
        extras += mc("Promoter", fmt(r["promoter"],"%"), "g" if sf(r["promoter"],0)>50 else "")
    st.markdown(f'<div class="sec">Fundamentals {stag(r['src'])}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="mg">
      {mc("EPS/PAT Gr", fmt(r['eps_g'],'%',0,'+') if ok(r['eps_g']) else "—", "g" if sf(r['eps_g'],0)>25 else ("r" if sf(r['eps_g'],0)<0 else ""))}
      {mc("Rev Growth", fmt(r['rev_g'],'%',0,'+') if ok(r['rev_g']) else "—", "g" if sf(r['rev_g'],0)>15 else "")}
      {mc("OPM",        fmt(r['opm'],'%',0),  "g" if sf(r['opm'],0)>15 else "")}
      {mc("PAT Margin", fmt(r['pm'],'%'),      "g" if sf(r['pm'],0)>15 else ("r" if sf(r['pm'],0)<0 else ""))}
      {mc("RoE",        fmt(r['roe'],'%'),     "g" if sf(r['roe'],0)>18 else ("r" if sf(r['roe'],0)<10 and ok(r['roe']) else ""))}
      {mc("RoCE",       fmt(r['roce'],'%'),    "g" if sf(r['roce'],0)>15 else "")}
      {mc("D/E",        fmt(r['de'],"",2),       cc(r['de'],0.3,0.5,True))}
      {mc("OCF/PAT",    fmt(r['ocf_p']),         "g" if sf(r['ocf_p'],0)>0.7 else ("r" if ok(r['ocf_p']) and sf(r['ocf_p'],0)<0.4 else ""))}
      {mc("Inst %",     fmt(r['inst'],'%'),    "g" if sf(r['inst'],0)>30 else "")}
      {mc("PE",         fmt(r['pe_c'],'x',0) if ok(r['pe_c']) else "—", "b")}
      {mc("MCap",       r['mcap_disp'])}
      {extras}
    </div>""", unsafe_allow_html=True)

    # Trade levels
    st.markdown(f'<div class="sec">Trade Levels {stag("ca")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tg">
      <div class="tc"><div class="tl">Stop Loss (2×ATR)</div><div class="tv r">{fmt(r['sl'],'',2,cur) if ok(r['sl']) else "—"}</div></div>
      <div class="tc"><div class="tl">Target 1 (+15%)</div><div class="tv g">{cur}{r['t1']:,.2f}</div></div>
      <div class="tc"><div class="tl">Target 2 (+30%)</div><div class="tv g">{cur}{r['t2']:,.2f}</div></div>
      <div class="tc"><div class="tl">Risk : Reward</div><div class="tv {"g" if ok(r['rr']) and sf(r['rr'],0)>=2 else ""}">{fmt(r['rr']),0,"—"}</div></div>
    </div>
    <p style="font-size:0.65rem;color:#aaa;margin-top:-2px;">
      EMA21: {cur}{r['ema21']:,.2f} &nbsp;·&nbsp; ATR: {fmt(r['atr_pct'],'%',2)}
    </p>""", unsafe_allow_html=True)

    # Signals
    st.markdown('<div class="sec">21-Point Signal Breakdown</div>', unsafe_allow_html=True)
    tmap = {0:"Tier 1 — Fundamentals (7pt)", 7:"Tier 2 — Momentum Triggers (8pt)", 15:"Tier 3 — Multibagger DNA (6pt)"}
    tsc = [0.0,0.0,0.0]; pills = ""
    for i,(lbl,detail,pts,status) in enumerate(r["sigs"]):
        if i in tmap: pills += f'<div class="tlbl">{tmap[i]}</div>'
        ti = 0 if i<7 else (1 if i<15 else 2); tsc[ti] += pts
        pills += f'<span class="pi {status}" title="{detail}">{lbl}: {detail} ({pts}pt)</span>'
    summ = " &nbsp;·&nbsp; ".join([f"T{i+1}: {tsc[i]:.1f}/{[7,8,6][i]}" for i in range(3)])
    st.markdown(f'<div style="font-size:0.66rem;color:#aaa;margin-bottom:5px;">{summ}</div><div class="srow">{pills}</div>', unsafe_allow_html=True)

    # Chart
    st.markdown(f'<div class="sec">Price Chart (1Y) {stag("yf")}</div>', unsafe_allow_html=True)
    ch = r["_chart"][["Close","EMA_21","EMA_50"]].copy()
    if "SMA_200" in r["_chart"].columns: ch["SMA_200"] = r["_chart"]["SMA_200"]
    ch.columns = [c.replace("_"," ") for c in ch.columns]
    st.line_chart(ch, height=270)
    st.markdown('<div class="sec">Volume</div>', unsafe_allow_html=True)
    st.bar_chart(r["_chart"][["Volume"]], height=110)

    # Score guide
    st.markdown("""<div class="sec">Score Guide</div><div class="mg">
      <div class="mce"><div class="ml">16–21</div><div class="mv g">Elite</div></div>
      <div class="mce"><div class="ml">12–15</div><div class="mv b">Strong</div></div>
      <div class="mce"><div class="ml">8–11</div><div class="mv a">Watch</div></div>
      <div class="mce"><div class="ml">Below 8</div><div class="mv r">Skip</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="disc">Educational tool only. Not financial advice. Always verify data on Screener.in before trading.</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;">
      <div style="font-size:2.5rem;margin-bottom:12px;">📊</div>
      <p style="font-size:0.86rem;color:#666;max-width:500px;margin:0 auto;line-height:1.75;">
        Score any stock on 21 momentum criteria.<br>
        Indian stocks pull real fundamentals from <strong>Screener.in</strong>.<br>
        Includes upside <strong>multiple scenarios</strong> and PAT growth timeline.
      </p>
      <div style="margin-top:18px;display:flex;flex-wrap:wrap;justify-content:center;gap:7px;">
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">BBOX.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">HFCL.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">ATHERENERG.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">KIMS.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">TATACONSUM.NS</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">NVDA</code>
        <code style="background:#f3f4f6;padding:4px 12px;border-radius:4px;font-size:0.77rem;">PLTR</code>
      </div>
    </div>""", unsafe_allow_html=True)
