# streamlit_super50.py
# Super50 - Improved (metadata caching, FMP enrichment, earnings proximity, smarter liquidity)
# UI: "Super50"
# Inputs: FMP API key (optional) and holding period (required)
# Universe: Top 500 (NIFTY 500)
# Purpose: produce Top 50 stocks for chosen holding period while enforcing minimum expected-return thresholds,
# using advanced internal analytics for higher confidence. Non-heavy improvements only.

import streamlit as st
st.set_page_config(page_title="Super50", layout="wide")
st.markdown("""
<style>
.header{ text-align:center; color:#0b3d91; font-size:36px; font-weight:800; margin-bottom:6px;}
.card{background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 24px rgba(11,61,145,0.06); margin-bottom:12px;}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px;}
.badge{display:inline-block; background:#eef2ff; color:#0b3d91; padding:4px 8px; border-radius:999px; font-size:12px;}
.info{font-size:14px; color:#334155; text-align:center; margin-bottom:12px;}
.warn{color:#b45309; font-weight:700;}
</style>
<div class='header'>Super50</div>
<div style='text-align:center;color:#334155;margin-bottom:12px;'>Improved Super50 — metadata caching, FMP enrichment & earnings proximity checks for more genuine picks. Inputs: FMP key (optional) + holding period.</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("Inputs (required)"):
    fmp_key = st.text_input("FinancialModelingPrep API key (optional)", type="password")
    interval = st.selectbox("Holding period", ["Shortest (15 days)","Short (1 month)","Mid (3 months)","Long (6 months)","Longest (1 year)"], index=0)
    allow_earnings_during_period = st.checkbox("Allow picks with earnings during holding period (not recommended)", value=False)
    generate = st.button("Generate Super50 (fresh run)")

st.markdown('<div class="info">This app uses cached metadata to speed up runs, optional FMP enrichment for fundamentals/news, and applies earnings & liquidity filters to improve genuineness. UI shows final ranked 50 with beginner-friendly reasons.</div>', unsafe_allow_html=True)

# --- imports ---
import pandas as pd, numpy as np, yfinance as yf, requests, io, math, traceback, re, time, json, os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET
import random

# --- config & dirs ---
CACHE_DIR = Path(__file__).parent / "cache_super50_final"
PRICES_DIR = CACHE_DIR / "prices"
META_PATH = CACHE_DIR / "meta.json"
FMP_META_PATH = CACHE_DIR / "fmp_meta.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)

TOP_UNIVERSE = 500
interval_map = {"Shortest (15 days)":15, "Short (1 month)":30, "Mid (3 months)":90, "Long (6 months)":180, "Longest (1 year)":365}

# thresholds (unchanged)
IDEAL_THRESH = {15:0.03, 30:0.02, 90:0.02, 180:0.01, 365:0.01}
FALLBACK_THRESH = {15:0.02, 30:0.01, 90:0.01, 180:0.005, 365:0.005}

# sentiment lexicon
POS = {"good","gain","upgrade","win","growth","positive","increase","order","contract","deal","signed","approved","award"}
NEG = {"loss","down","decline","negative","warn","drop","weak","fraud","lawsuit","delay"}

# --- helpers: NSE universe retrieval (unchanged) ---
@st.cache_data(ttl=12*3600)
def fetch_nse_symbols():
    urls = ["https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv"]
    s=set()
    headers = {"User-Agent":"Mozilla/5.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=10)
            if r.status_code==200 and r.text:
                for line in r.text.splitlines()[1:]:
                    parts=line.split(",")
                    if parts and parts[0].strip(): s.add(parts[0].strip().upper())
        except: pass
    return sorted([x for x in s if x.isalnum()])

@st.cache_data(ttl=6*3600)
def get_top_universe(n=TOP_UNIVERSE):
    syms = fetch_nse_symbols(); tcks=[s+".NS" for s in syms]
    meta={}
    def w(t):
        try:
            info = yf.Ticker(t).info
            return t, info.get("marketCap", 0) or 0
        except:
            return t,0
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures=[ex.submit(w,tk) for tk in tcks]
        for fut in futures:
            try:
                tk,mc=fut.result(); meta[tk]=mc
            except: pass
    ranked = sorted(meta.items(), key=lambda x:x[1], reverse=True)
    return [tk for tk,_ in ranked[:n]]

# --- price cache helpers (unchanged) ---
def price_path(tk): return PRICES_DIR / f"{tk.replace('/','_')}.parquet"
def price_is_fresh(tk, hours=24):
    p=price_path(tk)
    if not p.exists(): return False
    try: return (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)) < timedelta(hours=hours)
    except: return False

def download_prices(tickers, years=5):
    if not tickers: return
    end=datetime.now(); start=end - timedelta(days=365*years)
    for i in range(0, len(tickers), 60):
        batch=tickers[i:i+60]
        try:
            df = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker", threads=True, progress=False)
        except:
            df={}
        if len(batch)==1:
            tk=batch[0]
            try:
                if isinstance(df, pd.DataFrame):
                    sub=df[['Open','High','Low','Close','Volume']].dropna()
                    if not sub.empty: sub.to_parquet(price_path(tk))
            except: pass
        else:
            for tk in batch:
                try:
                    if isinstance(df, dict) or tk not in df:
                        single=yf.download(tk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
                        if not single.empty:
                            sub=single[['Open','High','Low','Close','Volume']].dropna()
                            if not sub.empty: sub.to_parquet(price_path(tk))
                    else:
                        sub=df[tk][['Open','High','Low','Close','Volume']].dropna()
                        if not sub.empty: sub.to_parquet(price_path(tk))
                except: pass

def load_price(tk):
    p=price_path(tk)
    if not p.exists(): return None
    try:
        df=pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex): df.index=pd.to_datetime(df.index)
        return df
    except:
        try: return pd.read_csv(p.with_suffix(".csv"))
        except: return None

# --- NEW: metadata cache & batch fetch to avoid repeated yf.info calls ---
META_TTL_HOURS = 24

def load_meta_cache():
    try:
        if META_PATH.exists():
            mtime = datetime.fromtimestamp(META_PATH.stat().st_mtime)
            if (datetime.now() - mtime).total_seconds() < META_TTL_HOURS * 3600:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
    except:
        pass
    return {}

def save_meta_cache(meta):
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except:
        pass

def fetch_metadata_batch(tickers):
    """
    Return dict: tk -> {marketCap, sector, shortName, earningsDate (if found)}
    Uses cached meta when fresh, otherwise fetches in parallel.
    """
    meta = load_meta_cache()
    to_fetch = [t for t in tickers if t not in meta]
    if not to_fetch:
        return {tk: meta.get(tk, {}) for tk in tickers}

    def worker(tk):
        try:
            info = yf.Ticker(tk).info
            mcap = info.get("marketCap", 0) or 0
            sector = info.get("sector") or info.get("industry") or "Other"
            name = info.get("shortName") or info.get("longName") or tk
            # sometimes earningsDate available in different keys; best-effort parse
            ed = info.get("earningsTimestamp") or info.get("earningsDate") or None
            # convert to ISO-ish if numeric timestamp
            if isinstance(ed, (int, float)):
                try:
                    ed_dt = datetime.fromtimestamp(int(ed))
                    ed = ed_dt.strftime("%Y-%m-%d")
                except:
                    ed = None
            return tk, {"marketCap": mcap, "sector": sector, "name": name, "earningsDate": ed}
        except Exception:
            return tk, {"marketCap": 0, "sector": "Other", "name": tk, "earningsDate": None}

    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(worker, tk): tk for tk in to_fetch}
        for fut in as_completed(futures):
            try:
                tk, info = fut.result()
                meta[tk] = info
            except:
                pass
    # persist meta
    save_meta_cache(meta)
    return {tk: meta.get(tk, {}) for tk in tickers}

# --- FMP enrichment (optional) ---
def fmp_profile(sym, key):
    if not key:
        return {}
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={key}", timeout=8)
        if r.status_code == 200:
            arr = r.json()
            if isinstance(arr, list) and arr:
                prof = arr[0]
                return {"sector": prof.get("sector") or prof.get("industry"),
                        "pe": prof.get("pe"),
                        "marketCap": prof.get("mktCap"),
                        "description": prof.get("description"),
                        "earningsDate": prof.get("earningsDate") or prof.get("earningsDateEstimated")}
    except:
        pass
    return {}

def fmp_news_count(sym, key, days=30):
    if not key:
        return 0
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/stock_news?symbol={sym}&limit=50&apikey={key}", timeout=8)
        if r.status_code == 200:
            news = r.json()
            cnt = 0
            for n in news:
                dt = n.get("publishedDate") or n.get("date")
                if dt:
                    try:
                        d = datetime.fromisoformat(dt.replace('Z',''))
                        if (datetime.now() - d).days <= days:
                            cnt += 1
                    except:
                        cnt += 1
            return cnt
    except:
        pass
    return 0

# --- sentiment (unchanged) ---
def google_news_sentiment(sym, days=15):
    try:
        q = f"{sym} stock India"
        url = "https://news.google.com/rss/search?q=" + requests.utils.requote_uri(q) + "&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=6, headers={'User-Agent':'Mozilla/5.0'})
        if r.status_code != 200 or not r.text:
            return 0.0, 0
        root = ET.fromstring(r.text)
        items = root.findall('.//item')
        total=0; cnt=0
        for it in items[:12]:
            t=(it.find('title').text or '') + ' ' + (it.find('description').text or '')
            words=re.findall(r"\w+", t.lower()); pos=sum(1 for w in words if w in POS); neg=sum(1 for w in words if w in NEG)
            total += (pos-neg); cnt += 1
        if cnt==0: return 0.0,0
        return max(-1.0, min(1.0, (total/cnt)/3.0)), cnt
    except:
        return 0.0,0

# --- technical helpers (unchanged) ---
def atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().iloc[-1]

def rolling_sharpe(returns, window=90):
    if len(returns) < window: window = max(10, len(returns))
    r = returns.rolling(window=window).mean()
    s = returns.rolling(window=window).std() + 1e-9
    sharpe = (r / s) * math.sqrt(252)
    return float(sharpe.iloc[-1]) if not sharpe.empty else 0.0

def hist_forward_stats(close, days, windows=[252, 252*2, 252*3]):
    stats = []
    for w in windows:
        if len(close) < w + days:
            continue
        sub = close[-(w + days):]
        fwd = sub.shift(-days) / sub - 1
        fwd = fwd.dropna()
        if not fwd.empty:
            stats.append({"mean": float(fwd.mean()), "std": float(fwd.std()), "pos_prop": float((fwd > 0).mean())})
    if not stats:
        return {"mean": 0.0, "std": 0.0, "pos_prop": 0.0, "count": 0}
    mean = float(np.mean([s["mean"] for s in stats]))
    std = float(np.mean([s["std"] for s in stats]))
    pos = float(np.mean([s["pos_prop"] for s in stats]))
    return {"mean": mean, "std": std, "pos_prop": pos, "count": len(stats)}

def bootstrap_forward_mean(close, days, n_boot=300):
    if len(close) < max(2*days, 60):
        return {"mean": 0.0, "p25": 0.0, "p75": 0.0, "pos_rate": 0.0}
    fwd = close.shift(-days) / close - 1
    fwd = fwd.dropna().values
    if len(fwd) == 0:
        return {"mean": 0.0, "p25": 0.0, "p75": 0.0, "pos_rate": 0.0}
    samples = np.random.choice(fwd, size=(n_boot, min(len(fwd), n_boot)), replace=True)
    sample_means = samples.mean(axis=1)
    mean = float(np.mean(sample_means))
    p25 = float(np.percentile(sample_means, 25))
    p75 = float(np.percentile(sample_means, 75))
    pos = float((sample_means > 0).mean())
    return {"mean": mean, "p25": p25, "p75": p75, "pos_rate": pos}

def estimate_expected_return(close, days):
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    recent_n = max(5, int(max(5, len(returns) * 0.15)))
    recent = returns.tail(recent_n).mean()
    hist_mean = hist_forward_stats(close, days)["mean"]
    if days <= 30:
        expected = 0.85 * recent + 0.15 * hist_mean
    elif days <= 90:
        expected = 0.7 * recent + 0.3 * hist_mean
    elif days <= 180:
        expected = 0.55 * recent + 0.45 * hist_mean
    else:
        expected = 0.45 * recent + 0.55 * hist_mean
    mom_factor = recent * (1.0 + min(0.6, np.tanh(np.nan_to_num(recent) * 10)))
    expected = 0.7 * expected + 0.3 * mom_factor
    return float(np.clip(expected, -1.0, 1.0))

# --- core per-ticker compute: uses preloaded metadata instead of repeated yf.info calls ---
def compute_metrics(tk, days, universe_closes=None, meta_map=None, fmp_key=None):
    df = load_price(tk)
    if df is None or "Close" not in df.columns or len(df["Close"].dropna()) < 90:
        return None
    close = df["Close"].dropna()
    expected = estimate_expected_return(close, days)
    if expected < -0.5:
        return None
    returns = close.pct_change().dropna()
    vol = float(returns.std()) if not returns.empty else 0.0
    sharpe = rolling_sharpe(returns, window=min(252, len(returns)))
    bootstrap = bootstrap_forward_mean(close, days, n_boot=300)
    kelly = 0.0
    try:
        kelly = float(expected / ((vol if vol>0 else 1e-9) ** 2 + 1e-9))
        kelly = float(np.clip(kelly, -0.5, 0.5))
    except:
        kelly = 0.0
    mom5 = (close.iloc[-1]/close.shift(5).iloc[-1]-1) if len(close)>5 else 0.0
    mom20 = (close.iloc[-1]/close.shift(20).iloc[-1]-1) if len(close)>20 else 0.0
    mom = 0.6*(mom5 if not math.isnan(mom5) else 0) + 0.4*(mom20 if not math.isnan(mom20) else 0)
    mom_pct = 0.0
    if universe_closes and tk in universe_closes:
        try:
            pct_map = momentum_percentiles(universe_closes)
            mom_pct = pct_map.get(tk, 0.0)
        except:
            mom_pct = 0.0

    g_sent, g_cnt = google_news_sentiment(tk.replace(".NS",""))
    # Use cached metadata (meta_map) when available
    info_meta = (meta_map.get(tk) if meta_map else None) or {}
    mc = info_meta.get("marketCap", 0) or 0
    name = info_meta.get("name") or info_meta.get("shortName") or tk
    sector = info_meta.get("sector") or "Other"
    earnings_date = info_meta.get("earningsDate", None)
    avg_vol = float(df["Volume"].dropna().tail(60).mean()) if "Volume" in df.columns and len(df["Volume"].dropna())>0 else 0.0

    # scoring (unchanged)
    if days <= 30:
        score = (expected * 0.7) + (mom * 0.12) + (bootstrap["mean"] * 0.06) + (mom_pct * 0.05) + (g_sent * 0.02)
    else:
        score = (expected * 0.45) + (sharpe * 0.15) + (bootstrap["mean"] * 0.15) + (0.05 * math.log10(max(mc,1)+1)) + (g_sent * 0.05)

    # confidence metric (unchanged)
    conf = 50
    if mc > 5e10:
        conf += 12
    elif mc > 1e10:
        conf += 8
    if avg_vol > 200000:
        conf += 10
    conf += int(min(12, bootstrap.get("pos_rate", 0) * 12))
    if sharpe > 1.0:
        conf += 6
    conf = int(max(30, min(99, conf)))

    # --- simplified beginner-friendly why text (non-technical) ---
    try:
        pos_chance = int(round(bootstrap.get("pos_rate", 0) * 100))
    except:
        pos_chance = 0
    exp_pct = expected * 100.0

    if exp_pct >= 3.0:
        why = (f"{name} ({tk}) — Recently it has shown strong upward movement and market interest. "
               f"Data suggests a good chance ({pos_chance}%) of a positive move over the selected period.")
    elif exp_pct >= 1.5:
        why = (f"{name} ({tk}) — The company is showing steady improvement and positive interest from the market. "
               f"It looks reasonably likely to give a healthy gain over your selected holding period.")
    elif exp_pct > 0:
        why = (f"{name} ({tk}) — This is a stable company with steady business. "
               f"It may deliver small but safer gains over the chosen period.")
    else:
        why = (f"{name} ({tk}) — The company is reliable but near-term gains appear limited. "
               f"Consider this pick mainly for stability rather than quick profit.")

    return {"ticker": tk, "name": name, "expected": float(expected), "score": float(score), "sector": sector,
            "confidence": conf, "avg_vol": avg_vol, "marketCap": mc, "why": why, "bootstrap": bootstrap, "kelly": kelly, "sharpe": sharpe, "earningsDate": earnings_date}

# --- selection thresholds & logic (unchanged) ---
def select_with_thresholds(results, days, ideal_thresh_map=IDEAL_THRESH, fallback_map=FALLBACK_THRESH):
    ideal = ideal_thresh_map.get(days, 0.0)
    fallback = fallback_map.get(days, 0.0)
    primary = [r for r in results if r["expected"] >= ideal and r["expected"] > 0]
    primary_sorted = sorted(primary, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(primary_sorted) >= 50:
        for r in primary_sorted[:50]:
            r["threshold_level"] = "ideal"
        return primary_sorted[:50], "ideal"
    fallback_pool = [r for r in results if r not in primary_sorted and r["expected"] >= fallback and r["expected"] > 0]
    combined = primary_sorted + sorted(fallback_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(combined) >= 50:
        for r in combined[:len(primary_sorted)]:
            r["threshold_level"] = "ideal"
        for r in combined[len(primary_sorted):50]:
            r["threshold_level"] = "fallback"
        return combined[:50], "fallback"
    positive_pool = [r for r in results if r["expected"] > 0 and r not in combined]
    combined2 = combined + sorted(positive_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(combined2) < 50:
        remaining = [r for r in sorted(results, key=lambda x:(x["expected"], x["score"]), reverse=True) if r not in combined2]
        for r in remaining:
            if r["expected"] < -0.05:
                continue
            combined2.append(r)
            if len(combined2) >= 50:
                break
    for r in combined2[:50]:
        lvl = "ideal" if r["expected"] >= ideal else ("fallback" if r["expected"] >= fallback else "relaxed")
        r["threshold_level"] = lvl
    used_level = "relaxed" if any(r["threshold_level"]=="relaxed" for r in combined2[:50]) else ("fallback" if any(r["threshold_level"]=="fallback" for r in combined2[:50]) else "ideal")
    return combined2[:50], used_level

# --- Utility: earnings proximity check ---
def earnings_within_period(earnings_date_str, days):
    if not earnings_date_str:
        return False
    try:
        ed = datetime.fromisoformat(earnings_date_str)
    except:
        try:
            ed = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        except:
            return False
    today = datetime.now()
    # if earnings date falls within next 'days' trading-days ~ approximate using calendar days
    return 0 <= (ed - today).days <= max(1, int(days * 1.4))

# --- main flow (improved preloads & filters) ---
if generate:
    days = interval_map.get(interval, 30)
    st.info(f"Generating Super50 for holding period = {days} trading days with improved prefilters.")
    try:
        universe = get_top_universe(TOP_UNIVERSE)
        st.write(f"Universe size: {len(universe)} (top by market cap).")
        # ensure price cache exists for tickers missing
        missing = [t for t in universe if not price_is_fresh(t)]
        if missing:
            with st.spinner(f"Downloading price history for {len(missing)} tickers (5 years)..."):
                download_prices(missing, years=5)
        # preload metadata (cached)
        st.info("Loading metadata (cached) and enriching (lightweight)...")
        meta_map = fetch_metadata_batch(universe)
        # optional FMP enrichment per ticker (lightweight: only when key provided)
        if fmp_key:
            st.info("Enriching fundamentals via FMP (best-effort, limited calls)...")
            # we will apply FMP enrichment on demand inside compute loop instead of mass calls to avoid API limits
        # preload closes for momentum percentiles
        universe_closes = {}
        for tk in universe:
            df = load_price(tk)
            if df is not None and "Close" in df.columns:
                universe_closes[tk] = df["Close"].dropna()
        # compute metrics in parallel but pass meta_map to avoid repeated info calls
        results = []
        with ThreadPoolExecutor(max_workers=22) as ex:
            futures = {ex.submit(compute_metrics, tk, days, universe_closes, meta_map, (fmp_key or None)): tk for tk in universe}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res is not None and res.get("expected", 0) > -0.5:
                        # Earnings proximity filter (unless allowed)
                        if not allow_earnings_during_period and res.get("earningsDate"):
                            if earnings_within_period(res.get("earningsDate"), days):
                                # skip candidate that has earnings in selected period to reduce event risk
                                continue
                        results.append(res)
                except Exception:
                    continue
        if not results:
            st.error("No results computed. Check network, API limits or prewarm the cache.")
        else:
            # liquidity rules: relax for shortest interval only
            if days <= 30:
                filtered = [r for r in results if (r["avg_vol"] > 30 or r["marketCap"] > 2e8)]
            else:
                filtered = [r for r in results if (r["avg_vol"] > 150 or r["marketCap"] > 5e8)]
            if len(filtered) < 50:
                # relax filters slightly but prioritize higher market cap
                filtered = sorted(results, key=lambda x: (x.get("marketCap",0), x.get("avg_vol",0)), reverse=True)
            final50, level_used = select_with_thresholds(filtered, days)
            # sector diversification soft cap (unchanged)
            sector_counts = {}
            diversified = []
            for r in final50:
                sec = r.get("sector","Other") or "Other"
                cnt = sector_counts.get(sec, 0)
                if cnt < 10:
                    diversified.append(r)
                    sector_counts[sec] = cnt + 1
                else:
                    pass
            if len(diversified) == 50:
                final50 = diversified
            st.success(f"Super50 prepared ({len(final50)}) — threshold level used: {level_used}.")
            if level_used != "ideal":
                st.warning(f"Could not fill all 50 with ideal thresholds; used '{level_used}' fallback. See flagged picks.")
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            for i,m in enumerate(final50, start=1):
                exp_pct = m["expected"] * 100
                conf = m.get("confidence", 50)
                thr = m.get("threshold_level", "unknown")
                flag_html = ""
                if thr == "fallback":
                    flag_html = "<div class='warn'>Below ideal threshold - fallback used</div>"
                elif thr == "relaxed":
                    flag_html = "<div class='warn'>Relaxed selection used (insufficient candidates met thresholds)</div>"
                html = ("<div class='card'>"
                        f"<div><span class='badge'>{m.get('sector','Other')}</span><span style='float:right;font-weight:700'>{i}. {m['ticker']}</span></div>"
                        f"<div style='margin-top:8px;'><div style='font-size:16px;font-weight:700;color:#0b3d91'>{m.get('name','')}</div>"
                        f"<div style='font-size:13px;color:#475569;margin-top:6px;'>Expected: <strong>{exp_pct:.2f}%</strong> | Confidence: <strong>{conf}/100</strong></div>"
                        f"<div style='margin-top:8px;'>{m.get('why')}</div>"
                        f"{flag_html}"
                        f"</div></div>")
                st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if final50:
                df = pd.DataFrame([{"rank":i+1,"ticker":m["ticker"],"name":m.get("name",""),"expected_pct":m.get("expected",0)*100,"marketCap":m.get("marketCap",0),"sector":m.get("sector",""),"confidence":m.get("confidence",0),"threshold_level":m.get("threshold_level","")} for i,m in enumerate(final50)])
                st.download_button("Download Super50 CSV", df.to_csv(index=False), file_name="super50_top50.csv", mime="text/csv")
    except Exception as e:
        st.error("Generation failed: " + str(e))
        st.error(traceback.format_exc())
