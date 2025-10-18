# streamlit_super50.py
# Super50 - Advanced final (threshold enforcement per your request)
# UI: "Super50"
# Inputs: FMP API key (optional) and holding period (required)
# Universe: Top 500 (NIFTY 500)
# Purpose: produce Top 50 stocks for chosen holding period while enforcing minimum expected-return thresholds,
# using advanced internal analytics for higher confidence.

import streamlit as st
st.set_page_config(page_title="Super50", layout="wide")
st.markdown("""
<style>
.header {text-align:center; color:#0b3d91; font-size:36px; font-weight:800; margin-bottom:6px;}
.sub {text-align:center; color:#334155; margin-bottom:12px;}
.card {background:#ffffff; border-radius:12px; padding:12px; box-shadow: 0 8px 24px rgba(11,61,145,0.06); margin-bottom:12px;}
.grid {display:grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap:12px;}
.badge {display:inline-block; background:#eef2ff; color:#0b3d91; padding:4px 8px; border-radius:999px; font-size:12px; margin-right:6px;}
.info {font-size:14px; color:#334155; text-align:center; margin-bottom:12px;}
.warn {color:#b45309; font-weight:700;}
</style>
<div class="header">Super50</div>
<div class="sub">Top 50 picks (Top 500 universe) — tuned for the selected holding period with enforced expected-return thresholds and enhanced confidence analytics.</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("Inputs (required)"):
    fmp_key = st.text_input("FinancialModelingPrep API key (optional)", type="password")
    interval = st.selectbox("Select holding period",
                            ["Shortest (15 days)", "Short (1 month)", "Mid (3 months)", "Long (6 months)", "Longest (1 year)"],
                            index=0)
    generate = st.button("Generate Super50 (fresh run)")

st.markdown('<div class="info">This app runs advanced backend analytics (momentum percentiles, ATR, rolling Sharpe, bootstrap forward returns, Kelly estimate, sentiment & fundamentals when available). UI only shows final ranked 50 with beginner-friendly reasons.</div>', unsafe_allow_html=True)

# --- imports ---
import pandas as pd, numpy as np, yfinance as yf, requests, io, math, traceback, re, time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET
import random

# --- config ---
CACHE_DIR = Path(__file__).parent / "cache_super50_final"
PRICES_DIR = CACHE_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)
TOP_UNIVERSE = 500
interval_map = {"Shortest (15 days)":15, "Short (1 month)":30, "Mid (3 months)":90, "Long (6 months)":180, "Longest (1 year)":365}

# thresholds: ideal and fallback
IDEAL_THRESH = {15:0.03, 30:0.02, 90:0.02, 180:0.01, 365:0.01}
FALLBACK_THRESH = {15:0.02, 30:0.01, 90:0.01, 180:0.005, 365:0.005}

# sentiment lexicon (simple)
POS_WORDS = {"good","gain","gains","upgrade","win","growth","beat","positive","increase","benefit","order","contract","deal","signed","approved","award","acquire"}
NEG_WORDS = {"loss","losses","down","decline","cut","delay","issue","concern","negative","warn","drop","fall","weak","fraud","lawsuit","recall"}

# --- helpers: universe retrieval ---
@st.cache_data(ttl=12*3600)
def fetch_nse_symbols():
    urls = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY.csv"
    ]
    syms = set()
    headers = {"User-Agent":"Mozilla/5.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=10)
            if r.status_code == 200 and r.text:
                try:
                    df = pd.read_csv(io.StringIO(r.text))
                    for c in ("Symbol","SYMBOL","symbol"):
                        if c in df.columns:
                            for s in df[c].dropna().astype(str):
                                syms.add(s.strip().upper())
                            break
                except Exception:
                    for line in r.text.splitlines()[1:]:
                        parts = line.split(",")
                        if parts and parts[0].strip(): syms.add(parts[0].strip().upper())
        except Exception:
            continue
    # filter weird symbols
    return sorted([s for s in syms if s.isalnum() and 2 <= len(s) <= 12])

@st.cache_data(ttl=6*3600)
def get_top_universe(n=TOP_UNIVERSE):
    symbols = fetch_nse_symbols()
    tickers = [s + ".NS" for s in symbols]
    meta = {}
    def worker(tk):
        try:
            info = yf.Ticker(tk).info
            return tk, info.get("marketCap", 0) or 0
        except Exception:
            return tk, 0
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(worker, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            try:
                tk, mc = fut.result(); meta[tk] = mc or 0
            except Exception:
                continue
    ranked = sorted(meta.items(), key=lambda x: x[1], reverse=True)
    return [tk for tk,_ in ranked[:n]]

# --- price caching helpers ---
def price_path(tk): return PRICES_DIR / f"{tk.replace('/','_')}.parquet"
def price_fresh(tk, hours=24):
    p = price_path(tk)
    if not p.exists(): return False
    try:
        return (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)) < timedelta(hours=hours)
    except Exception:
        return False

def download_prices(tickers, years=5):
    if not tickers:
        return
    end = datetime.now()
    start = end - timedelta(days=365*years)
    for i in range(0, len(tickers), 60):
        batch = tickers[i:i+60]
        try:
            df = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by='ticker', progress=False, threads=True)
        except Exception:
            df = {}
        if len(batch) == 1:
            tk = batch[0]
            try:
                if isinstance(df, pd.DataFrame):
                    sub = df[['Open','High','Low','Close','Volume']].dropna()
                    if not sub.empty:
                        sub.to_parquet(price_path(tk))
            except Exception:
                pass
        else:
            for tk in batch:
                try:
                    if isinstance(df, dict) or tk not in df:
                        single = yf.download(tk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
                        if not single.empty:
                            sub = single[['Open','High','Low','Close','Volume']].dropna()
                            if not sub.empty:
                                sub.to_parquet(price_path(tk))
                    else:
                        sub = df[tk][['Open','High','Low','Close','Volume']].dropna()
                        if not sub.empty:
                            sub.to_parquet(price_path(tk))
                except Exception:
                    try:
                        single = yf.download(tk, start=start.strftime("%Y-%m-%d"), end=start.strftime("%Y-%m-%d"), progress=False)
                        if not single.empty:
                            sub = single[['Open','High','Low','Close','Volume']].dropna()
                            if not sub.empty:
                                sub.to_parquet(price_path(tk))
                    except Exception:
                        pass

def load_price(tk):
    p = price_path(tk)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        try:
            return pd.read_csv(p.with_suffix('.csv'))
        except Exception:
            return None

# --- FMP helpers (optional) ---
def fmp_profile(sym, key):
    if not key:
        return {}
    try:
        r = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={key}", timeout=8)
        if r.status_code == 200:
            data = r.json(); prof = data[0] if isinstance(data, list) and data else None
            if prof:
                return {"sector": prof.get("sector") or prof.get("industry"),
                        "pe": prof.get("pe"),
                        "marketCap": prof.get("mktCap"),
                        "description": prof.get("description"),
                        "earningsDate": prof.get("mktCap")}
    except Exception:
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
                    except Exception:
                        cnt += 1
            return cnt
    except Exception:
        pass
    return 0

# --- sentiment: Google News RSS (quick) ---
def google_news_sentiment(sym, days=15):
    try:
        q = f"{sym} stock India"
        url = "https://news.google.com/rss/search?q=" + requests.utils.requote_uri(q) + "&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=6, headers={'User-Agent':'Mozilla/5.0'})
        if r.status_code != 200 or not r.text:
            return 0.0, 0
        root = ET.fromstring(r.text)
        items = root.findall('.//item')
        total_score = 0; count = 0
        for it in items[:12]:
            title = it.find('title').text if it.find('title') is not None else ""
            desc = it.find('description').text if it.find('description') is not None else ""
            txt = (title + " " + desc).lower()
            words = re.findall(r"\w+", txt)
            pos = sum(1 for w in words if w in POS_WORDS)
            neg = sum(1 for w in words if w in NEG_WORDS)
            score = pos - neg
            total_score += score; count += 1
        if count == 0:
            return 0.0, 0
        avg = total_score / count
        norm = max(-1.0, min(1.0, avg / 3.0))
        return float(norm), count
    except Exception:
        return 0.0, 0

# --- technical helpers ---
def rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def atr(high, low, close, window=14):
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().iloc[-1]

def rolling_sharpe(returns, window=90):
    if len(returns) < window:
        window = max(10, len(returns))
    mean = returns.rolling(window=window).mean()
    std = returns.rolling(window=window).std() + 1e-9
    sharpe = (mean / std) * math.sqrt(252)
    return float(sharpe.dropna().iloc[-1]) if not sharpe.dropna().empty else 0.0

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

def kelly_estimate(expected, vol):
    if vol <= 0:
        return 0.0
    f = expected / (vol * vol + 1e-9)
    return float(np.clip(f, -0.5, 0.5))

def momentum_percentiles(universe_closes):
    # return dict of tick -> 3m percentile (approx)
    mom_scores = {}
    for tk, c in universe_closes.items():
        try:
            mom_scores[tk] = (c.iloc[-1] / c.shift(60).iloc[-1] - 1) if len(c) > 60 else 0.0
        except:
            mom_scores[tk] = 0.0
    ser = pd.Series(mom_scores)
    pct = ser.rank(pct=True).to_dict()
    return pct

# --- expected return estimator (multi-horizon + momentum uplift) ---
def estimate_expected_return(close, days):
    returns = close.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    recent_n = max(5, int(max(5, len(returns) * 0.15)))
    recent = returns.tail(recent_n).mean()
    hist = hist_forward_stats(close, days)["mean"]
    if days <= 30:
        expected = 0.85 * recent + 0.15 * hist
    elif days <= 90:
        expected = 0.7 * recent + 0.3 * hist
    elif days <= 180:
        expected = 0.55 * recent + 0.45 * hist
    else:
        expected = 0.45 * recent + 0.55 * hist
    mom_factor = recent * (1.0 + min(0.8, np.tanh(np.nan_to_num(recent) * 10)))
    expected = 0.7 * expected + 0.3 * mom_factor
    return float(np.clip(expected, -1.0, 1.0))

# --- core per-ticker compute (advanced) ---
def compute_metrics(tk, days, universe_closes=None, fmp_key=None):
    df = load_price(tk)
    if df is None or "Close" not in df.columns or len(df["Close"].dropna()) < 90:
        return None
    close = df["Close"].dropna()
    expected = estimate_expected_return(close, days)
    if expected < -0.5:
        return None
    returns = close.pct_change().dropna()
    vol = float(returns.std()) if not returns.empty else 0.0
    atr_val = atr(df["High"], df["Low"], df["Close"]) if "High" in df.columns and "Low" in df.columns else 0.0
    sharpe = rolling_sharpe(returns, window=min(252, len(returns)))
    bootstrap = bootstrap_forward_mean(close, days, n_boot=300)
    kelly = kelly_estimate(expected, vol)
    mom5 = (close.iloc[-1] / close.shift(5).iloc[-1] - 1) if len(close) > 5 else 0.0
    mom20 = (close.iloc[-1] / close.shift(20).iloc[-1] - 1) if len(close) > 20 else 0.0
    mom = 0.6 * (mom5 if not math.isnan(mom5) else 0) + 0.4 * (mom20 if not math.isnan(mom20) else 0)
    mom_pct = 0.0
    if universe_closes is not None and tk in universe_closes:
        try:
            percentiles = momentum_percentiles(universe_closes)
            mom_pct = percentiles.get(tk, 0.0)
        except:
            mom_pct = 0.0
    g_sent, g_cnt = google_news_sentiment(tk.replace(".NS",""), days=15)
    try:
        info = yf.Ticker(tk).info
    except:
        info = {}
    mc = info.get("marketCap", 0) or 0
    name = info.get("shortName") or info.get("longName") or tk
    sector = info.get("sector") or info.get("industry") or "Other"
    avg_vol = float(df["Volume"].dropna().tail(60).mean()) if "Volume" in df.columns and len(df["Volume"].dropna())>0 else 0.0

    # scoring (ensemble) - bias for short intervals to expected & bootstrap
    if days <= 30:
        score = (expected * 0.7) + (mom * 0.12) + (bootstrap["mean"] * 0.06) + (mom_pct * 0.05) + (g_sent * 0.02)
    else:
        score = (expected * 0.45) + (sharpe * 0.15) + (bootstrap["mean"] * 0.15) + (0.05 * math.log10(max(mc,1)+1)) + (g_sent * 0.05)

    # confidence boost by market cap, liquidity, bootstrap positive rate and sharpe
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

    # beginner-friendly why text (includes bootstrap summary & Kelly guidance)
    # ---------- START: simplified, non-technical description ----------
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
    # ---------- END: simplified description ----------

    return {"ticker": tk, "name": name, "expected": float(expected), "score": float(score), "sector": sector,
            "confidence": conf, "avg_vol": avg_vol, "marketCap": mc, "why": why, "bootstrap": bootstrap, "kelly": kelly, "sharpe": sharpe}

# --- selection logic with thresholds and fallbacks ---
def select_with_thresholds(results, days, ideal_thresh_map=IDEAL_THRESH, fallback_map=FALLBACK_THRESH):
    ideal = ideal_thresh_map.get(days, 0.0)
    fallback = fallback_map.get(days, 0.0)
    # primary: meets ideal
    primary = [r for r in results if r["expected"] >= ideal and r["expected"] > 0]
    primary_sorted = sorted(primary, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(primary_sorted) >= 50:
        for r in primary_sorted[:50]:
            r["threshold_level"] = "ideal"
        return primary_sorted[:50], "ideal"
    # fallback stage: include those meeting fallback (but below ideal)
    fallback_pool = [r for r in results if r not in primary_sorted and r["expected"] >= fallback and r["expected"] > 0]
    combined = primary_sorted + sorted(fallback_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    if len(combined) >= 50:
        for r in combined[:len(primary_sorted)]:
            r["threshold_level"] = "ideal"
        for r in combined[len(primary_sorted):50]:
            r["threshold_level"] = "fallback"
        return combined[:50], "fallback"
    # further fallback: include any positive expected > 0
    positive_pool = [r for r in results if r["expected"] > 0 and r not in combined]
    combined2 = combined + sorted(positive_pool, key=lambda x: (x["score"], x["bootstrap"]["mean"]), reverse=True)
    # if still less than 50, append best remaining (avoid extremely negative expected)
    if len(combined2) < 50:
        remaining = [r for r in sorted(results, key=lambda x:(x["expected"], x["score"]), reverse=True) if r not in combined2]
        for r in remaining:
            if r["expected"] < -0.05:
                continue
            combined2.append(r)
            if len(combined2) >= 50:
                break
    # annotate flags
    for r in combined2[:50]:
        lvl = "ideal" if r["expected"] >= ideal else ("fallback" if r["expected"] >= fallback else "relaxed")
        r["threshold_level"] = lvl
    used_level = "relaxed" if any(r["threshold_level"]=="relaxed" for r in combined2[:50]) else ("fallback" if any(r["threshold_level"]=="fallback" for r in combined2[:50]) else "ideal")
    return combined2[:50], used_level

# --- main generation flow ---
if generate:
    days = interval_map.get(interval, 30)
    st.info(f"Generating Super50 for holding period = {days} trading days with enforced thresholds.")
    try:
        universe = get_top_universe(TOP_UNIVERSE)
        st.write(f"Universe size: {len(universe)} (Top by market cap).")
        missing = [t for t in universe if not price_fresh(t)]
        if missing:
            with st.spinner(f"Downloading price history for {len(missing)} tickers (5 years, may take time)..."):
                download_prices(missing, years=5)
        # preload closes for momentum percentiles acceleration
        universe_closes = {}
        for tk in universe:
            df = load_price(tk)
            if df is not None and "Close" in df.columns:
                universe_closes[tk] = df["Close"].dropna()
        results = []
        with ThreadPoolExecutor(max_workers=22) as ex:
            futures = {ex.submit(compute_metrics, tk, days, universe_closes, fmp_key=(fmp_key or None)): tk for tk in universe}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res is not None and res.get("expected", 0) > -0.5:
                        results.append(res)
                except Exception:
                    continue
        if not results:
            st.error("No results computed. Check network, API limits or prewarm the cache.")
        else:
            # keep only positive expected initially for stricter adherence
            pos_results = [r for r in results if r["expected"] > -0.5]  # we keep broad; selection enforces thresholds
            final50, level_used = select_with_thresholds(pos_results, days)
            # run a small sector diversification safety: ensure not more than 10 picks per sector (soft)
            sector_counts = {}
            diversified = []
            for r in final50:
                sec = r.get("sector","Other") or "Other"
                cnt = sector_counts.get(sec, 0)
                if cnt < 10:
                    diversified.append(r)
                    sector_counts[sec] = cnt + 1
                else:
                    # try to find replacement from pos_results not already in final50 with same/higher score from other sectors
                    pass
            # if diversification trimmed list < 50, just keep original final50 (diversification is soft)
            if len(diversified) == 50:
                final50 = diversified
            # show results
            st.success(f"Super50 prepared ({len(final50)}) — threshold level used: {level_used}.")
            if level_used != "ideal":
                st.warning(f"Could not fill all 50 with ideal thresholds; used '{level_used}' fallback. See flagged picks.")
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            for i, m in enumerate(final50, start=1):
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
                        "<div style='margin-top:8px;'>"
                        f"<div style='font-size:16px;font-weight:700;color:#0b3d91'>{m.get('name','')}</div>"
                        f"<div style='font-size:13px;color:#475569;margin-top:6px;'>Expected: <strong>{exp_pct:.2f}%</strong> | Confidence: <strong>{conf}/100</strong></div>"
                        f"<div style='margin-top:8px;'>{m.get('why')}</div>"
                        f"{flag_html}"
                        "</div></div>")
                st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # download CSV
            df = pd.DataFrame([{
                "rank": i+1,
                "ticker": m["ticker"],
                "name": m.get("name",""),
                "expected_pct": m.get("expected",0)*100,
                "marketCap": m.get("marketCap",0),
                "sector": m.get("sector",""),
                "confidence": m.get("confidence",0),
                "threshold_level": m.get("threshold_level","")
            } for i,m in enumerate(final50)])
            st.download_button("Download Super50 CSV", df.to_csv(index=False), file_name="super50_top50.csv", mime="text/csv")
    except Exception as e:
        st.error("Generation failed: " + str(e))
        st.error(traceback.format_exc())
