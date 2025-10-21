# prewarm_cache.py
# Prewarm cache: historical prices + metadata (marketCap, sector, name, earningsDate)
# Usage: python prewarm_cache.py
#
# Produces:
#   ./cache_super50_final/prices/<TICKER>.parquet
#   ./cache_super50_final/meta.json

import time
import requests
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
OUT_DIR = Path(__file__).parent / "cache_super50_final"
PRICES_DIR = OUT_DIR / "prices"
META_PATH = OUT_DIR / "meta.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = 5                    # historical years to fetch
BATCH_SIZE = 60              # yfinance batch size
SLEEP_BETWEEN_BATCH = 2.0    # seconds between batches
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 4.0
TIMEOUT = 30

NSE_CSV_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY.csv"
]

def fetch_nse_symbols():
    headers = {"User-Agent": "Mozilla/5.0"}
    syms = set()
    for u in NSE_CSV_URLS:
        try:
            r = requests.get(u, headers=headers, timeout=10)
            if r.status_code == 200 and r.text:
                # try to parse csv lines
                for line in r.text.splitlines()[1:]:
                    parts = line.split(",")
                    if parts and parts[0].strip():
                        syms.add(parts[0].strip().upper())
        except Exception as e:
            print(f"Warning fetching {u}: {e}")
            continue
    cleaned = sorted([s for s in syms if s.isalnum() and 2 <= len(s) <= 12])
    print(f"Found {len(cleaned)} symbols from NSE sources.")
    return cleaned

def price_file_path(ticker):
    return PRICES_DIR / f"{ticker.replace('/','_')}.parquet"

def already_downloaded_price(ticker, max_age_hours=48):
    p = price_file_path(ticker)
    if not p.exists():
        return False
    try:
        age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600.0
        return age < max_age_hours
    except:
        return False

def load_meta():
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_meta(meta):
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Error saving meta:", e)

def fetch_marketcaps_and_meta(tickers):
    """
    Best-effort fetch using yfinance.info in parallel.
    Returns dict: ticker -> {marketCap, sector, name, earningsDate}
    """
    existing = load_meta()
    to_fetch = [t for t in tickers if t not in existing]
    meta = existing.copy()
    if not to_fetch:
        print("All metadata already cached.")
        return meta

    def worker(tk):
        try:
            info = yf.Ticker(tk).info
            mcap = info.get("marketCap", 0) or 0
            sector = info.get("sector") or info.get("industry") or "Other"
            name = info.get("shortName") or info.get("longName") or tk
            ed = info.get("earningsTimestamp") or info.get("earningsDate") or None
            if isinstance(ed, (int, float)):
                try:
                    ed_dt = datetime.fromtimestamp(int(ed))
                    ed = ed_dt.strftime("%Y-%m-%d")
                except:
                    ed = None
            return tk, {"marketCap": mcap, "sector": sector, "name": name, "earningsDate": ed}
        except Exception as e:
            return tk, {"marketCap": 0, "sector": "Other", "name": tk, "earningsDate": None}

    print(f"Fetching metadata for {len(to_fetch)} tickers (in parallel)...")
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(worker, tk): tk for tk in to_fetch}
        for fut in as_completed(futures):
            try:
                tk, info = fut.result()
                meta[tk] = info
            except Exception as e:
                tk = futures.get(fut, "unknown")
                meta[tk] = {"marketCap": 0, "sector": "Other", "name": tk, "earningsDate": None}
    save_meta(meta)
    print(f"Metadata saved to {META_PATH}")
    return meta

def download_batch_prices(batch, years=YEARS):
    end = datetime.now(); start = end - timedelta(days=365*years)
    start_str = start.strftime("%Y-%m-%d"); end_str = end.strftime("%Y-%m-%d")
    print(f"Downloading batch of {len(batch)} tickers...")
    try:
        df_all = yf.download(batch, start=start_str, end=end_str, group_by="ticker", threads=True, progress=False)
    except Exception as e:
        print("Batch download failed:", e)
        df_all = {}

    for tk in batch:
        path = price_file_path(tk)
        if already_downloaded_price(tk):
            print(f"  Skipping cached: {tk}")
            continue
        success = False
        try:
            if isinstance(df_all, dict) or tk not in df_all:
                df = yf.download(tk, start=start_str, end=end_str, progress=False)
            else:
                df = df_all[tk]
            if isinstance(df, pd.DataFrame) and not df.empty:
                df2 = df[['Open','High','Low','Close','Volume']].dropna()
                if not df2.empty:
                    df2.to_parquet(path)
                    print(f"  Saved {tk} ({len(df2)} rows).")
                    success = True
        except Exception as e:
            print(f"  Error saving {tk}: {e}")
        if not success:
            # retry single downloads
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    single = yf.download(tk, start=start_str, end=end_str, progress=False)
                    if isinstance(single, pd.DataFrame) and not single.empty:
                        single2 = single[['Open','High','Low','Close','Volume']].dropna()
                        if not single2.empty:
                            single2.to_parquet(path)
                            print(f"  [retry ok] Saved {tk} ({len(single2)} rows).")
                            success = True
                            break
                except Exception as e:
                    print(f"    Retry {attempt+1} failed for {tk}: {e}")
                time.sleep(RETRY_SLEEP)
            if not success:
                print(f"  Failed to download {tk} after retries.")

def main():
    print("=== Prewarm cache (prices + metadata) ===")
    symbols = fetch_nse_symbols()
    if not symbols:
        print("No symbols found; exiting.")
        return
    tickers_ns = [s + ".NS" for s in symbols]

    # Step 1: fetch metadata & market caps to rank universe
    print("Fetching market caps & metadata (best-effort)...")
    meta = fetch_marketcaps_and_meta(tickers_ns)

    # rank by market cap
    ranked = sorted(meta.items(), key=lambda x: x[1].get("marketCap", 0), reverse=True)
    top_n = 500 if len(ranked) >= 500 else len(ranked)
    universe = [tk for tk,_ in ranked[:top_n]]
    print(f"Universe size for prewarm: {len(universe)} (top by market cap).")

    # Step 2: prepare list of tickers to download prices for (skip recent caches)
    to_download = []
    for tk in universe:
        if not already_downloaded_price(tk):
            to_download.append(tk)
    print(f"Tickers to download/update: {len(to_download)}")

    # Step 3: download in batches
    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i:i+BATCH_SIZE]
        try:
            download_batch_prices(batch, years=YEARS)
        except Exception as e:
            print(f"Batch error: {e}")
            # fallback single downloads
            for tk in batch:
                try:
                    download_batch_prices([tk], years=YEARS)
                except Exception as e2:
                    print(f"  Single download failed for {tk}: {e2}")
        time.sleep(SLEEP_BETWEEN_BATCH)

    # Step 4: ensure metadata saved (may have been updated during fetch)
    save_meta(meta)
    print("Prewarm complete.")
    print("Files saved under:", str(OUT_DIR))
    print(f"Meta file: {META_PATH}")
    print("Run your Streamlit app (streamlit run streamlit_super50.py)")

if __name__ == "__main__":
    main()
