# prewarm_cache.py
# Helper to pre-download and cache historical price data used by the Super50 app.
# Usage: python prewarm_cache.py
#
# It will create (if not existing): ./cache_super50_final/prices/
# and save one parquet file per ticker (e.g. TCS.NS.parquet).
#
# Notes:
# - Requires: yfinance, pandas, pyarrow, requests
# - Downloads up to 5 years of daily data by default (configurable).
# - Designed to be robust (retries, batch downloads) and resume-friendly.

import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
OUT_DIR = Path(__file__).parent / "cache_super50_final" / "prices"
OUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = 5                      # how many years of historical data to fetch
BATCH_SIZE = 60                 # number of tickers to request in each yfinance batch
SLEEP_BETWEEN_BATCH = 2.0      # seconds pause between batches to be polite
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 4.0              # seconds before retry attempt
TIMEOUT = 30                   # request timeout for metadata/API calls

NSE_CSV_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY.csv"
]

def fetch_nse_symbols():
    """
    Fetch symbols from public NSE CSV endpoints and return a cleaned sorted list.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    syms = set()
    for u in NSE_CSV_URLS:
        try:
            r = requests.get(u, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200 and r.text:
                # Try to parse as CSV; fallback to line parsing
                try:
                    df = pd.read_csv(pd.io.common.StringIO(r.text))
                    for col in ("Symbol", "SYMBOL", "symbol"):
                        if col in df.columns:
                            for s in df[col].dropna().astype(str):
                                syms.add(s.strip().upper())
                            break
                except Exception:
                    for line in r.text.splitlines()[1:]:
                        parts = line.split(",")
                        if parts and parts[0].strip():
                            syms.add(parts[0].strip().upper())
        except Exception as e:
            print(f"Warning: couldn't fetch {u} -> {e}")
            continue
    # basic cleanup: keep alphanumeric tickers (no special chars) and reasonable length
    cleaned = sorted([s for s in syms if s.isalnum() and 2 <= len(s) <= 12])
    print(f"Fetched {len(cleaned)} unique NSE symbols (raw).")
    return cleaned

def price_file_path(ticker):
    """
    Local parquet path for a given ticker (ticker must include .NS)
    """
    return OUT_DIR / f"{ticker.replace('/','_')}.parquet"

def already_downloaded(ticker, max_age_hours=48):
    """
    Check if file exists and is recent enough (in hours).
    """
    p = price_file_path(ticker)
    if not p.exists(): 
        return False
    try:
        age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600.0
        return age < max_age_hours
    except Exception:
        return False

def download_batch(batch, years=YEARS):
    """
    Download a batch of tickers using yfinance. Save each ticker individually to parquet.
    """
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    tickers = batch[:]  # list of tickers with .NS suffix
    print(f"Downloading batch of {len(tickers)} tickers ({tickers[0]} ... {tickers[-1]})")
    try:
        # yfinance can download multiple tickers at once and returns a hierarchical DataFrame
        df_all = yf.download(tickers, start=start_str, end=end_str, group_by="ticker", threads=True, progress=False, timeout=TIMEOUT)
    except Exception as e:
        print(f"Batch-level download error: {e}. Falling back to single downloads.")
        df_all = {}

    # If df_all is a DataFrame for multiple tickers, df_all[ticker] will be available
    for tk in tickers:
        path = price_file_path(tk)
        # skip if recently downloaded
        if already_downloaded(tk):
            print(f"  Skipping {tk} — cached.")
            continue
        success = False
        # Try to extract the per-ticker DataFrame from df_all if possible
        try:
            if isinstance(df_all, dict) or tk not in df_all:
                # fallback: download single ticker
                single = yf.download(tk, start=start_str, end=end_str, progress=False, timeout=TIMEOUT)
                df = single
            else:
                # df_all[ticker] may exist
                df = df_all[tk]
            if isinstance(df, pd.DataFrame) and not df.empty:
                # keep only needed columns and drop rows with missing close
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                if not df.empty:
                    # write parquet
                    try:
                        df.to_parquet(path)
                        success = True
                        print(f"  Saved {tk} ({len(df)} rows).")
                    except Exception as e:
                        print(f"  Error saving {tk} -> {e}")
            else:
                print(f"  No data for {tk} (empty).")
        except Exception as e:
            print(f"  Error processing {tk} -> {e}")
        # retry single-downloads if initial attempt failed
        if not success:
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    single = yf.download(tk, start=start_str, end=end_str, progress=False, timeout=TIMEOUT)
                    if isinstance(single, pd.DataFrame) and not single.empty:
                        single = single[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                        if not single.empty:
                            single.to_parquet(path)
                            print(f"  [retry ok] Saved {tk} ({len(single)} rows).")
                            success = True
                            break
                except Exception as e:
                    print(f"    Retry {attempt+1} for {tk} failed -> {e}")
                time.sleep(RETRY_SLEEP)
            if not success:
                print(f"  Failed to download {tk} after retries — skipping for now.")
    # polite pause
    time.sleep(SLEEP_BETWEEN_BATCH)

def main():
    print("=== Super50 prewarm cache utility ===")
    symbols = fetch_nse_symbols()
    if not symbols:
        print("No symbols found — aborting.")
        return
    # Build the universe (top by market cap). We'll get .NS tickers and try to rank by market cap in a lightweight way.
    tickers_ns = [s + ".NS" for s in symbols]
    # attempt to get market caps (best-effort) and rank
    print("Fetching market caps for universe ranking (best-effort, this may take a while)...")
    market_caps = {}
    # do this in chunks to avoid too many API calls at once
    for i in range(0, len(tickers_ns), 40):
        batch = tickers_ns[i:i+40]
        for tk in batch:
            try:
                info = yf.Ticker(tk).info
                mc = info.get("marketCap", 0) or 0
                market_caps[tk] = mc
            except Exception:
                market_caps[tk] = 0
        time.sleep(0.6)
    ranked = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    top_n = 500 if len(ranked) >= 500 else len(ranked)
    universe = [tk for tk,_ in ranked[:top_n]]
    print(f"Universe size for prewarm: {len(universe)} (top by market cap).")

    # Build list of tickers that need download
    to_download = []
    for tk in universe:
        p = price_file_path(tk)
        if p.exists():
            # keep existing files but allow refresh only if older than 48 hours (configurable)
            age_hours = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600.0
            if age_hours > 48:
                to_download.append(tk)
            else:
                # skip recently cached
                continue
        else:
            to_download.append(tk)

    print(f"Tickers to download/update: {len(to_download)}")

    # Download in batches
    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i:i+BATCH_SIZE]
        try:
            download_batch(batch, years=YEARS)
        except Exception as e:
            print(f"Batch {i // BATCH_SIZE} failed with error: {e}")
            # attempt element-wise downloads for resilience
            for tk in batch:
                try:
                    download_batch([tk], years=YEARS)
                except Exception as e2:
                    print(f"  Single download for {tk} failed: {e2}")

    print("Prewarm complete. Cached files are in:", str(OUT_DIR))
    print("Run the Streamlit app (streamlit run streamlit_super50.py).")

if __name__ == "__main__":
    main()
