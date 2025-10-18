
# prewarm_cache.py - optional helper to pre-download price history for Top500
import yfinance as yf, requests
from pathlib import Path
from datetime import datetime, timedelta
OUT = Path(__file__).parent / 'cache_super50_final' / 'prices'; OUT.mkdir(parents=True, exist_ok=True)
def fetch_nse_symbols():
    urls=['https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv','https://archives.nseindia.com/content/equities/EQUITY_L.csv']
    s=set()
    for u in urls:
        try:
            r=requests.get(u, timeout=10)
            if r.status_code==200 and r.text:
                for line in r.text.splitlines()[1:]:
                    parts=line.split(',')
                    if parts and parts[0].strip(): s.add(parts[0].strip().upper())
        except: pass
    return sorted([x for x in s if x.isalnum()])[:500]
def price_file(tk): return OUT / f"{tk.replace('/','_')}.parquet"
def download_prices(tickers, years=3):
    end=datetime.now(); start=end - timedelta(days=365*years)
    for i in range(0,len(tickers),60):
        batch=tickers[i:i+60]
        try: df=yf.download(batch, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), group_by='ticker', progress=False)
        except: df={}
        for tk in batch:
            t=tk + '.NS'
            try:
                single=yf.download(t, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
                if not single.empty:
                    sub=single[['Open','High','Low','Close','Volume']].dropna()
                    if not sub.empty: sub.to_parquet(price_file(t))
            except: pass
if __name__=='__main__':
    ticks=fetch_nse_symbols(); print('Downloading', len(ticks), 'tickers...'); download_prices(ticks, years=3); print('Done.')
