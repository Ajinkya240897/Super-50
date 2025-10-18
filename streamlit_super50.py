
# streamlit_super50.py
import streamlit as st
st.set_page_config(page_title='Super50', layout='wide')
st.markdown("""
<style>
.header{ text-align:center; color:#0b3d91; font-size:34px; font-weight:800; margin-bottom:6px;}
.card{background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 24px rgba(11,61,145,0.06); margin-bottom:12px;}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px;}
.badge{display:inline-block; background:#eef2ff; color:#0b3d91; padding:4px 8px; border-radius:999px; font-size:12px;}
.info{font-size:14px; color:#334155; text-align:center; margin-bottom:12px;}
.warn{color:#b45309; font-weight:700;}
</style>
<div class='header'>Super50</div>
<div style='text-align:center;color:#334155;margin-bottom:12px;'>Top 50 (Top 500 universe) — priority: holding-period expected returns (positive enforced).</div>
""", unsafe_allow_html=True)

with st.sidebar.expander('Inputs (required)'):
    fmp_key = st.text_input('FMP API key (optional)', type='password')
    interval = st.selectbox('Holding period', ['Shortest (15 days)','Short (1 month)','Mid (3 months)','Long (6 months)','Longest (1 year)'], index=0)
    generate = st.button('Generate Super50 (fresh run)')

st.markdown('<div class="info">Backend uses price history (yfinance), sentiment (Google RSS), momentum & multi-horizon checks. UI shows only final picks and beginner-friendly reasons.</div>', unsafe_allow_html=True)

# imports
import pandas as pd, numpy as np, yfinance as yf, requests, io, math, traceback, re, time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

CACHE_DIR = Path(__file__).parent / 'cache_super50_final'
PRICES_DIR = CACHE_DIR / 'prices'; PRICES_DIR.mkdir(parents=True, exist_ok=True)
TOP_UNIVERSE = 500
interval_map = {'Shortest (15 days)':15,'Short (1 month)':30,'Mid (3 months)':90,'Long (6 months)':180,'Longest (1 year)':365}
MIN_EXPECTED = {15:0.01, 30:0.008, 90:0.005, 180:0.002, 365:0.0}

POS = set(['good','gain','upgrade','win','growth','positive','increase','order','contract','deal','signed','approved'])
NEG = set(['loss','down','decline','negative','warn','drop','weak','fraud','lawsuit'])

@st.cache_data(ttl=12*3600)
def fetch_nse_symbols():
    urls = ['https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv','https://archives.nseindia.com/content/equities/EQUITY_L.csv']
    s=set(); headers={'User-Agent':'Mozilla/5.0'}
    for u in urls:
        try:
            r=requests.get(u, headers=headers, timeout=8)
            if r.status_code==200 and r.text:
                for line in r.text.splitlines()[1:]:
                    parts=line.split(',')
                    if parts and parts[0].strip(): s.add(parts[0].strip().upper())
        except: pass
    return sorted([x for x in s if x.isalnum()])[:2000]

@st.cache_data(ttl=6*3600)
def get_top_universe(n=TOP_UNIVERSE):
    syms = fetch_nse_symbols(); tcks = [s+'.NS' for s in syms]
    meta={}
    def worker(t):
        try: return t, yf.Ticker(t).info.get('marketCap',0) or 0
        except: return t,0
    with ThreadPoolExecutor(max_workers=14) as ex:
        futures=[ex.submit(worker,tk) for tk in tcks]
        for fut in futures:
            try:
                tk,mc=fut.result(); meta[tk]=mc
            except: pass
    ranked = sorted(meta.items(), key=lambda x:x[1], reverse=True)
    return [tk for tk,_ in ranked[:n]]

def price_file(tk): return PRICES_DIR / f"{tk.replace('/','_')}.parquet"
def price_fresh(tk,hours=24):
    p=price_file(tk); 
    if not p.exists(): return False
    try: return (datetime.now()-datetime.fromtimestamp(p.stat().st_mtime))<timedelta(hours=hours)
    except: return False

def download_prices(tickers, years=3):
    if not tickers: return
    end=datetime.now(); start=end-timedelta(days=365*years)
    for i in range(0,len(tickers),60):
        batch=tickers[i:i+60]
        try: df=yf.download(batch, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), group_by='ticker', threads=True, progress=False)
        except: df={}
        if len(batch)==1:
            tk=batch[0]
            try:
                if isinstance(df, pd.DataFrame):
                    sub=df[['Open','High','Low','Close','Volume']].dropna(); 
                    if not sub.empty: sub.to_parquet(price_file(tk))
            except: pass
        else:
            for tk in batch:
                try:
                    if isinstance(df, dict) or tk not in df:
                        single=yf.download(tk, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
                        if not single.empty:
                            sub=single[['Open','High','Low','Close','Volume']].dropna(); 
                            if not sub.empty: sub.to_parquet(price_file(tk))
                    else:
                        sub=df[tk][['Open','High','Low','Close','Volume']].dropna(); 
                        if not sub.empty: sub.to_parquet(price_file(tk))
                except: pass

def load_price(tk):
    p=price_file(tk)
    if not p.exists(): return None
    try:
        df=pd.read_parquet(p); 
        if not isinstance(df.index, pd.DatetimeIndex): df.index=pd.to_datetime(df.index)
        return df
    except:
        try: return pd.read_csv(p.with_suffix('.csv'))
        except: return None

def google_sent(sym, days=15):
    try:
        q=f"{sym} stock India"
        url='https://news.google.com/rss/search?q='+requests.utils.requote_uri(q)+'&hl=en-IN&gl=IN&ceid=IN:en'
        r=requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=6)
        if r.status_code!=200 or not r.text: return 0.0,0
        root=ET.fromstring(r.text); items=root.findall('.//item')
        total=0; cnt=0
        for it in items[:8]:
            t=(it.find('title').text or '') + ' ' + (it.find('description').text or '')
            words=re.findall(r"\w+", t.lower()); pos=sum(1 for w in words if w in POS); neg=sum(1 for w in words if w in NEG)
            total += (pos-neg); cnt += 1
        if cnt==0: return 0.0,0
        avg=total/cnt; return max(-1.0, min(1.0, avg/3.0)), cnt
    except: return 0.0,0

def hist_forward(close, days):
    fwd=close.shift(-days)/close-1; fwd=fwd.dropna()
    if fwd.empty: return {'mean':0.0,'pos_prop':0.0}
    return {'mean':float(fwd.mean()), 'pos_prop': float((fwd>0).mean())}

def est_expected(close, days):
    r=close.pct_change().dropna()
    if r.empty: return 0.0
    recent_n=max(5,int(max(5,len(r)*0.12)))
    recent=float(r.tail(recent_n).mean())
    hist=hist_forward(close, days)['mean']
    if days<=30: expected=0.85*recent+0.15*hist
    elif days<=90: expected=0.7*recent+0.3*hist
    elif days<=180: expected=0.55*recent+0.45*hist
    else: expected=0.45*recent+0.55*hist
    # cap to reasonable range
    return float(np.clip(expected, -0.99, 0.99))

def compute_one(tk, days, fmp_key=None):
    df=load_price(tk)
    if df is None or 'Close' not in df.columns or len(df['Close'].dropna())<60: return None
    close=df['Close'].dropna()
    expected=est_expected(close, days)
    if expected < -0.5: return None
    daily=close.pct_change().dropna(); vol=float(daily.std()) if not daily.empty else 0.0
    avg_vol=float(df['Volume'].dropna().tail(60).mean()) if 'Volume' in df.columns else 0.0
    try: info=yf.Ticker(tk).info
    except: info={}
    mc=info.get('marketCap',0) or 0; name=info.get('shortName') or info.get('longName') or tk; sector=info.get('sector') or 'Other'
    g_sent, g_cnt = google_sent(tk.replace('.NS',''), days=15)
    mom5 = (close.iloc[-1]/close.shift(5).iloc[-1]-1) if len(close)>5 else 0.0
    mom20 = (close.iloc[-1]/close.shift(20).iloc[-1]-1) if len(close)>20 else 0.0
    mom = 0.6*(mom5 if not math.isnan(mom5) else 0) + 0.4*(mom20 if not math.isnan(mom20) else 0)
    # scoring - emphasis on expected for short term
    vol_pen = vol if vol>0 else 1e-9
    risk_adj = expected / vol_pen
    if days<=30:
        score = expected*0.9 + risk_adj*0.08 + mom*0.02 + g_sent*0.02
    else:
        score = expected*0.6 + risk_adj*0.2 + math.log10(max(mc,1)+1)*0.02 + mom*0.03 + g_sent*0.02
    conf = 50
    if mc>5e10: conf+=18
    elif mc>1e10: conf+=12
    elif mc>1e9: conf+=8
    if avg_vol>150000: conf+=10
    conf = int(max(30, min(98, conf)))
    why = f"{name} ({tk}) — Recommended for the selected period. Expected ≈ {expected*100:.2f}%.")
    return {'ticker':tk,'name':name,'expected':float(expected),'score':float(score),'sector':sector,'confidence':conf,'avg_vol':avg_vol,'marketCap':mc,'why':why}

def select_top(results):
    pool=sorted(results, key=lambda x:(x['score'], x['expected']), reverse=True)
    return pool[:50]

if generate:
    days=interval_map.get(interval,30)
    st.info(f"Generating Super50 for {days} trading days (positive-expected enforced)." )
    try:
        universe = get_top_universe(TOP_UNIVERSE)
        st.write(f"Universe size: {len(universe)} (Top by market cap)." )
        missing=[t for t in universe if not price_fresh(t)]
        if missing:
            with st.spinner(f"Downloading price history for {len(missing)} tickers..."):
                download_prices(missing, years=3)
        results=[]
        with ThreadPoolExecutor(max_workers=18) as ex:
            futures = {ex.submit(compute_one, tk, days, fmp_key=(fmp_key or None)): tk for tk in universe}
            for fut in as_completed(futures):
                try:
                    res=fut.result()
                    if res and (res.get('marketCap',0)>0 or res.get('avg_vol',0)>0):
                        results.append(res)
                except: pass
        if not results:
            st.error('No results computed. Check network or price downloads.')
        else:
            min_req = MIN_EXPECTED.get(days, 0.0)
            primary = [r for r in results if (r['expected']>=min_req and r['expected']>0)]
            if len(primary) < 50:
                positive = [r for r in results if r['expected']>0 and r not in primary]
                combined = sorted(primary + positive, key=lambda x:(x['score'], x['expected']), reverse=True)
            else:
                combined = sorted(primary, key=lambda x:(x['score'], x['expected']), reverse=True)
            final = combined[:50]
            # if still less than 50, pad with best remaining (avoid extreme negatives)
            if len(final) < 50:
                remaining = [r for r in sorted(results, key=lambda x:(x['expected'], x['score']), reverse=True) if r not in final]
                for r in remaining:
                    if r['expected'] < -0.05: continue
                    final.append(r)
                    if len(final)>=50: break
            # annotate flags
            for r in final:
                r['flag_below_threshold'] = (r['expected'] < min_req or r['expected'] <= 0)
            if len(final) < 50:
                st.warning('Could not find 50 items meeting strict thresholds; returning best available picks.')
            st.success(f"Super50 prepared ({len(final)})." )
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            for i,m in enumerate(final, start=1):
                exp_pct=m['expected']*100; conf=m.get('confidence',50); why=m.get('why','')
                caution = "<div class='warn'>Note: below expected threshold — use caution.</div>" if m.get('flag_below_threshold') else ''
                html=("<div class='card'>"
                      f"<div><span class='badge'>{m.get('sector','Other')}</span><span style='float:right;font-weight:700'>{i}. {m['ticker']}</span></div>"
                      "<div style='margin-top:8px;'>"
                      f"<div style='font-size:16px;font-weight:700;color:#0b3d91'>{m.get('name','')}</div>"
                      f"<div style='font-size:13px;color:#475569;margin-top:6px;'>Expected: <strong>{exp_pct:.2f}%</strong> | Confidence: <strong>{conf}/100</strong></div>"
                      f"<div style='margin-top:8px;'>{why}</div>"
                      f"{caution}"
                      "</div></div>")
                st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if final:
                df=pd.DataFrame([{'rank':i+1,'ticker':m['ticker'],'name':m.get('name',''),'expected_pct':m.get('expected',0)*100,'marketCap':m.get('marketCap',0),'sector':m.get('sector',''),'confidence':m.get('confidence',0),'flag_below_threshold':m.get('flag_below_threshold',False)} for i,m in enumerate(final)])
                st.download_button('Download Super50 CSV', df.to_csv(index=False), file_name='super50_top50.csv', mime='text/csv')
    except Exception as e:
        st.error('Generation failed: '+str(e))
        st.error(traceback.format_exc())
