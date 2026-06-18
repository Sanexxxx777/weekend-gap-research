#!/usr/bin/env python3
"""15m свечи перпов (последние ~50 дней) для правила '+10мин после открытия'. Только что доступно у HL."""
import requests, time, csv, os
HERE=os.path.dirname(os.path.abspath(__file__))
TICKERS=["AAPL","AMD","AMZN","ARM","BABA","BB","BIRD","BX","COIN","COST","CRCL","CRWV","DKNG",
         "EBAY","GME","GOOGL","HIMS","HOOD","INTC","LITE","LLY","META","MRVL","MSFT","MSTR","MU",
         "NFLX","NVDA","ORCL","PLTR","RIVN","RKLB","SNDK","TSLA","TSM","USAR","ZM"]
now=int(time.time()*1000)
rows=[]
for c in TICKERS:
    try:
        r=requests.post("https://api.hyperliquid.xyz/info",json={"type":"candleSnapshot",
          "req":{"coin":f"xyz:{c}","interval":"15m","startTime":now-50*86400000,"endTime":now}},timeout=25).json()
        if isinstance(r,list):
            for x in r: rows.append([c,int(x["t"]),int(x["T"]),float(x["o"]),float(x["h"]),float(x["l"]),float(x["c"])])
        time.sleep(0.1)
    except Exception as e: print(c,"ERR",e)
with open(os.path.join(HERE,"perp_15m.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ticker","t_open_ms","T_close_ms","open","high","low","close"]); w.writerows(rows)
print(f"perp_15m.csv: {len(rows)} строк, {len(set(r[0] for r in rows))} тикеров")
