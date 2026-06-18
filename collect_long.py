#!/usr/bin/env python3
"""
Сбор ДЛИННОЙ истории акций (yfinance daily 2022+) для всех 37 тикеров.
Цель: проверить weekend-gap mean-reversion на ВСЕХ прошлых данных, включая bear-режим 2022.
Это РОДСТВЕННАЯ форма (вход на открытии пн, не в выходные через перп) — тест робастности феномена.
Выход: stock_long.csv (ticker, date, open, high, low, close).
"""
import yfinance as yf, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["AAPL","AMD","AMZN","ARM","BABA","BB","BIRD","BX","COIN","COST","CRCL","CRWV","DKNG",
           "EBAY","GME","GOOGL","HIMS","HOOD","INTC","LITE","LLY","META","MRVL","MSFT","MSTR","MU",
           "NFLX","NVDA","ORCL","PLTR","RIVN","RKLB","SNDK","TSLA","TSM","USAR","ZM"]
rows=[]
for t in TICKERS:
    try:
        d=yf.download(t,start="2022-01-01",interval="1d",progress=False,auto_adjust=False,threads=False)
        if d is None or d.empty:
            print(f"{t:6} пусто"); continue
        def col(n): return d[n][t] if (n,t) in d.columns else d[n]
        n=0
        for ts,o,h,l,c in zip(d.index,col("Open"),col("High"),col("Low"),col("Close")):
            rows.append((t,ts.date().isoformat(),float(o),float(h),float(l),float(c))); n+=1
        print(f"{t:6} {n} дней ({d.index[0].date()}..{d.index[-1].date()})")
    except Exception as e:
        print(f"{t:6} ERR {e}")
with open(os.path.join(HERE,"stock_long.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ticker","date","open","high","low","close"]); w.writerows(rows)
print(f"\nИТОГО {len(rows)} строк (ticker×день) -> stock_long.csv")
