#!/usr/bin/env python3
"""Проверка: для каждого equity-кандидата HL — есть ли перп-история + US-биржа (yfinance tz)."""
import requests, time
import yfinance as yf

CANDS = ["AAPL","AMD","AMZN","ARM","AVGO","BB","BX","COIN","COST","CRCL","CRWV","DELL","DKNG",
         "EBAY","GME","GOOGL","HIMS","HOOD","IBM","INTC","LITE","LLY","META","MRVL","MSFT","MSTR",
         "MU","NFLX","NOW","NVDA","ORCL","PLTR","RIVN","RKLB","SNDK","SPCX","TSLA","WDC","ZM",
         "ASML","BABA","TSM","NBIS","SMSN","SOFTBANK","HYUNDAI","KIOXIA","SKHX","BIRD","CBRS",
         "USAR","QNT","MINIMAX","PURRDAT"]
now = int(time.time()*1000)

def perp_bars(c):
    try:
        r = requests.post("https://api.hyperliquid.xyz/info", json={"type":"candleSnapshot",
            "req":{"coin":"xyz:"+c,"interval":"1h","startTime":now-300*86400000,"endTime":now}}, timeout=20).json()
        return len(r) if isinstance(r, list) else 0
    except Exception:
        return 0

print("%-8s%9s  %-22s%8s  %s" % ("TICKER","perp_1h","yf_tz","yf_days","verdict"))
us_ok = []
for c in CANDS:
    pb = perp_bars(c)
    tz = "-"; nd = 0
    try:
        fi = yf.Ticker(c).fast_info
        tz = str(fi.get("timezone") or fi.get("exchangeTimezoneName") or "-")
    except Exception:
        pass
    try:
        d = yf.download(c, start="2025-10-01", interval="1d", progress=False, auto_adjust=False, threads=False)
        nd = len(d) if d is not None else 0
    except Exception:
        pass
    us = (tz == "America/New_York") and pb >= 300 and nd >= 50
    if us:
        us_ok.append(c)
    print("%-8s%9d  %-22s%8d  %s" % (c, pb, tz, nd, "US-OK" if us else "skip"))
    time.sleep(0.1)

print("\nГОДНЫХ US-tradeable (perp>=300, yf tz=NY, daily>=50): %d" % len(us_ok))
print(sorted(us_ok))
