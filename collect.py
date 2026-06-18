#!/usr/bin/env python3
"""
Сборщик ground-truth датасета для weekend-gap исследования (HL stock-перпы).
Honest: перп в любой момент = close ПОСЛЕДНЕЙ ЗАВЕРШЁННОЙ 1h свечи (T<=target), без look-ahead.
Выход: weekend_panel.csv (главное), perp_hourly.csv, stock_daily.csv.
Запуск на сервере (нужен yfinance). ET-моменты конвертируются с учётом DST через zoneinfo.
"""
import json, datetime, time, sys, csv, os
from zoneinfo import ZoneInfo
import requests
import yfinance as yf

UTC = ZoneInfo("UTC"); NY = ZoneInfo("America/New_York")
HL = "https://api.hyperliquid.xyz/info"
OUT = os.path.dirname(os.path.abspath(__file__))

# US-listed equities с регулярным US-расписанием (гигиена вселенной, НЕ отбор по винрейту).
# Иностранные (SMSN/SOFTBANK/HYUNDAI/KIOXIA), индексы (SP500/NIFTY/VIX), HL-нативное — исключены.
# полный US-tradeable набор HL (perp>=300 свечей, yf tz=America/New_York, daily>=50) — проверено probe_universe.py
EQUITIES = ["AAPL","AMD","AMZN","ARM","BABA","BB","BIRD","BX","COIN","COST","CRCL","CRWV","DKNG",
            "EBAY","GME","GOOGL","HIMS","HOOD","INTC","LITE","LLY","META","MRVL","MSFT","MSTR","MU",
            "NFLX","NVDA","ORCL","PLTR","RIVN","RKLB","SNDK","TSLA","TSM","USAR","ZM"]
EQUITIES = sorted(set(EQUITIES))
MARKET_PERP = "SP500"   # xyz:SP500 — market proxy, торгуется в выходные

def fetch_perp_1h(coin):
    """Все 1h свечи перпа за всю историю (один запрос, ~5000 баров хватает на ~200д)."""
    now = int(time.time()*1000)
    req = {"type":"candleSnapshot","req":{"coin":f"xyz:{coin}","interval":"1h",
           "startTime": now-300*86400000, "endTime": now}}
    for attempt in range(3):
        try:
            r = requests.post(HL, json=req, timeout=30).json()
            if isinstance(r, list):
                # (t_open, T_close, open, high, low, close)
                return [(int(c["t"]), int(c["T"]), float(c["o"]), float(c["h"]),
                         float(c["l"]), float(c["c"])) for c in r]
        except Exception as e:
            time.sleep(1.5)
    return []

def perp_at(candles, target_ms):
    """Honest: close последней свечи, ПОЛНОСТЬЮ завершённой к target (T<=target)."""
    best = None
    for t,T,o,h,l,c in candles:
        if T <= target_ms:
            best = c
        else:
            break
    return best

def et_to_ms(date, hh, mm):
    """ET-время заданной даты -> UTC ms (DST автоматически)."""
    dt = datetime.datetime(date.year, date.month, date.day, hh, mm, tzinfo=NY)
    return int(dt.timestamp()*1000)

def main():
    print("=== Сбор перпов (1h) ===", flush=True)
    perp = {}
    for c in EQUITIES + [MARKET_PERP]:
        cd = fetch_perp_1h(c)
        perp[c] = cd
        print(f"  {c:7} {len(cd)} свечей" + ("" if cd else "  <<< ПУСТО"), flush=True)
        time.sleep(0.15)

    print("\n=== Сбор акций (yfinance daily) ===", flush=True)
    stock = {}
    for c in EQUITIES:
        try:
            d = yf.download(c, start="2025-10-01", interval="1d", progress=False,
                            auto_adjust=False, threads=False)
            if d is None or d.empty:
                print(f"  {c:7} ПУСТО (нет в yfinance)", flush=True); continue
            def col(name):
                return d[name][c] if (name,c) in d.columns else d[name]
            rows = []
            for ts, o, h, l, cl in zip(d.index, col("Open"), col("High"), col("Low"), col("Close")):
                rows.append((ts.date(), float(o), float(h), float(l), float(cl)))
            stock[c] = rows
            print(f"  {c:7} {len(rows)} дней", flush=True)
        except Exception as e:
            print(f"  {c:7} ERR {e}", flush=True)

    # --- сырьё на диск ---
    with open(os.path.join(OUT,"perp_hourly.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ticker","t_open_ms","T_close_ms","open","high","low","close"])
        for c,cd in perp.items():
            for row in cd: w.writerow([c]+list(row))
    with open(os.path.join(OUT,"stock_daily.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["ticker","date","open","high","low","close"])
        for c,rows in stock.items():
            for d,o,h,l,cl in rows: w.writerow([c,d.isoformat(),o,h,l,cl])

    # --- weekend-панель ---
    # перп-точки в ET (decision = sun20/mon0630; targets = mon open/close из акции)
    PERP_POINTS = {"fri1600":("fri",16,0),"sat1200":("sat",12,0),"sun1200":("sun",12,0),
                   "sun2000":("sun",20,0),"mon0000":("mon",0,0),"mon0600":("mon",6,0),
                   "mon0630":("mon",6,30),"mon0800":("mon",8,0),"mon0900":("mon",9,0),
                   "mon0930":("mon",9,30),"mon1100":("mon",11,0),"mon1200":("mon",12,0),
                   "mon1600":("mon",16,0)}
    DOW = {"fri":4,"sat":5,"sun":6,"mon":0}

    panel_path = os.path.join(OUT,"weekend_panel.csv")
    fields = (["ticker","fri_date","mon_date","is_holiday_mon",
               "fri_close_stock","mon_open_stock","mon_high_stock","mon_low_stock","mon_close_stock"]
              + [f"p_{k}" for k in PERP_POINTS]
              + [f"spx_{k}" for k in PERP_POINTS]
              + ["spx_fri_close"])
    n_rows=0
    with open(panel_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(fields)
        for c in EQUITIES:
            if c not in stock or not perp.get(c): continue
            srows = stock[c]
            # индекс дат акции -> (o,h,l,cl)
            sidx = {d:(o,h,l,cl) for d,o,h,l,cl in srows}
            dates = [d for d,_,_,_,_ in srows]
            for i,d in enumerate(dates):
                if d.weekday()!=4:  # только пятницы
                    continue
                # следующая торговая сессия (mon или вт если праздник)
                mon = None; is_hol=0
                for j in range(i+1, min(i+4,len(dates))):
                    mon = dates[j]; break
                if mon is None: continue
                if (mon - d).days != 3:  # не обычный понедельник
                    is_hol = 1
                fri_close = sidx[d][3]
                mo,mh,ml,mc = sidx[mon]
                # перп-точки (ET-моменты ОТНОСИТЕЛЬНО недели пятницы d)
                def point_ms(spec):
                    name,hh,mm = spec
                    if name=="fri": dd=d
                    elif name=="sat": dd=d+datetime.timedelta(days=1)
                    elif name=="sun": dd=d+datetime.timedelta(days=2)
                    else: dd=mon  # mon = реальная след. сессия
                    return et_to_ms(dd,hh,mm)
                prow=[c,d.isoformat(),mon.isoformat(),is_hol,fri_close,mo,mh,ml,mc]
                for k,spec in PERP_POINTS.items():
                    prow.append(perp_at(perp[c], point_ms(spec)))
                for k,spec in PERP_POINTS.items():
                    prow.append(perp_at(perp[MARKET_PERP], point_ms(spec)))
                prow.append(perp_at(perp[MARKET_PERP], point_ms(("fri",16,0))))
                w.writerow(prow); n_rows+=1
    print(f"\n=== weekend_panel.csv: {n_rows} строк (ticker×weekend) ===", flush=True)
    print("Файлы:", panel_path)

if __name__=="__main__":
    main()
