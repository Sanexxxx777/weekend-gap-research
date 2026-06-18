#!/usr/bin/env python3
"""
Проверка идеи Саши: Heikin Ashi тренд как режим-фильтр для weekend-gap long-fade.
HA сглаживает шум -> может отличить "коррекция (быстрый отскок)" от "структурный bear (серия красных HA)".
Калибровка на длинной истории акций (2022+, есть bear). Сравнение с MA200 (который НЕ разделил).
HA на SPY (рыночный режим) и BTC (crypto-режим). Daily + Weekly(W-FRI). decision = пятница (без look-ahead).
"""
import pandas as pd, numpy as np

COST=0.001; THR=0.02
oh=pd.read_csv("macro_ohlc.csv",parse_dates=["date"])

def heikin(df):
    """df: index=date, cols open/high/low/close. Возвращает ha с ha_green, ha_streak."""
    df=df.sort_index()
    hac=(df.open+df.high+df.low+df.close)/4
    hao=pd.Series(index=df.index,dtype=float)
    hao.iloc[0]=(df.open.iloc[0]+df.close.iloc[0])/2
    for i in range(1,len(df)):
        hao.iloc[i]=(hao.iloc[i-1]+hac.iloc[i-1])/2
    green=hac>hao
    # streak: длина текущей серии цвета (+N зелёных / -N красных)
    streak=[]; cur=0
    for g in green:
        if streak and ((g and cur>0) or ((not g) and cur<0)): cur += (1 if g else -1)
        else: cur = 1 if g else -1
        streak.append(cur)
    return pd.DataFrame({"ha_green":green,"ha_streak":streak},index=df.index)

def build(sym):
    d=oh[oh.sym==sym].set_index("date")[["open","high","low","close"]].sort_index()
    ha_d=heikin(d)
    # weekly W-FRI
    w=d.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    ha_w=heikin(w)
    return ha_d, ha_w
spy_d,spy_w=build("SPY"); btc_d,btc_w=build("BTC")

# weekend-панель акций
st=pd.read_csv("stock_long.csv",parse_dates=["date"]).sort_values(["ticker","date"])
rows=[]
for tic,g in st.groupby("ticker"):
    g=g.reset_index(drop=True)
    for i in range(len(g)-1):
        if g.date[i].weekday()!=4: continue
        nxt=g.iloc[i+1]
        if (nxt.date-g.date[i]).days<1: continue
        rows.append({"ticker":tic,"fri":g.date[i],"fri_close":g.close[i],"mon_open":nxt.open,"mon_close":nxt.close})
p=pd.DataFrame(rows)
p["gap"]=p.mon_open/p.fri_close-1; p["side"]=np.where(p.gap>0,-1,1)
p["fade_pnl"]=p.side*(p.mon_close/p.mon_open-1)-COST
p=p[(p.gap.abs()>=THR)&(p.side==1)].copy()   # ТОЛЬКО LONG (основная сторона)

def at(ser,d):
    s=ser[ser.index<=d]; return s.iloc[-1] if len(s) else np.nan
p["spy_ha_d"]=p.fri.map(lambda d:at(spy_d.ha_green,d))
p["spy_ha_streak"]=p.fri.map(lambda d:at(spy_d.ha_streak,d))
p["spy_ha_w"]=p.fri.map(lambda d:at(spy_w.ha_green,d))
p["btc_ha_d"]=p.fri.map(lambda d:at(btc_d.ha_green,d))
p["btc_ha_w"]=p.fri.map(lambda d:at(btc_w.ha_green,d))
p=p.dropna(subset=["spy_ha_d","spy_ha_w","btc_ha_d"])

def boot(a,n=3000):
    a=np.asarray(a,float)
    if len(a)<4: return 1.0
    rng=np.random.default_rng(0); return float((rng.choice(a,size=(n,len(a)),replace=True).mean(1)<=0).mean())
def show(mask,label):
    s=p[mask]; wk=s.groupby("fri").fade_pnl.mean()
    if len(s)==0: print(f"  {label:32} нет"); return
    print(f"  {label:32} n={len(s):4} exp {s.fade_pnl.mean()*100:+.2f}% WR{(s.fade_pnl>0).mean()*100:.0f}% "
          f"weekend {wk.mean()*100:+.2f}% ({(wk>0).sum()}/{len(wk)}) p={boot(wk.values):.3f}")

print("="*100)
print(f"LONG-FADE по Heikin Ashi режиму (длинная история {p.fri.min().date()}..{p.fri.max().date()}, |gap|>={THR*100:.0f}%)")
print("ЦЕЛЬ: HA-bull -> long значим (+,p<0.1); HA-bear -> убыток/незначим (фильтр работает)")
print("="*100)
print("\n[1] SPY HA дневная свеча:")
show(p.spy_ha_d==True,"SPY HA зелёная (аптренд)"); show(p.spy_ha_d==False,"SPY HA красная (даунтренд)")
print("\n[2] SPY HA НЕДЕЛЬНАЯ свеча (логичнее для weekend):")
show(p.spy_ha_w==True,"SPY HA-week зелёная"); show(p.spy_ha_w==False,"SPY HA-week красная")
print("\n[3] SPY HA-streak (сила/длина тренда):")
show(p.spy_ha_streak>=2,"streak>=+2 (устойч.аптренд)"); show((p.spy_ha_streak>=-1)&(p.spy_ha_streak<2),"streak -1..+1 (слабо)")
show(p.spy_ha_streak<=-2,"streak<=-2 (устойч.даунтренд)")
print("\n[4] BTC HA дневная (crypto-режим):")
show(p.btc_ha_d==True,"BTC HA зелёная"); show(p.btc_ha_d==False,"BTC HA красная")
print("\n[5] BTC HA недельная:")
show(p.btc_ha_w==True,"BTC HA-week зелёная"); show(p.btc_ha_w==False,"BTC HA-week красная")
print("\n[6] КОМБО SPY-week зелёная И BTC-day зелёная:")
show((p.spy_ha_w==True)&(p.btc_ha_d==True),"оба HA-бычьи"); show(~((p.spy_ha_w==True)&(p.btc_ha_d==True)),"иначе")
print("\n[baseline] для сравнения — без фильтра:")
show(p.index>=0,"ВСЕ long")
