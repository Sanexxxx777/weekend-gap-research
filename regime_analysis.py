#!/usr/bin/env python3
"""
КАЛИБРОВКА РЕЖИМ-ДЕТЕКТОРА на длинной истории акций (2022+, есть bull И bear).
Режим определяется на ПЯТНИЦУ (decision до входа в выходные, без look-ahead).
Стратегия: fade weekend-gap (вход open пн, выход close пн) — родственная форма перп-стратегии.
Вопрос: какой индикатор режима чисто разделяет "long-fade работает (bull)" vs "разворачивается (bear)".
"""
import pandas as pd, numpy as np

COST=0.001; THR=0.02
st=pd.read_csv("stock_long.csv",parse_dates=["date"]).sort_values(["ticker","date"])
mac=pd.read_csv("macro_long.csv",parse_dates=["date"])
spy=mac[mac.sym=="SPY"].set_index("date")["close"].sort_index()
vix=mac[mac.sym=="VIX"].set_index("date")["close"].sort_index()
btc=mac[mac.sym=="BTC"].set_index("date")["close"].sort_index()

# режим-фичи по дате (на каждый торговый день SPY)
reg=pd.DataFrame(index=spy.index)
reg["spy"]=spy; reg["ma50"]=spy.rolling(50).mean(); reg["ma200"]=spy.rolling(200).mean()
reg["above200"]=spy>reg.ma200; reg["above50"]=spy>reg.ma50
reg["mom20"]=spy/spy.shift(20)-1
reg["vix"]=vix.reindex(spy.index).ffill()
reg["vix_ma20"]=reg["vix"].rolling(20).mean()
reg["btc_mom20"]=(btc.reindex(spy.index).ffill())/(btc.reindex(spy.index).ffill().shift(20))-1

# weekend-панель акций
rows=[]
for tic,g in st.groupby("ticker"):
    g=g.reset_index(drop=True)
    for i in range(len(g)-1):
        if g.date[i].weekday()!=4: continue
        nxt=g.iloc[i+1]
        if (nxt.date-g.date[i]).days<1: continue
        rows.append({"ticker":tic,"fri":g.date[i],"fri_close":g.close[i],"mon_open":nxt.open,"mon_close":nxt.close})
p=pd.DataFrame(rows)
p["gap"]=p.mon_open/p.fri_close-1
p["side"]=np.where(p.gap>0,-1,1)
p["fade_pnl"]=p.side*(p.mon_close/p.mon_open-1)-COST
p=p[p.gap.abs()>=THR].copy()
# приклеить режим пятницы (last reg-строка <= fri)
def regat(d,col):
    s=reg[col]; s=s[s.index<=d]
    return s.iloc[-1] if len(s) else np.nan
for col in ["above200","above50","mom20","vix","vix_ma20","btc_mom20"]:
    p[col]=p.fri.map(lambda d:regat(d,col))
p=p.dropna(subset=["above200","mom20","vix"])

def boot(a,n=3000):
    a=np.asarray(a,float)
    if len(a)<4: return 1.0
    rng=np.random.default_rng(0); return float((rng.choice(a,size=(n,len(a)),replace=True).mean(1)<=0).mean())
def show(mask,label):
    s=p[mask]; L=s[s.side==1]; S=s[s.side==-1]
    wkL=L.groupby("fri").fade_pnl.mean()
    print(f"  {label:34} | ALL n={len(s):4} {s.fade_pnl.mean()*100:+.2f}% | "
          f"LONG n={len(L):4} {L.fade_pnl.mean()*100:+.2f}% WR{(L.fade_pnl>0).mean()*100:.0f}% p={boot(wkL.values):.3f} | "
          f"SHORT n={len(S):4} {S.fade_pnl.mean()*100:+.2f}%")

print("="*120)
print(f"FADE по режиму (длинная история {p.fri.min().date()}..{p.fri.max().date()}, |gap|>={THR*100:.0f}%, costs {COST*100:.1f}%)")
print("="*120)
print("\n[A] SPY vs MA200 (главный тренд-фильтр):")
show(p.above200==True,"BULL (SPY>MA200)"); show(p.above200==False,"BEAR (SPY<MA200)")
print("\n[B] SPY 20-дн momentum:")
show(p.mom20>0,"mom20>0 (растёт)"); show(p.mom20<=0,"mom20<=0 (падает)")
print("\n[C] VIX уровень:")
show(p.vix<20,"VIX<20 (спокойно)"); show((p.vix>=20)&(p.vix<28),"VIX 20-28"); show(p.vix>=28,"VIX>=28 (паника)")
print("\n[D] SPY vs MA50 (быстрый):")
show(p.above50==True,"SPY>MA50"); show(p.above50==False,"SPY<MA50")
print("\n[E] КОМБО (bull-фильтр: SPY>MA200 И VIX<25):")
show((p.above200==True)&(p.vix<25),"BULL+спокойно (торгуем LONG)")
show(~((p.above200==True)&(p.vix<25)),"иначе (пауза/осторожно)")
print("\n[F] по годам (контекст):")
for yr in sorted(p.fri.dt.year.unique()):
    show(p.fri.dt.year==yr,f"{yr}")
