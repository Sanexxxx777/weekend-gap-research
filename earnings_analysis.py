#!/usr/bin/env python3
"""
Проверка earnings-фильтра: гэп рядом с отчётом = реакция на новость = CONTINUATION (не fade)?
Гипотеза: long-fade на earnings-гэпах хуже/убыточен, на обычных — лучше. Если да -> фильтр "пропускать earnings".
Данные: длинная история акций (2022+, больше n) + earnings_dates (yfinance). LONG-сторона (основная).
"""
import pandas as pd, numpy as np
COST=0.001; THR=0.02
st=pd.read_csv("stock_long.csv",parse_dates=["date"]).sort_values(["ticker","date"])
ed=pd.read_csv("earnings_dates.csv",parse_dates=["earn_date"])
EARN={t:set(g.earn_date.dt.date) for t,g in ed.groupby("ticker")}

rows=[]
for tic,g in st.groupby("ticker"):
    g=g.reset_index(drop=True)
    for i in range(len(g)-1):
        if g.date[i].weekday()!=4: continue
        nxt=g.iloc[i+1]
        if (nxt.date-g.date[i]).days<1: continue
        rows.append({"ticker":tic,"fri":g.date[i],"mon":nxt.date,"fri_close":g.close[i],"mon_open":nxt.open,"mon_close":nxt.close})
p=pd.DataFrame(rows)
p["gap"]=p.mon_open/p.fri_close-1; p["side"]=np.where(p.gap>0,-1,1)
p["fade_pnl"]=p.side*(p.mon_close/p.mon_open-1)-COST
p=p[(p.gap.abs()>=THR)&(p.side==1)].copy()   # LONG only

def near_earn(r,lo,hi):
    """есть ли earnings в окне [fri+lo, fri+hi] дней."""
    es=EARN.get(r.ticker,set())
    for off in range(lo,hi+1):
        if (r.fri.date()+pd.Timedelta(days=off)) in es: return True
    return False

def boot(a,n=3000):
    a=np.asarray(a,float)
    if len(a)<4: return 1.0
    rng=np.random.default_rng(0); return float((rng.choice(a,size=(n,len(a)),replace=True).mean(1)<=0).mean())
def show(mask,label):
    s=p[mask]; wk=s.groupby("fri").fade_pnl.mean()
    if len(s)==0: print(f"  {label:40} нет"); return
    print(f"  {label:40} n={len(s):4} exp {s.fade_pnl.mean()*100:+.2f}% WR{(s.fade_pnl>0).mean()*100:.0f}% "
          f"weekend {wk.mean()*100:+.2f}% ({(wk>0).sum()}/{len(wk)}) p={boot(wk.values):.3f}")

print("="*104)
print(f"LONG-FADE: earnings рядом vs нет (длинная история {p.fri.min().date()}..{p.fri.max().date()}, |gap|>={THR*100:.0f}%)")
print("="*104)
# разные окна earnings относительно пятницы входа
for lo,hi,name in [(-3,3,"±3д от пятницы (тесное)"),(0,4,"пт..пт+4 (отчёт на той неделе)"),
                   (-5,0,"пт-5..пт (отчёт только что прошёл)"),(-7,7,"±7д (широкое)")]:
    p["earn"]=p.apply(lambda r:near_earn(r,lo,hi),axis=1)
    print(f"\n[окно {name}]")
    show(p.earn==True, "EARNINGS рядом (ожидаем continuation)")
    show(p.earn==False,"БЕЗ earnings (ожидаем чистый fade)")

# взаимодействие с размером гэпа (earnings = большие гэпы?)
print("\n[earnings ±3д × размер гэпа]")
p["earn"]=p.apply(lambda r:near_earn(r,-3,3),axis=1)
print(f"  доля earnings-гэпов: {p.earn.mean()*100:.0f}% | средн |gap| earnings={p[p.earn].gap.abs().mean()*100:.1f}% vs без={p[~p.earn].gap.abs().mean()*100:.1f}%")
