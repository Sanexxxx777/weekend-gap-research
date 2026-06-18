#!/usr/bin/env python3
"""
ФИНАЛЬНЫЙ бэктест по правилам Саши + проверка Webo. Период: с декабря 2025 (перп-эра).
Вход: вс 20:00 ET (p_sun2000). Выход: открытие NASDAQ (p_mon0930), 1h-данные.
Правила: (1) порог >1.5%; (2) асимметрия gaps up/down; (3) per-ticker long/short/skip с OOS-split;
(4) правило "+10мин не подтянулось -> выход" на 15m-окне (~7 пн); (5) crypto-связь (бета к BTC).
costs 0.19% RT. Честные оговорки про малую выборку — в выводе.
"""
import pandas as pd, numpy as np, datetime
from zoneinfo import ZoneInfo
import wg_lib as wg

NY=ZoneInfo("America/New_York"); COST=wg.RT_COST
DEC=pd.Timestamp("2025-12-01")

p = wg.add_features(wg.load(only_regular_monday=True), entry="p_sun2000")
p = p[p.fri_date >= DEC].copy()
p = p.dropna(subset=["p_sun2000","p_mon0930","fri_close_stock"])
p["pnl"] = np.where(p.gap>0,-1,1)*(p.p_mon0930/p.entry_perp-1) - COST
p["dir"] = np.where(p.gap>0,"short(gap_up)","long(gap_dn)")

def boot(a,n=3000):
    a=np.asarray(a,float)
    if len(a)<4: return 1.0
    rng=np.random.default_rng(0); return float((rng.choice(a,size=(n,len(a)),replace=True).mean(1)<=0).mean())

print("="*74)
print(f"ПЕРИОД С ДЕКАБРЯ: {p.fri_date.min().date()}..{p.fri_date.max().date()}, выходных {p.fri_date.nunique()}, тикеров {p.ticker.nunique()}")
print(f"вход вс 20:00 ET, выход открытие NASDAQ, costs {COST*100:.2f}%")
print("="*74)

print("\n--- 1. ПОРОГ ГЭПА (проверка Webo >1.5%) ---")
for thr in [0.01,0.015,0.02,0.025,0.03]:
    s=p[p.gap.abs()>=thr]; wk=s.groupby("fri_date").pnl.mean()
    print(f"  |gap|>={thr*100:>4.1f}%: n={len(s):3} exp {s.pnl.mean()*100:+.2f}% WR {(s.pnl>0).mean()*100:.0f}% "
          f"weekend {wk.mean()*100:+.2f}% ({(wk>0).sum()}/{len(wk)}) p={boot(wk.values):.3f}")

print("\n--- 2. АСИММЕТРИЯ gaps UP vs DOWN (Webo: down закрывается чаще) ---")
for thr in [0.015,0.02]:
    print(f"  порог {thr*100:.1f}%:")
    for dr in ["short(gap_up)","long(gap_dn)"]:
        s=p[(p.gap.abs()>=thr)&(p.dir==dr)]
        if len(s)==0: continue
        print(f"    {dr:16} n={len(s):3} exp {s.pnl.mean()*100:+.2f}% WR {(s.pnl>0).mean()*100:.0f}% p={boot(s.pnl.values):.3f}")

print("\n--- 3. PER-TICKER long/short/skip (порог 1.5%) + OOS-split проверка overfit ---")
thr=0.015
dates=np.sort(p.fri_date.unique()); cut=dates[len(dates)//2]
def ticker_table(df):
    out={}
    for t in df.ticker.unique():
        for dr in ["short(gap_up)","long(gap_dn)"]:
            s=df[(df.ticker==t)&(df.dir==dr)&(df.gap.abs()>=thr)]
            if len(s)>=1: out[(t,dr)]=(len(s), s.pnl.mean())
    return out
full=ticker_table(p)
# списки по ВСЕМ данным
good_long=sorted([t for (t,d),(n,e) in full.items() if d=="long(gap_dn)" and e>0 and n>=2], key=lambda t:-full[(t,"long(gap_dn)")][1])
good_short=sorted([t for (t,d),(n,e) in full.items() if d=="short(gap_up)" and e>0 and n>=2], key=lambda t:-full[(t,"short(gap_up)")][1])
bad=[t for (t,d),(n,e) in full.items() if e<0 and n>=3]
print(f"  GOOD_LONG (gap_dn->buy, exp>0, n>=2): {good_long}")
print(f"  GOOD_SHORT (gap_up->sell, exp>0, n>=2): {good_short}")
print(f"  УБЫТОЧНЫЕ (exp<0, n>=3): {sorted(set(bad))}")
# OOS: список на train-половине -> проверить на test-половине
tr=p[p.fri_date<cut]; te=p[p.fri_date>=cut]
tr_tab=ticker_table(tr)
tr_good=set(t for (t,d),(n,e) in tr_tab.items() if e>0 and n>=2)
te_in = te[te.apply(lambda r:(r.ticker in tr_good) and abs(r.gap)>=thr, axis=1)]
te_out= te[te.apply(lambda r:(r.ticker not in tr_good) and abs(r.gap)>=thr, axis=1)]
print(f"  OOS-ТЕСТ: тикеры 'хорошие на train' -> на test exp {te_in.pnl.mean()*100:+.2f}% (n={len(te_in)}); "
      f"остальные на test exp {te_out.pnl.mean()*100:+.2f}% (n={len(te_out)})")
print(f"  => если 'хорошие' НЕ лучше остальных на test = списки = OVERFIT (не воспроизводятся)")

print("\n--- 4. ПРАВИЛО '+10мин не подтянулось -> выход' (15m-окно, мало пн) ---")
h15=pd.read_csv("perp_15m.csv")
H15={t:g.sort_values("t_open_ms")[["t_open_ms","T_close_ms","high","low","close"]].values for t,g in h15.groupby("ticker")}
def et_ms(d,hh,mm):
    dd=pd.Timestamp(d); return int(datetime.datetime(dd.year,dd.month,dd.day,hh,mm,tzinfo=NY).timestamp()*1000)
def perp15_at(t,ms):
    cd=H15.get(t);
    if cd is None: return None
    best=None
    for to,tc,hi,lo,cl in cd:
        if tc<=ms: best=cl
        else: break
    return best
rows_a=[]; rows_c=[]
for r in p[p.gap.abs()>=0.015].itertuples():
    t=r.ticker; e=r.entry_perp; side=-1 if r.gap>0 else 1; fc=r.fri_close_stock
    o930=perp15_at(t,et_ms(r.mon_date,9,30)); o945=perp15_at(t,et_ms(r.mon_date,9,45)); o1030=perp15_at(t,et_ms(r.mon_date,10,30))
    if o930 is None or o945 is None: continue   # только пн с 15m данными
    # (a) выход на открытии
    rows_a.append(side*(o930/e-1)-COST)
    # (c) правило Саши: подтянулся ли gap к 9:45? |perp945/fc-1| < |gap| -> держим до 10:30, иначе выход 9:45
    pulled = abs(o945/fc-1) < abs(r.gap)
    exitpx = (o1030 if o1030 else o945) if pulled else o945
    rows_c.append(side*(exitpx/e-1)-COST)
if rows_a:
    a=np.array(rows_a); c=np.array(rows_c)
    print(f"  пн с 15m-данными: {len(a)} сделок")
    print(f"  (a) выход на открытии 9:30:        exp {a.mean()*100:+.2f}% WR {(a>0).mean()*100:.0f}%")
    print(f"  (c) правило +10мин (Саша):         exp {c.mean()*100:+.2f}% WR {(c>0).mean()*100:.0f}%")
    print(f"  => {'правило ПОМОГАЕТ' if c.mean()>a.mean() else 'правило НЕ помогает на этой выборке'} (n мал, осторожно)")
else:
    print("  нет пересечения 15m-данных с понедельниками периода")

print("\n--- 5. CRYPTO-СВЯЗЬ (бета к BTC) на периоде с декабря ---")
try:
    btc=pd.read_csv("btc_hourly.csv"); hh=pd.read_csv("perp_hourly.csv")
    dec_ms=int(DEC.timestamp()*1000)
    piv=hh[hh.t_open_ms>=dec_ms].pivot_table(index="t_open_ms",columns="ticker",values="close")
    rb=np.log(btc[btc.t_open_ms>=dec_ms].set_index("t_open_ms")["close"]).diff()
    betas={}
    for c in piv.columns:
        if c=="SP500": continue
        xy=pd.concat([np.log(piv[c]).diff(),rb],axis=1).dropna()
        if len(xy)>100 and xy.iloc[:,1].var()>0: betas[c]=float(np.cov(xy.iloc[:,0],xy.iloc[:,1])[0,1]/xy.iloc[:,1].var())
    hi=set(c for c,b in betas.items() if b>=0.4); lo=set(c for c,b in betas.items() if b<0.4)
    for grp,names in [("HIGH crypto-beta",hi),("LOW crypto-beta",lo)]:
        s=p[(p.ticker.isin(names))&(p.gap.abs()>=0.015)]
        if len(s)==0: continue
        wk=s.groupby("fri_date").pnl.mean()
        print(f"  {grp:16} n={len(s):3} exp {s.pnl.mean()*100:+.2f}% WR {(s.pnl>0).mean()*100:.0f}% "
              f"weekend {wk.mean()*100:+.2f}% p={boot(wk.values):.3f}")
except Exception as e:
    print("  crypto-блок:", e)
