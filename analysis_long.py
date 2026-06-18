#!/usr/bin/env python3
"""
Анализ weekend-gap mean-reversion на ДЛИННОЙ истории акций (2022+, все прошлые данные).
gap = open(след.сессия после пятницы) / fri_close - 1.  FADE: gap>0 short / gap<0 long.
вход = open понедельника, выход = close понедельника. costs параметр.
ВАЖНО: это РОДСТВЕННАЯ форма (вход на открытии акции), НЕ идентична перп-стратегии (вход в выходные).
Цель — проверить РОБАСТНОСТЬ феномена на сотнях выходных и в РАЗНЫХ режимах (bear 2022 / bull 2023-26).
"""
import pandas as pd, numpy as np

COST = 0.001   # round-trip для ликвидных US-акций (taker+slip консервативно); покажем и 0.002
df = pd.read_csv("stock_long.csv", parse_dates=["date"])
df = df.sort_values(["ticker","date"]).reset_index(drop=True)

# построить weekend-панель: пятница -> следующая торговая сессия
rows=[]
for tic, g in df.groupby("ticker"):
    g=g.reset_index(drop=True)
    for i in range(len(g)-1):
        if g.date[i].weekday()!=4: continue          # только пятницы
        nxt=g.iloc[i+1]
        if (nxt.date - g.date[i]).days < 1: continue
        rows.append({"ticker":tic,"fri":g.date[i],"yr":g.date[i].year,
                     "fri_close":g.close[i],"mon_open":nxt.open,"mon_close":nxt.close,
                     "is_mon":(nxt.date-g.date[i]).days==3})
p=pd.DataFrame(rows)
p["gap"]=p.mon_open/p.fri_close-1
p["fade_pnl"]=np.where(p.gap>0,-1,1)*(p.mon_close/p.mon_open-1)-COST   # fade на open, выход close

print("="*72)
print("WEEKEND-GAP REVERSION на ДЛИННОЙ истории акций (вход open пн, выход close пн)")
print("="*72)
print(f"всего (ticker×weekend): {len(p)} | тикеров: {p.ticker.nunique()} | период: {p.fri.min().date()}..{p.fri.max().date()}")
print(f"выходных (уникальных пятниц): {p.fri.nunique()}")

def stats(sub, label):
    s=sub[sub.gap.abs()>=0.02]
    if len(s)==0: return f"{label}: нет"
    wk=s.groupby("fri").fade_pnl.mean()
    return (f"{label:18} n={len(s):4} вых={s.fri.nunique():3} | exp {s.fade_pnl.mean()*100:+.2f}%/сделку "
            f"WR {(s.fade_pnl>0).mean()*100:.0f}% | weekend {wk.mean()*100:+.2f}% ({(wk>0).sum()}/{len(wk)}) "
            f"p={'%.3f'%bootstrap(wk.values)}")

def bootstrap(a,n=3000):
    a=np.asarray(a,float)
    if len(a)<5: return 1.0
    rng=np.random.default_rng(0)
    return float((rng.choice(a,size=(n,len(a)),replace=True).mean(1)<=0).mean())

print("\n--- 1. КОНВЕРГЕНЦИЯ: реверсия gap внутри пн (slope (close/open-1)~gap, |gap|>=2%) ---")
big=p[p.gap.abs()>=0.02]
slope=np.polyfit(big.gap,big.mon_close/big.mon_open-1,1)[0]
print(f"  slope={slope:.3f} corr={np.corrcoef(big.gap,big.mon_close/big.mon_open-1)[0,1]:.3f} (n={len(big)})")
print(f"  => отрицательный slope = gap откатывает внутри пн (mean-reversion)")

print("\n--- 2. FADE |gap|>=2% ПО ГОДАМ (режимы: 2022 bear, 2023-25 bull, 2026) ---")
for yr in sorted(p.yr.unique()):
    print("  "+stats(p[p.yr==yr], str(yr)))
print("  "+stats(p,"ВСЕ ГОДЫ"))

print("\n--- 3. FADE по ПОРОГАМ гэпа (все годы) ---")
for thr in [0.01,0.02,0.03,0.05]:
    s=p[p.gap.abs()>=thr]; wk=s.groupby("fri").fade_pnl.mean()
    print(f"  |gap|>={thr*100:.0f}%: n={len(s):4} вых={s.fri.nunique():3} exp {s.fade_pnl.mean()*100:+.2f}% "
          f"WR {(s.fade_pnl>0).mean()*100:.0f}% weekend {wk.mean()*100:+.2f}% p={bootstrap(wk.values):.3f}")

print("\n--- 4. costs sensitivity (|gap|>=2%, все годы) ---")
for c in [0.0,0.001,0.002,0.003]:
    pnl=np.where(big.gap>0,-1,1)*(big.mon_close/big.mon_open-1)-c
    print(f"  cost={c*100:.1f}%: exp {pnl.mean()*100:+.2f}%/сделку WR {(pnl>0).mean()*100:.0f}%")
