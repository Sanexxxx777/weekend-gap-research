#!/usr/bin/env python3
"""
ПОЛНЫЙ бэктест weekend-gap на ВСЕХ US-tradeable HL-перпах. Гэп считается к NASDAQ/NYSE.
ИСТОЧНИКИ (что и где смотрим):
  - Цена акции на бирже: yfinance daily (Open=open NASDAQ 9:30 ET, Close=close 16:00 ET).
  - Цена перпа: HL candleSnapshot 1h (xyz:TICKER), honest = last завершённая свеча <= момента.
  - gap = perp(момент решения) / fri_close_stock - 1   (перп vs пятничный close на NASDAQ)
  - таргет/исход: open понедельника на NASDAQ (mon_open_stock) и далее.
  - costs: round-trip 0.19% (taker 0.045%/side + slip 0.05%/side). walk-forward: train 60% ранних выходных, test 40% свежих.
"""
import wg_lib as wg, pandas as pd, numpy as np

print("="*70)
print("ДАТАСЕТ (источник: HL perps + yfinance daily NASDAQ/NYSE)")
print("="*70)
raw = wg.load(only_regular_monday=False)
print(f"всего строк (тикер×выходной): {len(raw)} | тикеров: {raw.ticker.nunique()} | выходных: {raw.fri_date.nunique()}")
print(f"период: {raw.fri_date.min().date()} -> {raw.fri_date.max().date()}")

d = wg.load(only_regular_monday=True)
d = wg.add_features(d, entry="p_mon0000")
print(f"\nвалидных обычных пн (есть перп Mon00:00 + open/close NASDAQ): {len(d)}")
print(f"holiday-пн отброшено: {(wg.load(only_regular_monday=False).is_holiday_mon==1).sum()}")

# === 1. РАСПРЕДЕЛЕНИЕ ГЭПОВ (перп Mon00:00 vs пятничный close NASDAQ) ===
print("\n" + "="*70)
print("1. ГЭП = перп(Mon 00:00 ET) / пятничный_close_NASDAQ - 1")
print("="*70)
print(f"mean {d.gap.mean()*100:+.2f}%  median {d.gap.median()*100:+.2f}%  std {d.gap.std()*100:.2f}%")
for thr in [0.01,0.02,0.03,0.05]:
    print(f"  |gap|>={thr*100:.0f}%: {(d.gap.abs()>=thr).sum()} наблюдений")

# === 2. КОНВЕРГЕНЦИЯ: перп сходится к пятнице или к open понедельника (NASDAQ)? ===
print("\n" + "="*70)
print("2. К ЧЕМУ СХОДИТСЯ ГЭП: регрессия (open_NASDAQ/fri-1) ~ gap")
print("="*70)
big = d[d.gap.abs()>=0.02]
slope = np.polyfit(big.gap, big.open_ret, 1)[0]; corr = np.corrcoef(big.gap, big.open_ret)[0,1]
print(f"на |gap|>=2% (n={len(big)}): slope={slope:.3f} corr={corr:.3f}")
print(f"=> {slope*100:.0f}% гэпа реализуется в open NASDAQ, {(1-slope)*100:.0f}% откатывается (фейдится)")

# === 3. FADE walk-forward по ВСЕМ именам (без сужения корзины) ===
print("\n" + "="*70)
print("3. FADE гэпа по ВСЕМ 37 именам, walk-forward + costs 0.19%")
print("   (short если перп>пятницы, long если <; выход = перп на open NASDAQ)")
print("="*70)
def fade(thr):
    return lambda r: 0 if abs(r.gap)<thr else (-1 if r.gap>0 else 1)
for thr in [0.015,0.02,0.025,0.03]:
    for exit in ["p_mon0930","p_mon1600"]:
        m = wg.walk_forward(d, fade(thr), exit=exit)
        kf = wg.kfold(d, fade(thr), exit=exit)
        tag = "open" if exit=="p_mon0930" else "close-пн"
        print(f"  thr={thr*100:.1f}% exit={tag:8}: TRAIN {m['train_exp']*100:+.2f}%(n{m['train_n']}) "
              f"TEST {m['test_exp']*100:+.2f}%(n{m['test_n']}) WR{m['test_wr']*100:.0f}% p={m['test_p']:.3f} "
              f"kf={[round(x*100,1) for x in kf]}")

# === 4. РАЗБИВКА ПО ТИКЕРАМ (кто даёт сделки и edge) ===
print("\n" + "="*70)
print("4. ПО ТИКЕРАМ (fade>=2%, entry Mon00:00, exit open): вклад каждого")
print("="*70)
f2 = fade(0.02)
rows=[]
for tic in sorted(d.ticker.unique()):
    sub = d[d.ticker==tic]
    pn = wg.pnl_series(sub, f2, exit="p_mon0930")
    if len(pn)>0:
        rows.append((tic, len(pn), pn.mean()*100, pn.sum()*100, (pn>0).mean()*100))
rows.sort(key=lambda x:-x[3])
print(f"{'тикер':7}{'сделок':>7}{'mean%':>8}{'sum%':>9}{'WR%':>6}")
for t,n,me,su,wr in rows:
    print(f"{t:7}{n:>7}{me:>8.2f}{su:>9.1f}{wr:>6.0f}")
print(f"\nвсего тикеров со сделками |gap|>=2%: {len(rows)}")
EOF_MARKER = True
