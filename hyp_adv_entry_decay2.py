import wg_lib as wg
import numpy as np
import pandas as pd

ENTRY = "p_mon0000"; EXIT = "p_mon0930"; THR = 0.02
def fade(r):
    if abs(r.gap) >= THR: return -1 if r.gap > 0 else +1
    return 0

d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry=ENTRY)
m = wg.walk_forward(d, fade, exit=EXIT)
cut = pd.Timestamp(m['cut_date'])

print("DEEPER LOO: drop_thin=True (removes GME,ZM,EBAY,BB,RKLB,DKNG,HIMS) + drop top contributors")
dt = wg.load(only_regular_monday=True, drop_thin=True)
dt = wg.add_features(dt, entry=ENTRY)
mt = wg.walk_forward(dt, fade, exit=EXIT)
kft = wg.kfold(dt, fade, exit=EXIT)
print(f"drop_thin=True: test n={mt['test_n']} exp={mt['test_exp']*100:.3f}% wr={mt['test_wr']*100:.1f}% p={mt['test_p']:.4f} kf=[{','.join(f'{x*100:+.2f}' for x in kft)}]")

# progressive LOO: drop top 1,2,3,4,5 signed contributors
te = d[d.fri_date >= cut].copy()
pn = wg.pnl_series(te, fade, exit=EXIT)
te_t = te.loc[pn.index].copy(); te_t["_pnl"] = pn.values
contrib = te_t.groupby("ticker")["_pnl"].sum().sort_values(ascending=False)
print("\nProgressive drop of top-K positive contributors:")
for k in range(0,6):
    drop = list(contrib.index[:k])
    dk = d[~d.ticker.isin(drop)].copy()
    mk = wg.walk_forward(dk, fade, exit=EXIT)
    print(f"  drop top-{k} {str(drop):<40} test n={mk['test_n']:>3} exp={mk['test_exp']*100:>7.3f}% p={mk['test_p']:.4f}")

# how concentrated: share of test_sum from top-2
total = contrib.sum()
print(f"\ntest_sum total={total*100:.2f}%, top2(GME,MSTR)={contrib.iloc[:2].sum()*100:.2f}% = {contrib.iloc[:2].sum()/total*100:.1f}% of total")
print(f"n trades GME={(te_t.ticker=='GME').sum()}, MSTR={(te_t.ticker=='MSTR').sum()}, total test trades={len(te_t)}")

# Per-trade robustness: median per-trade pnl (less sensitive to outliers)
print(f"\nTest per-trade: mean={pn.mean()*100:.3f}% median={pn.median()*100:.3f}% (median>0 => not driven by few big wins)")
print(f"# trades >0: {(pn>0).sum()}/{len(pn)}; remove single best trade -> exp={pn[pn<pn.max()].mean()*100:.3f}%")

# Offset: is p_sun2000/p_mon0600 the SAME trades or different? Check overlap of traded rows
print("\nOFFSET sanity: are neighbor entries trading mostly the same gaps (same edge shifted) or different selection?")
for e in ["p_sun2000","p_mon0000","p_mon0600","p_mon0800"]:
    de = wg.load(only_regular_monday=True, drop_thin=False)
    de = wg.add_features(de, entry=e)
    sig = de.apply(fade, axis=1)
    n_signal = (sig!=0).sum()
    print(f"  {e}: rows with |gap|>=2% = {n_signal} (avg |gap| among them = {de.loc[sig!=0,'gap'].abs().mean()*100:.2f}%)")
