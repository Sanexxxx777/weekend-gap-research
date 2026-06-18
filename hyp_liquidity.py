import wg_lib as wg
import numpy as np
import pandas as pd

# Signal: fade |gap|>=2% (short positive gap, long negative gap)
def fade2(r):
    if abs(r.gap) >= 0.02:
        return -1 if r.gap > 0 else 1
    return 0

EXIT = "p_mon0930"
ENTRY = "p_mon0630"

print("="*70)
print("STEP 1: drop_thin=False vs drop_thin=True  (fade |gap|>=2%)")
print("="*70)

for dt in (False, True):
    d = wg.load(only_regular_monday=True, drop_thin=dt)
    d = wg.add_features(d, entry=ENTRY)
    m = wg.walk_forward(d, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kf = wg.kfold(d, fade2, exit=EXIT)
    print(f"\n--- drop_thin={dt} ---")
    print(f"train_n={m['train_n']} train_exp={m['train_exp']*100:.3f}% train_wr={m['train_wr']*100:.1f}%")
    print(f"test_n ={m['test_n']} test_exp ={m['test_exp']*100:.3f}% test_wr ={m['test_wr']*100:.1f}% test_p={m['test_p']:.4f}")
    print(f"all_n  ={m['all_n']} all_exp  ={m['all_exp']*100:.3f}%")
    print(f"kfold = {[round(x*100,3) for x in kf]}")
    print(f"cut_date={m['cut_date']}")

print("\n" + "="*70)
print("STEP 2: per-ticker TEST contribution + leave-one-out  (drop_thin=False)")
print("="*70)

d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry=ENTRY)

# reconstruct test split exactly as walk_forward
dates = np.sort(d.fri_date.unique())
cut = dates[int(len(dates)*0.6)]
te = d[d.fri_date >= cut].copy()

# pnl on test set
pn = wg.pnl_series(te, fade2, exit=EXIT, cost=wg.RT_COST)
te_traded = te.loc[pn.index].copy()
te_traded["pnl"] = pn.values

contrib = te_traded.groupby("ticker")["pnl"].agg(["sum","count","mean"]).sort_values("sum")
contrib["sum_pct"] = contrib["sum"]*100
contrib["mean_pct"] = contrib["mean"]*100
print("\nPer-ticker TEST sum (sorted by sum, pct points):")
print(contrib[["count","sum_pct","mean_pct"]].to_string())

total_test_sum = pn.sum()
total_test_n = len(pn)
print(f"\nTOTAL test_sum={total_test_sum*100:.3f}pp over n={total_test_n} -> exp={total_test_sum/total_test_n*100:.4f}%")

# rank by absolute contribution
contrib_abs = contrib.reindex(contrib["sum"].abs().sort_values(ascending=False).index)
print("\nRanked by |contribution|:")
print((contrib_abs[["count"]].assign(sum_pp=contrib_abs["sum"]*100)).to_string())

top1 = contrib_abs.index[0]
top2 = contrib_abs.index[1]
print(f"\nTop-1 |contrib|: {top1}  ({contrib.loc[top1,'sum']*100:.3f}pp)")
print(f"Top-2 |contrib|: {top2}  ({contrib.loc[top2,'sum']*100:.3f}pp)")

def loo(exclude):
    de = d[~d.ticker.isin(exclude)].copy()
    m = wg.walk_forward(de, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kf = wg.kfold(de, fade2, exit=EXIT)
    return m, kf

print("\n--- Leave-one-out (drop top-1 by |contrib|): exclude", top1, "---")
m1, kf1 = loo({top1})
print(f"test_n={m1['test_n']} test_exp={m1['test_exp']*100:.3f}% test_wr={m1['test_wr']*100:.1f}% test_p={m1['test_p']:.4f}")
print(f"kfold = {[round(x*100,3) for x in kf1]}")

print("\n--- Leave-two-out (drop top-1+top-2): exclude", top1, top2, "---")
m2, kf2 = loo({top1, top2})
print(f"test_n={m2['test_n']} test_exp={m2['test_exp']*100:.3f}% test_wr={m2['test_wr']*100:.1f}% test_p={m2['test_p']:.4f}")
print(f"kfold = {[round(x*100,3) for x in kf2]}")

print("\n" + "="*70)
print("STEP 3: inversion sanity (drop_thin=False)")
print("="*70)
def fade2_inv(r):
    s = fade2(r)
    return -s
mi = wg.walk_forward(d, fade2_inv, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
print(f"inverted test_exp={mi['test_exp']*100:.3f}% test_wr={mi['test_wr']*100:.1f}%")
