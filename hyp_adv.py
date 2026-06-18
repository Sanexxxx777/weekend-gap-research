import wg_lib as wg, numpy as np, pandas as pd
pd.set_option('display.width', 240)

THIN = wg.THIN
ENTRY = "p_mon0000"
THR   = 0.025
EXIT  = "p_mon0930"

def cand(r):
    if abs(r.gap) < THR: return 0
    if r.gap > 0: return 0
    return 1

def cand_inv(r):
    s = cand(r)
    return -s

def runwf(d, fn, exit=EXIT):
    return wg.walk_forward(d, fn, exit=exit, train_frac=0.6)

d0 = wg.add_features(wg.load(True, False), entry=ENTRY)

print("=== 0. REPRODUCE candidate (long_bias, entry=p_mon0000, thr=0.025, exit=p_mon0930) ===")
m = runwf(d0, cand)
print(f"  TRAIN exp={m['train_exp']*100:.4f}% n={m['train_n']} wr={m['train_wr']:.3f} sharpe={m['train_sharpe']:.3f}")
print(f"  TEST  exp={m['test_exp']*100:.4f}% n={m['test_n']} wr={m['test_wr']:.3f} sharpe={m['test_sharpe']:.3f} p={m['test_p']:.4f}")
print(f"  ALL   exp={m['all_exp']*100:.4f}% n={m['all_n']}")
print(f"  kfold {[round(x*100,4) for x in wg.kfold(d0, cand, exit=EXIT)]}  cut={m['cut_date']}")

print("\n=== also reproduce BOTH-SIDED prod candidate (same entry/thr/exit) ===")
def both(r):
    if abs(r.gap) < THR: return 0
    return -1 if r.gap > 0 else 1
mb = runwf(d0, both)
print(f"  TEST  exp={mb['test_exp']*100:.4f}% n={mb['test_n']} p={mb['test_p']:.4f} kfold={[round(x*100,3) for x in wg.kfold(d0, both, exit=EXIT)]}")

print("\n=== 1. INVERSION (candidate long_bias) ===")
mi = runwf(d0, cand_inv)
print(f"  TEST inverted exp={mi['test_exp']*100:.4f}% n={mi['test_n']} p={mi['test_p']:.4f}")

print("\n=== 2. OFFSET entry +-1 decision-point ===")
DP = wg.DECISION_POINTS
i = DP.index(ENTRY)
for j in [i-1, i+1]:
    if j < 0 or j >= len(DP):
        print(f"  offset idx {j}: out of range (entry={ENTRY} at edge of DECISION_POINTS)")
        continue
    e = DP[j]
    dE = wg.add_features(wg.load(True, False), entry=e)
    mE = runwf(dE, cand)
    print(f"  entry={e}: TRAIN exp={mE['train_exp']*100:.4f}% n={mE['train_n']} | TEST exp={mE['test_exp']*100:.4f}% n={mE['test_n']} p={mE['test_p']:.4f} kfold={[round(x*100,3) for x in wg.kfold(dE,cand,exit=EXIT)]}")

print("\n=== 3. TICKER LEAVE-OUT-TOP-2 (by test_sum contribution) ===")
dates = np.sort(d0.fri_date.unique()); cut = dates[int(len(dates)*0.6)]
te = d0[d0.fri_date >= cut].copy()
pn_te = wg.pnl_series(te, cand, exit=EXIT)
te_used = te.loc[pn_te.index].copy()
te_used["_pnl"] = pn_te.values
contrib = te_used.groupby("ticker")["_pnl"].sum().sort_values(ascending=False)
print("  per-ticker test_sum contribution:")
print(contrib.to_string())
top2 = list(contrib.head(2).index)
print(f"  top2 = {top2}")
d_lo = d0[~d0.ticker.isin(top2)].copy()
mlo = runwf(d_lo, cand)
print(f"  LOO-top2 TEST exp={mlo['test_exp']*100:.4f}% n={mlo['test_n']} wr={mlo['test_wr']:.3f} p={mlo['test_p']:.4f}")
print(f"  LOO-top2 kfold {[round(x*100,3) for x in wg.kfold(d_lo, cand, exit=EXIT)]}")

print("\n=== 4. SUBPERIOD: split test in half by time ===")
te_dates = np.sort(te.fri_date.unique())
half = te_dates[len(te_dates)//2]
te_h1 = te[te.fri_date < half]; te_h2 = te[te.fri_date >= half]
for lbl, sub in [("H1", te_h1), ("H2", te_h2)]:
    pn = wg.pnl_series(sub, cand, exit=EXIT)
    if len(pn):
        print(f"  {lbl}: n={len(pn)} exp={pn.mean()*100:.4f}% wr={(pn>0).mean():.3f} sum={pn.sum()*100:.3f}% dates {pd.Timestamp(sub.fri_date.min()).date()}..{pd.Timestamp(sub.fri_date.max()).date()}")
    else:
        print(f"  {lbl}: n=0")

print("\n=== extra: per-trade test list (candidate) ===")
out = te_used[["fri_date","ticker","gap","entry_perp","_pnl"]].copy()
out["gap%"] = (out.gap*100).round(2); out["pnl%"] = (out._pnl*100).round(3)
print(out[["fri_date","ticker","gap%","pnl%"]].to_string(index=False))
