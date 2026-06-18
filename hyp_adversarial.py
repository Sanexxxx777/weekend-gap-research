import wg_lib as wg
import numpy as np
import pandas as pd

EXIT = "p_mon0930"
COST = wg.RT_COST

def sig_long(r):
    return 1 if r.gap <= -0.02 else 0

def sig_inv(r):
    # inversion: flip the side of the candidate signal
    return -1 if r.gap <= -0.02 else 0

def report(tag, d, signal_fn):
    m = wg.walk_forward(d, signal_fn, exit=EXIT, cost=COST, train_frac=0.6)
    kf = wg.kfold(d, signal_fn, exit=EXIT)
    print(f"=== {tag} ===")
    print(f"  TRAIN exp={m['train_exp']*100:+.3f}% n={m['train_n']}  TEST exp={m['test_exp']*100:+.3f}% n={m['test_n']} wr={m['test_wr']:.2f} p={m['test_p']:.4f}")
    print(f"  kfold={[round(x*100,3) for x in kf]}")
    return m, kf

print("############ BASELINE ############")
d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry="p_mon0630")
mb, kfb = report("BASELINE LONG fade gap<=-2% entry=p_mon0630 exit=p_mon0930", d, sig_long)

print("\n############ 1. INVERSION ############")
mi, kfi = report("INVERTED (short instead of long)", d, sig_inv)
# Inversion of pure-long is short on same rows. Net pnl = -(raw)-cost, NOT exactly -baseline (cost double-charged both ways).
# Better check: inverted test mean should be significantly < 0.
print(f"  -> inversion test_exp = {mi['test_exp']*100:+.3f}% (want significantly < 0; p that mean>0 irrelevant, check mean<0)")

print("\n############ 2. OFFSET (entry +-1 decision point) ############")
# DECISION_POINTS order: p_sun2000, p_mon0000, p_mon0600, p_mon0630, p_mon0800, p_mon0900
# p_mon0630 neighbors: p_mon0600 (-1), p_mon0800 (+1)
for entry in ["p_mon0600", "p_mon0800"]:
    de = wg.load(only_regular_monday=True, drop_thin=False)
    de = wg.add_features(de, entry=entry)
    report(f"OFFSET entry={entry}", de, sig_long)

print("\n############ 3. TICKER LEAVE-ONE-OUT (top-2 by test_sum) ############")
# Build test-set pnl per ticker to find top contributors
dates = np.sort(d.fri_date.unique())
cut = dates[int(len(dates)*0.6)]
te = d[d.fri_date >= cut].copy()
pn_te = wg.pnl_series(te, sig_long, exit=EXIT, cost=COST)
te_sel = te.loc[pn_te.index].copy()
te_sel["_pnl"] = pn_te.values
contrib = te_sel.groupby("ticker")["_pnl"].agg(["sum","count"]).sort_values("sum", ascending=False)
print("  Test-set pnl contribution by ticker:")
print(contrib.to_string())
top2 = list(contrib.index[:2])
print(f"  Top-2 tickers by test_sum: {top2}")

d_loo = d[~d.ticker.isin(top2)].copy()
mloo, kfloo = report(f"LOO drop top-2 {top2}", d_loo, sig_long)

print("\n############ 4. SUBPERIOD (split test in half by time) ############")
te_dates = np.sort(te.fri_date.unique())
mid = te_dates[len(te_dates)//2]
te_h1 = te[te.fri_date < mid]
te_h2 = te[te.fri_date >= mid]
for half, sub in [("H1", te_h1), ("H2", te_h2)]:
    pn = wg.pnl_series(sub, sig_long, exit=EXIT, cost=COST)
    if len(pn):
        print(f"  {half}: exp={pn.mean()*100:+.3f}% n={len(pn)} wr={(pn>0).mean():.2f} sum={pn.sum()*100:+.3f}% (dates {pd.Timestamp(te_dates[0]).date()}..{pd.Timestamp(mid).date()}..{pd.Timestamp(te_dates[-1]).date()})")
    else:
        print(f"  {half}: empty")

print("\n############ EXTRA: bootstrap p after dropping top-2 (test set only) ############")
te_loo = d_loo[d_loo.fri_date >= cut]
pn_loo = wg.pnl_series(te_loo, sig_long, exit=EXIT, cost=COST)
print(f"  test_n after LOO={len(pn_loo)} exp={pn_loo.mean()*100:+.3f}% p={wg.bootstrap_p(pn_loo):.4f}")

print("\n############ EXTRA: how many distinct tickers / mondays in test ############")
print(f"  test rows={len(te_sel)} distinct tickers={te_sel.ticker.nunique()} distinct mondays={te_sel.fri_date.nunique()}")
