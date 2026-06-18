import wg_lib as wg
import numpy as np
import pandas as pd

# ---- reference config: fade |gap|>=2% ----
def fade2(r):
    if abs(r.gap) >= 0.02:
        return -1 if r.gap > 0 else 1
    return 0

def fade2_inv(r):
    s = fade2(r)
    return -s

EXIT = "p_mon0930"
ENTRY = "p_mon0630"
DT = False  # drop_thin=False (claimed n=71). Will also report True.

def fmt(m, tag):
    return (f"[{tag}] train_n={m['train_n']} train_exp={m['train_exp']*100:.3f}% | "
            f"test_n={m['test_n']} test_exp={m['test_exp']*100:.3f}% "
            f"test_wr={m['test_wr']*100:.1f}% test_sharpe={m['test_sharpe']:.3f} "
            f"test_p={m['test_p']:.4f}")

print("="*78)
print("STEP 0: REPRODUCE claimed config (entry=p_mon0630, exit=p_mon0930)")
print("="*78)
for dt in (False, True):
    d = wg.load(only_regular_monday=True, drop_thin=dt)
    d = wg.add_features(d, entry=ENTRY)
    m = wg.walk_forward(d, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kf = wg.kfold(d, fade2, exit=EXIT)
    print(fmt(m, f"drop_thin={dt}"))
    print(f"   kfold={[round(x*100,3) for x in kf]} cut={m['cut_date']}")

# main dataset for the rest
d = wg.load(only_regular_monday=True, drop_thin=DT)
d = wg.add_features(d, entry=ENTRY)

print("\n" + "="*78)
print("STEP 1: INVERSION (must be significantly <0 to prove edge is not noise)")
print("="*78)
mi = wg.walk_forward(d, fade2_inv, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
print(fmt(mi, "INVERTED"))
# p that inverted mean < 0: bootstrap_p gives P(mean<=0). For inversion we want test_exp<0 AND significant.
pn_inv = wg.pnl_series(d[d.fri_date >= np.sort(d.fri_date.unique())[int(len(np.sort(d.fri_date.unique()))*0.6)]],
                       fade2_inv, exit=EXIT, cost=wg.RT_COST)
# one-sided p that inverted edge < 0  ==  P(mean>=0) under bootstrap
arr = np.asarray(pn_inv, float)
rng = np.random.default_rng(0)
means = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1) if len(arr)>=5 else np.array([0.0])
p_inv_neg = float((means >= 0).mean())
print(f"   inverted test_exp={mi['test_exp']*100:.3f}%  p(inv_mean>=0)={p_inv_neg:.4f}  (want <0.05 => significantly negative)")

print("\n" + "="*78)
print("STEP 2: OFFSET entry to neighbor decision moments (edge must survive)")
print("="*78)
# DECISION_POINTS order: p_sun2000,p_mon0000,p_mon0600,p_mon0630,p_mon0800,p_mon0900
# p_mon0630 neighbors: p_mon0600 (-1) and p_mon0800 (+1)
for off_entry in ("p_mon0600", "p_mon0630", "p_mon0800"):
    do = wg.load(only_regular_monday=True, drop_thin=DT)
    do = wg.add_features(do, entry=off_entry)
    mo = wg.walk_forward(do, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kfo = wg.kfold(do, fade2, exit=EXIT)
    print(fmt(mo, f"entry={off_entry}"))
    print(f"   kfold={[round(x*100,3) for x in kfo]}")

print("\n" + "="*78)
print("STEP 3: TICKER LEAVE-ONE-OUT (drop top-2 by |test contribution|)")
print("="*78)
dates = np.sort(d.fri_date.unique())
cut = dates[int(len(dates)*0.6)]
te = d[d.fri_date >= cut].copy()
pn = wg.pnl_series(te, fade2, exit=EXIT, cost=wg.RT_COST)
te_traded = te.loc[pn.index].copy()
te_traded["pnl"] = pn.values
contrib = te_traded.groupby("ticker")["pnl"].agg(["sum","count","mean"])
contrib_abs = contrib.reindex(contrib["sum"].abs().sort_values(ascending=False).index)
print("Top tickers by |test contribution| (pp):")
print((contrib_abs.assign(sum_pp=contrib_abs["sum"]*100, mean_pp=contrib_abs["mean"]*100)
       [["count","sum_pp","mean_pp"]]).head(8).to_string())
print(f"\nbaseline test_sum={pn.sum()*100:.3f}pp n={len(pn)} exp={pn.mean()*100:.4f}% test_p(base)={wg.bootstrap_p(pn):.4f}")

top1, top2 = contrib_abs.index[0], contrib_abs.index[1]
print(f"\nTop-1 by |contrib|: {top1} ({contrib.loc[top1,'sum']*100:.3f}pp)  Top-2: {top2} ({contrib.loc[top2,'sum']*100:.3f}pp)")

def loo(exclude):
    de = d[~d.ticker.isin(exclude)].copy()
    m = wg.walk_forward(de, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kf = wg.kfold(de, fade2, exit=EXIT)
    return m, kf

m1, kf1 = loo({top1})
print("\n-- LOO drop top-1 ("+top1+") --")
print(fmt(m1, "LOO-1")); print(f"   kfold={[round(x*100,3) for x in kf1]}")

m2, kf2 = loo({top1, top2})
print("\n-- LOO drop top-2 ("+top1+","+top2+") --")
print(fmt(m2, "LOO-2")); print(f"   kfold={[round(x*100,3) for x in kf2]}")

print("\n" + "="*78)
print("STEP 4: SUBPERIOD — split TEST in half by time, sign in both halves?")
print("="*78)
te_dates = np.sort(te.fri_date.unique())
mid = te_dates[len(te_dates)//2]
h1 = te[te.fri_date < mid]
h2 = te[te.fri_date >= mid]
for name, sub in (("TEST-H1", h1), ("TEST-H2", h2)):
    pns = wg.pnl_series(sub, fade2, exit=EXIT, cost=wg.RT_COST)
    if len(pns):
        print(f"[{name}] n={len(pns)} exp={pns.mean()*100:.3f}% wr={(pns>0).mean()*100:.1f}% "
              f"sum={pns.sum()*100:.3f}pp p={wg.bootstrap_p(pns):.4f}  dates {str(pd.Timestamp(te_dates[0]).date())}..{str(pd.Timestamp(mid).date())}")
    else:
        print(f"[{name}] n=0")
print(f"   (test split at {str(pd.Timestamp(mid).date())}, full test {len(te_dates)} weekends)")
