import wg_lib as wg
import numpy as np
import pandas as pd

ENTRY = "p_mon0000"
EXIT = "p_mon0930"
THR = 0.02

def fade(r):
    if abs(r.gap) >= THR:
        return -1 if r.gap > 0 else +1
    return 0

def fade_inv(r):
    s = fade(r)
    return -s

def load_feat(entry=ENTRY, drop_thin=False):
    d = wg.load(only_regular_monday=True, drop_thin=drop_thin)
    d = wg.add_features(d, entry=entry)
    return d

print("="*90)
print("BASELINE reproduce: fade |gap|>=2%, entry=p_mon0000, exit=p_mon0930, drop_thin=False")
print("="*90)
d = load_feat()
m = wg.walk_forward(d, fade, exit=EXIT)
kf = wg.kfold(d, fade, exit=EXIT)
print(f"train: n={m['train_n']} exp={m['train_exp']*100:.3f}% wr={m['train_wr']*100:.1f}% sum={m['train_sum']*100:.2f}% sharpe={m['train_sharpe']:.3f}")
print(f"test : n={m['test_n']} exp={m['test_exp']*100:.3f}% wr={m['test_wr']*100:.1f}% sum={m['test_sum']*100:.2f}% sharpe={m['test_sharpe']:.3f} p={m['test_p']:.4f}")
print(f"all  : n={m['all_n']} exp={m['all_exp']*100:.3f}%")
print(f"kfold: [{','.join(f'{x*100:+.3f}' for x in kf)}]  cut={m['cut_date']}")

print()
print("="*90)
print("TEST 1 INVERSION: invert champion signal -> expect significantly < 0 on test")
print("="*90)
mi = wg.walk_forward(d, fade_inv, exit=EXIT)
kfi = wg.kfold(d, fade_inv, exit=EXIT)
# bootstrap p for mean<0: equivalently 1 - p(mean>0). bootstrap_p gives frac(mean<=0).
pn_inv = wg.pnl_series(d[d.fri_date >= pd.Timestamp(m['cut_date'])], fade_inv, exit=EXIT)
p_inv_neg = 1.0 - wg.bootstrap_p(pn_inv)  # frac of bootstrap means > 0; we want this LOW (mean<0)
print(f"inv test: n={mi['test_n']} exp={mi['test_exp']*100:.3f}% wr={mi['test_wr']*100:.1f}% sum={mi['test_sum']*100:.2f}% p(mean>0)={p_inv_neg:.4f}")
print(f"inv kfold: [{','.join(f'{x*100:+.3f}' for x in kfi)}]")
print(f"-> inversion significantly negative? test_exp={mi['test_exp']*100:.3f}% (want clearly <0), bootstrap_p(mean<=0)={mi['test_p']:.4f} (want HIGH ~1)")

print()
print("="*90)
print("TEST 2 OFFSET: shift entry to neighbors of p_mon0000 -> edge should survive")
print("="*90)
NEIGH = ["p_sun2000", "p_mon0000", "p_mon0600"]
for e in NEIGH:
    de = load_feat(entry=e)
    me = wg.walk_forward(de, fade, exit=EXIT)
    kfe = wg.kfold(de, fade, exit=EXIT)
    kfs = "[" + ",".join(f"{x*100:+.2f}" for x in kfe) + "]"
    mark = " <-- champion" if e == ENTRY else ""
    print(f"{e:<12} train_n={me['train_n']:>3} train_exp={me['train_exp']*100:>7.3f}%  test_n={me['test_n']:>3} test_exp={me['test_exp']*100:>7.3f}% wr={me['test_wr']*100:>5.1f}% p={me['test_p']:.3f} kf={kfs}{mark}")

print()
print("="*90)
print("TEST 3 TICKER LOO: drop top-2 tickers by |contribution to test_sum| -> edge>0 & p?")
print("="*90)
cut = pd.Timestamp(m['cut_date'])
te = d[d.fri_date >= cut].copy()
pn = wg.pnl_series(te, fade, exit=EXIT)
te_traded = te.loc[pn.index].copy()
te_traded["_pnl"] = pn.values
contrib = te_traded.groupby("ticker")["_pnl"].sum().sort_values(ascending=False)
print("Top tickers by test_sum contribution (signed):")
print((contrib*100).round(3).to_string())
top2 = list(contrib.index[:2])
print(f"\nTop-2 positive contributors: {top2}")

# LOO: drop these tickers entirely, recompute walk_forward
d_loo = d[~d.ticker.isin(top2)].copy()
m_loo = wg.walk_forward(d_loo, fade, exit=EXIT)
kf_loo = wg.kfold(d_loo, fade, exit=EXIT)
print(f"after dropping {top2}:")
print(f"  test: n={m_loo['test_n']} exp={m_loo['test_exp']*100:.3f}% wr={m_loo['test_wr']*100:.1f}% sum={m_loo['test_sum']*100:.2f}% p={m_loo['test_p']:.4f}")
print(f"  kfold: [{','.join(f'{x*100:+.3f}' for x in kf_loo)}]")

# Also drop the 2 most extreme by abs contribution (could be a big loser too)
contrib_abs = te_traded.groupby("ticker")["_pnl"].sum().abs().sort_values(ascending=False)
top2_abs = list(contrib_abs.index[:2])
d_loo2 = d[~d.ticker.isin(top2_abs)].copy()
m_loo2 = wg.walk_forward(d_loo2, fade, exit=EXIT)
print(f"\nDrop top-2 by |contribution| {top2_abs}:")
print(f"  test: n={m_loo2['test_n']} exp={m_loo2['test_exp']*100:.3f}% p={m_loo2['test_p']:.4f}")

print()
print("="*90)
print("TEST 4 SUBPERIOD: split TEST in half by time -> sign in both halves?")
print("="*90)
te_dates = np.sort(te.fri_date.unique())
mid = te_dates[len(te_dates)//2]
h1 = te[te.fri_date < mid]; h2 = te[te.fri_date >= mid]
for lab, sub in [("test_H1", h1), ("test_H2", h2)]:
    pns = wg.pnl_series(sub, fade, exit=EXIT)
    if len(pns):
        print(f"{lab}: n={len(pns)} exp={pns.mean()*100:.3f}% wr={(pns>0).mean()*100:.1f}% sum={pns.sum()*100:.2f}% p={wg.bootstrap_p(pns):.4f}  dates {pd.Timestamp(sub.fri_date.min()).date()}..{pd.Timestamp(sub.fri_date.max()).date()}")
    else:
        print(f"{lab}: n=0")
