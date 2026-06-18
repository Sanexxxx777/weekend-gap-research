import wg_lib as wg
import numpy as np
import pandas as pd

THR = 0.02
ENTRY = "p_mon0000"
EXIT = "p_mon0930"

def fade2(r):
    if r.gap >= THR:  return -1
    if r.gap <= -THR: return +1
    return 0

def inv_fade2(r):
    return -fade2(r)

# ---- baseline reproduce ----
d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry=ENTRY)
m = wg.walk_forward(d, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
kf = wg.kfold(d, fade2, exit=EXIT)
print("=== BASELINE reproduce (entry=p_mon0000, exit=p_mon0930) ===")
print(f"TRAIN exp={m['train_exp']*100:.3f}% n={m['train_n']} wr={m['train_wr']:.3f} sharpe={m['train_sharpe']:.3f}")
print(f"TEST  exp={m['test_exp']*100:.3f}% n={m['test_n']} wr={m['test_wr']:.3f} sharpe={m['test_sharpe']:.3f} p={m['test_p']:.4f}")
print(f"ALL   exp={m['all_exp']*100:.3f}% n={m['all_n']}")
print(f"cut={m['cut_date']}  kfold={[round(x*100,3) for x in kf]}")

# ---- 1. INVERSION ----
print("\n=== 1. INVERSION (should be significantly <0) ===")
mi = wg.walk_forward(d, inv_fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
# p that inverted mean > 0; for "significantly <0" we want test exp clearly negative
te_inv_pnl = wg.pnl_series(d[d.fri_date >= pd.Timestamp(m['cut_date'])], inv_fade2, EXIT, wg.RT_COST)
# one-sided p that inverted mean < 0 -> bootstrap fraction means>=0
arr = np.asarray(te_inv_pnl, float)
rng = np.random.default_rng(0)
p_inv_neg = 1.0
if len(arr) >= 5:
    means = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1)
    p_inv_neg = float((means >= 0).mean())
print(f"INV TEST exp={mi['test_exp']*100:.3f}% n={mi['test_n']} p(mean>=0)={p_inv_neg:.4f}")
print(f"  (baseline test exp was {m['test_exp']*100:.3f}%, expect ~ -that minus 2*cost asymmetry)")

# ---- 2. OFFSET: shift entry +/-1 decision moment ----
print("\n=== 2. OFFSET (shift entry to neighbor decision moment) ===")
DP = wg.DECISION_POINTS
idx = DP.index(ENTRY)
neighbors = []
if idx-1 >= 0: neighbors.append(DP[idx-1])
if idx+1 < len(DP): neighbors.append(DP[idx+1])
for ne in neighbors:
    dn = wg.load(only_regular_monday=True, drop_thin=False)
    dn = wg.add_features(dn, entry=ne)
    mn = wg.walk_forward(dn, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    kfn = wg.kfold(dn, fade2, exit=EXIT)
    print(f"entry={ne}: TEST exp={mn['test_exp']*100:.3f}% n={mn['test_n']} p={mn['test_p']:.4f} kfold={[round(x*100,2) for x in kfn]}")

# ---- 3. TICKER LOO: drop top-2 tickers by test_sum contribution ----
print("\n=== 3. TICKER LEAVE-ONE-OUT (drop top-2 contributors in TEST) ===")
cut = pd.Timestamp(m['cut_date'])
te = d[d.fri_date >= cut].copy()
pn_te = wg.pnl_series(te, fade2, EXIT, wg.RT_COST)
te_traded = te.loc[pn_te.index].copy()
te_traded["_pnl"] = pn_te.values
contrib = te_traded.groupby("ticker")["_pnl"].sum().sort_values(ascending=False)
print("Top contributors (test_sum by ticker):")
print((contrib.head(6)*100).round(3).to_string())
top2 = list(contrib.head(2).index)
print(f"Dropping top-2: {top2}")
d_loo = d[~d.ticker.isin(top2)].copy()
m_loo = wg.walk_forward(d_loo, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
kf_loo = wg.kfold(d_loo, fade2, exit=EXIT)
print(f"LOO TEST exp={m_loo['test_exp']*100:.3f}% n={m_loo['test_n']} wr={m_loo['test_wr']:.3f} p={m_loo['test_p']:.4f} kfold={[round(x*100,2) for x in kf_loo]}")

# ---- 4. SUBPERIOD: split test in half by time ----
print("\n=== 4. SUBPERIOD (split TEST in half by time) ===")
te_dates = np.sort(te.fri_date.unique())
mid = te_dates[len(te_dates)//2]
h1 = te[te.fri_date < mid]; h2 = te[te.fri_date >= mid]
for lab, sub in [("H1", h1), ("H2", h2)]:
    pn = wg.pnl_series(sub, fade2, EXIT, wg.RT_COST)
    if len(pn):
        print(f"{lab}: exp={pn.mean()*100:.3f}% n={len(pn)} wr={(pn>0).mean():.3f} sum={pn.sum()*100:.2f}% dates={pd.Timestamp(sub.fri_date.min()).date()}..{pd.Timestamp(sub.fri_date.max()).date()}")
    else:
        print(f"{lab}: no trades")

# ---- extra: how many distinct tickers/weekends contribute, concentration ----
print("\n=== EXTRA: test trade composition ===")
print(f"distinct tickers traded in test: {te_traded.ticker.nunique()}, distinct weekends: {te_traded.fri_date.nunique()}, total trades: {len(te_traded)}")
print(f"top-2 contributors sum = {contrib.head(2).sum()*100:.2f}% of total test_sum {pn_te.sum()*100:.2f}%")
