import wg_lib as wg
import numpy as np
import pandas as pd

# fade large gaps: short positive gap, long negative gap, |gap|>=2%
def fade_signal(r):
    if abs(r.gap) >= 0.02:
        return -1 if r.gap > 0 else 1
    return 0

ENTRY = "p_mon0630"
EXIT = "p_mon0930"

# Load ALL weekends, add features at entry
d_all = wg.load(only_regular_monday=False, drop_thin=False)
d_all = wg.add_features(d_all, entry=ENTRY)

reg = d_all[d_all.is_holiday_mon == 0].copy()
hol = d_all[d_all.is_holiday_mon == 1].copy()

def summarize(sub, name):
    print(f"\n===== {name} (rows after add_features: {len(sub)}, weekends: {sub.fri_date.nunique()}) =====")
    pn = wg.pnl_series(sub, fade_signal, exit=EXIT, cost=wg.RT_COST)
    n = len(pn)
    if n == 0:
        print("  no trades")
        return
    exp = pn.mean(); wr = (pn > 0).mean(); ssum = pn.sum()
    p = wg.bootstrap_p(pn)
    print(f"  ALL: n={n}  exp={exp*100:.3f}%  wr={wr*100:.1f}%  sum={ssum*100:.2f}%  p={p:.4f}")
    # weekends triggering
    trig = sub[sub.gap.abs() >= 0.02]
    print(f"  triggering rows: {len(trig)} across {trig.fri_date.nunique()} weekends")
    # walk_forward
    m = wg.walk_forward(sub, fade_signal, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
    print(f"  TRAIN: n={m['train_n']} exp={m['train_exp']*100:.3f}% wr={m['train_wr']*100:.1f}%")
    print(f"  TEST : n={m['test_n']} exp={m['test_exp']*100:.3f}% wr={m['test_wr']*100:.1f}% p={m['test_p']:.4f} cut={m['cut_date']}")
    # kfold
    kf = wg.kfold(sub, fade_signal, exit=EXIT)
    print(f"  KFOLD exp (%): {[round(x*100,3) for x in kf]}")
    return pn, trig

summarize(reg, "REGULAR Monday")
summarize(hol, "HOLIDAY Monday")

# Holiday breakdown per weekend and per-ticker concentration
print("\n\n===== HOLIDAY detail (n=3 weekends only) =====")
pn_hol = wg.pnl_series(hol, fade_signal, exit=EXIT, cost=wg.RT_COST)
hol_trig = hol[hol.gap.abs() >= 0.02].copy()
hol_trig = hol_trig.loc[pn_hol.index]
hol_trig["pnl"] = pn_hol
for fd, g in hol_trig.groupby("fri_date"):
    print(f"  weekend {fd.date()}: n={len(g)} exp={g.pnl.mean()*100:.3f}% wr={(g.pnl>0).mean()*100:.0f}% sum={g.pnl.sum()*100:.2f}%")

print("\n  Per-ticker contribution (holiday):")
tk = hol_trig.groupby("ticker").pnl.agg(["count", "sum", "mean"]).sort_values("sum")
print(tk.tail(5).to_string())
print("  ...")
print(tk.head(5).to_string())

# leave-one-out top contributor
print("\n  Leave-one-out by top |sum| ticker (holiday):")
top_tk = tk.sum().name if False else tk["sum"].abs().idxmax()
print(f"  top contributor ticker: {top_tk} (sum={tk.loc[top_tk,'sum']*100:.2f}%)")
loo = hol_trig[hol_trig.ticker != top_tk]
if len(loo):
    print(f"  without {top_tk}: n={len(loo)} exp={loo.pnl.mean()*100:.3f}% wr={(loo.pnl>0).mean()*100:.0f}%")

# Inversion sanity on holiday
def inv_signal(r):
    s = fade_signal(r)
    return -s
pn_inv = wg.pnl_series(hol, inv_signal, exit=EXIT, cost=wg.RT_COST)
print(f"\n  INVERSION holiday: n={len(pn_inv)} exp={pn_inv.mean()*100:.3f}% wr={(pn_inv>0).mean()*100:.1f}%")

# Inversion on regular test for reference
m_reg = wg.walk_forward(reg, inv_signal, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
print(f"  INVERSION regular TEST: exp={m_reg['test_exp']*100:.3f}%")
