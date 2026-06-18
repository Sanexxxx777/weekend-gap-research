import wg_lib as wg
import numpy as np
import pandas as pd

EXIT = "p_mon0930"
def fade(r):
    return (-1 if r.gap > 0 else +1) if abs(r.gap) >= 0.02 else 0

print("="*90)
print("PART B: per-ticker concentration on TEST split, fade|gap|>=2% exit=p_mon0930")
print("="*90)
for entry in ["p_sun2000","p_mon0000","p_mon0600"]:
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=entry)
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates)*0.6)]
    te = d[d.fri_date >= cut].copy()
    pn = wg.pnl_series(te, fade, exit=EXIT)
    te_sig = te.loc[pn.index].copy()
    te_sig["pnl"] = pn.values
    contrib = te_sig.groupby("ticker")["pnl"].sum().sort_values()
    total = pn.sum()
    print(f"\n--- entry={entry}  test_n={len(pn)} test_sum={total*100:.2f}% test_exp={pn.mean()*100:.3f}% ---")
    top = contrib.abs().sort_values(ascending=False).head(5)
    for tk in top.index:
        share = contrib[tk]/total*100 if total != 0 else 0
        print(f"  {tk:<8} sum={contrib[tk]*100:+.2f}%  share_of_total={share:+.1f}%  n={int((te_sig.ticker==tk).sum())}")
    # leave-one-out top contributor
    topk = contrib.abs().idxmax()
    pn_loo = pn[te_sig.ticker != topk]
    print(f"  LOO drop {topk}: test_exp={pn_loo.mean()*100:.3f}% (n={len(pn_loo)}), p={wg.bootstrap_p(pn_loo):.3f}")

print()
print("="*90)
print("PART C: inversion test (momentum instead of fade) on best honest entries")
print("="*90)
def momo(r):
    return (+1 if r.gap > 0 else -1) if abs(r.gap) >= 0.02 else 0
for entry in ["p_sun2000","p_mon0000","p_mon0600"]:
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=entry)
    m = wg.walk_forward(d, momo, exit=EXIT)
    print(f"  {entry:<12} INVERTED test_exp={m['test_exp']*100:+.3f}% (fade should be the profitable side)")
