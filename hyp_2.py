import wg_lib as wg
import numpy as np
import pandas as pd

d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry="p_mon0630")

def make_fade(th):
    t = th / 100.0
    def sig(r):
        if abs(r.gap) >= t:
            return -1 if r.gap > 0 else 1
        return 0
    return sig

def make_follow(th):
    t = th / 100.0
    def sig(r):
        if abs(r.gap) >= t:
            return 1 if r.gap > 0 else -1
        return 0
    return sig

# ----- Per-ticker concentration (leave-one-out) on TEST for fade thresholds 2.0 and 2.5 -----
def per_ticker_loo(th, exit="p_mon0930", train_frac=0.6):
    sig = make_fade(th)
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates)*train_frac)]
    te = d[d.fri_date >= cut].copy()
    pn = wg.pnl_series(te, sig, exit)
    te = te.loc[pn.index].copy()
    te["pnl"] = pn.values
    by = te.groupby("ticker")["pnl"].agg(["sum","count","mean"]).sort_values("sum", ascending=False)
    total = te["pnl"].sum()
    n = len(te)
    full_exp = te["pnl"].mean()
    print(f"\n--- FADE th={th}% exit={exit} TEST n={n} total_sum={total*100:.2f}% exp={full_exp*100:+.3f}% ---")
    print("top contributors by sum:")
    print((by.head(6)*np.array([100,1,100])).round(3))
    # leave-one-out by top contributor ticker
    print("leave-one-out (drop one ticker, recompute test exp):")
    for tk in by.head(3).index:
        sub = te[te.ticker != tk]
        print(f"  drop {tk:6s}: te_n={len(sub):3d} te_exp={sub['pnl'].mean()*100:+.3f}%")

per_ticker_loo(2.0)
per_ticker_loo(2.5)
per_ticker_loo(3.0)

# ----- Momentum band for big gaps at LONGER exits (the reference hint) -----
print("\n\n=== Does momentum (follow) appear for BIG gaps at LONGER holds? ===")
for th in [3.0, 4.0]:
    for exit in ["p_mon0930","p_mon1100","p_mon1200","p_mon1600"]:
        sig = make_follow(th)
        m = wg.walk_forward(d, sig, exit=exit)
        kf = wg.kfold(d, sig, exit=exit)
        kfs = " ".join(f"{x*100:+.2f}" for x in kf)
        print(f"FOLLOW th={th}% exit={exit:11s}: tr_n={m['train_n']:2d} tr={m['train_exp']*100:+.3f}% te_n={m['test_n']:2d} te={m['test_exp']*100:+.3f}% wr={m['test_wr']*100:4.1f}% p={m['test_p']:.3f} kf=[{kfs}]")
    print()

# Also fade at longer holds for big gaps (does fade decay -> momentum?)
print("=== FADE big gaps at longer holds (does edge flip to momentum?) ===")
for th in [3.0, 4.0]:
    for exit in ["p_mon0930","p_mon1100","p_mon1200","p_mon1600"]:
        sig = make_fade(th)
        m = wg.walk_forward(d, sig, exit=exit)
        kf = wg.kfold(d, sig, exit=exit)
        kfs = " ".join(f"{x*100:+.2f}" for x in kf)
        print(f"FADE   th={th}% exit={exit:11s}: tr_n={m['train_n']:2d} tr={m['train_exp']*100:+.3f}% te_n={m['test_n']:2d} te={m['test_exp']*100:+.3f}% wr={m['test_wr']*100:4.1f}% p={m['test_p']:.3f} kf=[{kfs}]")
    print()
