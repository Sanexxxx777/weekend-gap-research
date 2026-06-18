"""
Adversarial verification of candidate [sizing-weight]:
  base: fade |gap|>=2%, entry=p_mon0630, exit=p_mon0930
  sizing: weighted_exp = sum(w*pnl)/sum(w), w=|gap| (decision-info), SAME selected trades.
  Claimed: TRAIN 0.291% -> TEST 0.821% (n=71, p=0.0004), kfold=[-0.189,1.189,0.333]

Tests: reproduce, then INVERSION / OFFSET / TICKER LOO / SUBPERIOD.
All PnL via wg.pnl_series (source of truth). Weighted bootstrap p is computed on the
weighted-contribution series so it matches the weighted-mean estimand.
"""
import wg_lib as wg
import numpy as np, pandas as pd

ENTRY = "p_mon0630"
EXIT = "p_mon0930"

def fade_gap(thr=0.02):
    def f(r):
        if not np.isfinite(r.gap): return 0
        if abs(r.gap) < thr: return 0
        return -1 if r.gap > 0 else 1
    return f

def fade_gap_inverted(thr=0.02):
    def f(r):
        if not np.isfinite(r.gap): return 0
        if abs(r.gap) < thr: return 0
        return 1 if r.gap > 0 else -1   # momentum: same trades, flipped side
    return f

def w_abs_gap(r):
    return abs(r.gap) if np.isfinite(r.gap) else 0.0

def split(d, train_frac=0.6):
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates)*train_frac)]
    return d[d.fri_date < cut], d[d.fri_date >= cut], pd.Timestamp(cut)

def weighted_exp(sub, signal_fn, exit=EXIT, weight_fn=w_abs_gap):
    pn = wg.pnl_series(sub, signal_fn, exit, wg.RT_COST)
    if len(pn) == 0:
        return np.nan, 0, pd.Series([], dtype=float)
    w = sub.loc[pn.index].apply(weight_fn, axis=1).astype(float)
    sw = w.sum()
    if sw <= 0:
        return np.nan, len(pn), pd.Series([], dtype=float)
    we = float((w*pn).sum()/sw)
    # contribution series whose simple mean == weighted mean: c_i = w_i*pn_i*N/sum(w)
    contrib = (w*pn) * (len(pn)/sw)
    return we, int(len(pn)), contrib

def boot_p(contrib, n=4000, seed=0):
    x = np.asarray(contrib, float)
    if len(x) < 5: return 1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float((means <= 0).mean())

def boot_p_neg(contrib, n=4000, seed=0):
    """p that mean >= 0 (for inversion: want significantly <0)."""
    x = np.asarray(contrib, float)
    if len(x) < 5: return 1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float((means >= 0).mean())

def run_we(d, signal_fn, entry=ENTRY, exit=EXIT, label=""):
    tr, te, cut = split(d)
    tr_e, tr_n, tr_c = weighted_exp(tr, signal_fn, exit)
    te_e, te_n, te_c = weighted_exp(te, signal_fn, exit)
    p = boot_p(te_c)
    print(f"  {label}: cut={cut.date()} TRAIN w_exp={tr_e*100:.3f}% n={tr_n} | "
          f"TEST w_exp={te_e*100:.3f}% n={te_n} p={p:.4f}")
    return dict(tr_e=tr_e, te_e=te_e, te_n=te_n, te_c=te_c, p=p, cut=cut)

if __name__ == "__main__":
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=ENTRY)
    sig = fade_gap(0.02)

    print("=== 0. REPRODUCE candidate (w=|gap|, entry=p_mon0630, exit=p_mon0930) ===")
    base = run_we(d, sig, label="reproduce")
    kf = wg.kfold(d, sig, exit=EXIT)  # equal-weight kfold for reference
    # weighted kfold
    dates = np.sort(d.fri_date.unique())
    blocks = np.array_split(dates, 3)
    wk = []
    for b in blocks:
        sub = d[d.fri_date.isin(b)]
        e,_,_ = weighted_exp(sub, sig, EXIT)
        wk.append(round(e*100,3) if np.isfinite(e) else None)
    print(f"  weighted kfold (%): {wk}   | equal kfold (%): {[round(x*100,3) for x in kf]}")

    print("\n=== 1. INVERSION (flip side, same trades, w=|gap|) -> expect TEST significantly <0 ===")
    inv = run_we(d, fade_gap_inverted(0.02), label="inverted")
    p_neg = boot_p_neg(inv["te_c"])
    print(f"  inverted TEST w_exp={inv['te_e']*100:.3f}% p(mean>=0)={p_neg:.4f}")

    print("\n=== 2. OFFSET entry +/-1 decision-moment ===")
    # p_mon0630 neighbors: p_mon0600 (-1), p_mon0800 (+1)
    for off_entry in ["p_mon0600", "p_mon0800"]:
        d2 = wg.load(only_regular_monday=True, drop_thin=False)
        d2 = wg.add_features(d2, entry=off_entry)
        run_we(d2, fade_gap(0.02), entry=off_entry, label=f"entry={off_entry}")

    print("\n=== 3. TICKER LOO (drop top-2 tickers by |contribution| to test weighted sum) ===")
    tr, te, cut = split(d)
    te_pn = wg.pnl_series(te, sig, EXIT, wg.RT_COST)
    te_use = te.loc[te_pn.index].copy()
    te_use["_pn"] = te_pn.values
    te_use["_w"] = te_use.apply(w_abs_gap, axis=1)
    te_use["_contrib"] = te_use["_w"] * te_use["_pn"]
    by_tkr = te_use.groupby("ticker")["_contrib"].sum().sort_values(ascending=False)
    print("  top contributors to TEST weighted sum:")
    print(by_tkr.head(5).to_string())
    top2 = list(by_tkr.head(2).index)
    print(f"  dropping: {top2}")
    d_loo = d[~d.ticker.isin(top2)].copy()
    run_we(d_loo, sig, label=f"LOO drop {top2}")

    print("\n=== 4. SUBPERIOD: split TEST in half by time ===")
    te_dates = np.sort(te.fri_date.unique())
    mid = te_dates[len(te_dates)//2]
    te_h1 = te[te.fri_date < mid]; te_h2 = te[te.fri_date >= mid]
    for nm, sub in [("test_H1", te_h1), ("test_H2", te_h2)]:
        e, nn, cc = weighted_exp(sub, sig, EXIT)
        pp = boot_p(cc)
        print(f"  {nm}: w_exp={e*100:.3f}% n={nn} p={pp:.4f} dates={sub.fri_date.nunique()}")
