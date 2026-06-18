"""
Hypothesis [sizing-weight]:
  Fix best base signal (fade idio_gap vs fade gap, entry=p_mon0630, exit=p_mon0930).
  Compare equal-weight vs weighted position: w ~ |idio_gap| or w ~ (1-explained_frac).
  Weighted expectancy = sum(w*pnl)/sum(w) on train/test.
  Does concentration on more "phantom" gaps improve OOS?

Honesty:
  - thresholds chosen on TRAIN only.
  - weight is decision-info (|idio_gap|, 1-explained_frac, |gap|) -> NOT look-ahead.
  - PnL/split come from wg_lib (source of truth). I reuse pnl_series + replicate
    walk_forward's train/test cut so weighted exp aligns with equal-weight exp.
"""
import wg_lib as wg
import numpy as np
import pandas as pd

ENTRY = "p_mon0630"
EXIT = "p_mon0930"

# --- base signals (decision-fields only) ---
def fade_gap(thr):
    def f(r):
        if not np.isfinite(r.gap): return 0
        if abs(r.gap) < thr: return 0
        return -1 if r.gap > 0 else 1   # fade: short positive gap, long negative gap
    return f

def fade_idio(thr):
    def f(r):
        if not np.isfinite(r.idio_gap): return 0
        if abs(r.idio_gap) < thr: return 0
        return -1 if r.idio_gap > 0 else 1
    return f

# --- weight functions (decision-info, applied to the SAME selected trades) ---
def w_equal(r):           return 1.0
def w_abs_idio(r):        return abs(r.idio_gap) if np.isfinite(r.idio_gap) else 0.0
def w_one_minus_expl(r):  # (1-explained_frac) = |idio|/|gap| -> phantom share. clip to [0,inf)
    ef = r.explained_frac
    if not np.isfinite(ef): return 0.0
    return max(0.0, 1.0 - ef)
def w_abs_gap(r):         return abs(r.gap) if np.isfinite(r.gap) else 0.0


def weighted_exp(sub, signal_fn, weight_fn, exit=EXIT):
    """sum(w*pnl)/sum(w) reusing wg.pnl_series (source-of-truth PnL)."""
    pn = wg.pnl_series(sub, signal_fn, exit, wg.RT_COST)   # net per-trade pnl, index aligned to sub
    if len(pn) == 0:
        return np.nan, 0
    w = sub.loc[pn.index].apply(weight_fn, axis=1).astype(float)
    sw = w.sum()
    if sw <= 0:
        return np.nan, len(pn)
    return float((w * pn).sum() / sw), int(len(pn))


def split(d, train_frac=0.6):
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates) * train_frac)]
    return d[d.fri_date < cut], d[d.fri_date >= cut], str(pd.Timestamp(cut).date())


def report(name, d, signal_fn, weight_fns):
    tr, te, cut = split(d)
    print(f"\n=== {name} (cut={cut}) ===")
    # equal-weight reference via wg.walk_forward
    wf = wg.walk_forward(d, signal_fn, exit=EXIT, train_frac=0.6)
    print(f"  [wg.walk_forward EW] train_exp={wf['train_exp']*100:.3f}% n={wf['train_n']} | "
          f"test_exp={wf['test_exp']*100:.3f}% n={wf['test_n']} wr={wf['test_wr']*100:.1f}% p={wf['test_p']:.3f}")
    for wname, wfn in weight_fns:
        tr_e, tr_n = weighted_exp(tr, signal_fn, wfn)
        te_e, te_n = weighted_exp(te, signal_fn, wfn)
        all_e, all_n = weighted_exp(d, signal_fn, wfn)
        def fmt(x): return f"{x*100:.3f}%" if np.isfinite(x) else "nan"
        print(f"  [{wname:>14}] train_w_exp={fmt(tr_e)} (n={tr_n}) | test_w_exp={fmt(te_e)} (n={te_n}) | all_w_exp={fmt(all_e)}")
    return wf


if __name__ == "__main__":
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=ENTRY)
    d_thin = wg.load(only_regular_monday=True, drop_thin=True)
    d_thin = wg.add_features(d_thin, entry=ENTRY)

    print("data: rows total=%d | rows w/ idio=%d | dates total=%d | dates w/ idio=%d"
          % (len(d), d.idio_gap.notna().sum(), d.fri_date.nunique(), d[d.idio_gap.notna()].fri_date.nunique()))

    # ---- BASE A: fade gap, threshold from reference (|gap|>=2%) ----
    sig_gap = fade_gap(0.02)
    report("BASE fade |gap|>=2% (full data, drop_thin=False)", d, sig_gap,
           [("equal", w_equal), ("|gap|", w_abs_gap), ("|idio_gap|", w_abs_idio), ("1-expl_frac", w_one_minus_expl)])
    report("BASE fade |gap|>=2% (drop_thin=True)", d_thin, sig_gap,
           [("equal", w_equal), ("|gap|", w_abs_gap), ("|idio_gap|", w_abs_idio), ("1-expl_frac", w_one_minus_expl)])

    # ---- BASE B: fade idio_gap (only 9 dates have idio) ----
    di = d[d.idio_gap.notna()].copy()
    print("\n[fade idio subset] rows=%d dates=%d range=%s..%s"
          % (len(di), di.fri_date.nunique(), di.fri_date.min().date(), di.fri_date.max().date()))
    sig_idio = fade_idio(0.02)
    report("BASE fade |idio_gap|>=2% (idio subset)", di, sig_idio,
           [("equal", w_equal), ("|idio_gap|", w_abs_idio), ("1-expl_frac", w_one_minus_expl)])
