"""
Deeper checks on the surviving candidate:
  BASE = fade |gap|>=2%, entry=p_mon0630, exit=p_mon0930, weight w=|gap| (decision-info).
Checks:
  1. Weight chosen on TRAIN: is |gap|-weight > equal on TRAIN? (honesty: pick on train)
  2. kfold weighted (3 time blocks) - sign stability
  3. per-ticker concentration: leave-one-out by top weight contribution
  4. inversion test: invert signal -> weighted test_exp should flip negative
  5. compare graded weight (|gap|/thr capped) too
"""
import wg_lib as wg
import numpy as np
import pandas as pd

ENTRY, EXIT = "p_mon0630", "p_mon0930"

def fade_gap(thr=0.02):
    def f(r):
        if not np.isfinite(r.gap) or abs(r.gap) < thr: return 0
        return -1 if r.gap > 0 else 1
    return f

def invert(sig):
    return lambda r: -sig(r)

def w_equal(r): return 1.0
def w_abs_gap(r): return abs(r.gap) if np.isfinite(r.gap) else 0.0

def split(d, train_frac=0.6):
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates)*train_frac)]
    return d[d.fri_date < cut], d[d.fri_date >= cut], str(pd.Timestamp(cut).date())

def wexp(sub, sig, wfn, exit=EXIT):
    pn = wg.pnl_series(sub, sig, exit, wg.RT_COST)
    if len(pn)==0: return np.nan, 0
    w = sub.loc[pn.index].apply(wfn, axis=1).astype(float)
    sw = w.sum()
    if sw<=0: return np.nan, len(pn)
    return float((w*pn).sum()/sw), int(len(pn))

def wkfold(d, sig, wfn, exit=EXIT, k=3):
    dates = np.sort(d.fri_date.unique())
    blocks = np.array_split(dates, k)
    out=[]
    for b in blocks:
        sub = d[d.fri_date.isin(b)]
        e,_ = wexp(sub, sig, wfn, exit)
        out.append(e if np.isfinite(e) else 0.0)
    return out

if __name__=="__main__":
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=ENTRY)
    sig = fade_gap(0.02)
    tr, te, cut = split(d)

    print("=== candidate: fade|gap|>=2%, w=|gap|, exit=p_mon0930 (full data) ===")
    print("cut:", cut)
    for nm, wfn in [("equal",w_equal),("|gap|",w_abs_gap)]:
        tre,trn = wexp(tr,sig,wfn); tee,ten = wexp(te,sig,wfn)
        print(f"  {nm:>6}: train={tre*100:.3f}% (n={trn}) test={tee*100:.3f}% (n={ten})")

    print("\n--- kfold (3 time blocks) ---")
    print("  equal :", [round(x*100,3) for x in wkfold(d,sig,w_equal)])
    print("  |gap| :", [round(x*100,3) for x in wkfold(d,sig,w_abs_gap)])

    print("\n--- inversion test (invert signal) ---")
    isig = invert(sig)
    tee_eq,_ = wexp(te,isig,w_equal); tee_w,_ = wexp(te,isig,w_abs_gap)
    print(f"  inverted test equal={tee_eq*100:.3f}%  |gap|={tee_w*100:.3f}%")

    print("\n--- per-ticker concentration (TEST, w=|gap|) ---")
    pn = wg.pnl_series(te, sig, EXIT, wg.RT_COST)
    sub = te.loc[pn.index].copy()
    sub["pnl"]=pn.values
    sub["w"]=sub.apply(w_abs_gap,axis=1)
    sub["wp"]=sub.w*sub.pnl
    base_exp = sub.wp.sum()/sub.w.sum()
    print(f"  base weighted test_exp={base_exp*100:.3f}%  n={len(sub)}")
    # contribution share by ticker
    contrib = sub.groupby("ticker").agg(w=("w","sum"), wp=("wp","sum"), n=("pnl","size"))
    contrib["w_share_pct"]=100*contrib.w/sub.w.sum()
    contrib["contrib_to_exp_pct"]=100*contrib.wp/sub.wp.sum()
    print(contrib.sort_values("contrib_to_exp_pct",ascending=False).head(8).round(4).to_string())
    print("\n  leave-one-out (drop top contributor tickers one by one):")
    for tk in contrib.sort_values("contrib_to_exp_pct",ascending=False).head(5).index:
        s2 = sub[sub.ticker!=tk]
        e2 = s2.wp.sum()/s2.w.sum()
        print(f"    drop {tk:>6}: test_w_exp={e2*100:.3f}% (n={len(s2)})")

    print("\n--- offset sanity: also try equal-weight EW p-value via wg ---")
    wf = wg.walk_forward(d, sig, exit=EXIT, train_frac=0.6)
    print(f"  EW test_exp={wf['test_exp']*100:.3f}% p={wf['test_p']:.3f} wr={wf['test_wr']*100:.1f}% n={wf['test_n']}")
