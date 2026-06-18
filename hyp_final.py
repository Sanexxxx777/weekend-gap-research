import wg_lib as wg, numpy as np, pandas as pd
pd.set_option('display.width', 220)

THIN = wg.THIN
def make_fade(thr, long_bias=False, drop_thin=False):
    def fn(r):
        if drop_thin and r.ticker in THIN: return 0
        if abs(r.gap) < thr: return 0
        if r.gap > 0:
            return 0 if long_bias else -1
        return 1
    return fn

# TRAIN-BEST combo (selected ONLY on train above): entry=p_mon0000, thr=0.025, long_bias=True, exit=p_mon0930
ENTRY="p_mon0000"; EXIT="p_mon0930"; THR=0.025
d = wg.add_features(wg.load(True,False), entry=ENTRY)

def report(name, fn):
    m = wg.walk_forward(d, fn, exit=EXIT)
    kf = wg.kfold(d, fn, exit=EXIT)
    print(f"\n=== {name} ===")
    print("train_exp %.4f%% n=%d wr=%.1f%% | TEST_exp %.4f%% n=%d wr=%.1f%% p=%.4f | all_exp %.4f%%"
          % (m['train_exp']*100,m['train_n'],m['train_wr']*100,
             m['test_exp']*100,m['test_n'],m['test_wr']*100,m['test_p'],m['all_exp']*100))
    print("kfold(full):", [round(x*100,4) for x in kf])
    return m

# 1) The chosen combo (long_bias)
fn_best = make_fade(THR, long_bias=True, drop_thin=False)
m_best = report("CHOSEN COMBO  entry=p_mon0000 thr=2.5% long_bias exit=p_mon0930", fn_best)

# 2) Conservative both-sided (robustness reference, same entry/thr)
fn_both = make_fade(THR, long_bias=False, drop_thin=False)
report("both-sided  entry=p_mon0000 thr=2.5% exit=p_mon0930", fn_both)

# 3) with liquidity filter (drop_thin) — same combo
fn_lt = make_fade(THR, long_bias=True, drop_thin=True)
report("CHOSEN + drop_thin liquidity filter", fn_lt)

# --- per-ticker concentration on TEST of chosen combo ---
dates=np.sort(d.fri_date.unique()); cut=dates[int(len(dates)*0.6)]
te=d[d.fri_date>=cut]
pn=wg.pnl_series(te, fn_best, exit=EXIT)
tk=te.loc[pn.index].ticker
print("\n--- TEST per-ticker contribution (chosen combo) ---")
contrib = pd.DataFrame({'pnl':pn.values,'ticker':tk.values}).groupby('ticker').agg(n=('pnl','size'),sum=('pnl','sum'),mean=('pnl','mean')).sort_values('sum',ascending=False)
print(contrib.to_string())
print("TEST total n=%d sum=%.4f exp=%.5f" % (len(pn),pn.sum(),pn.mean()))

# leave-one-out top contributor
top=contrib.index[0]
keep=tk!=top
print(f"\nLeave-out top contributor {top}: remaining n={keep.sum()} exp={pn[keep.values].mean():.5f} sum={pn[keep.values].sum():.4f}")
top2=contrib.index[:2].tolist()
keep2=~tk.isin(top2)
print(f"Leave-out top-2 {top2}: remaining n={keep2.sum()} exp={pn[keep2.values].mean():.5f}")

# inversion test of chosen combo on test (sanity)
def inv(r):
    s=fn_best(r); return -s
mi=wg.walk_forward(d, inv, exit=EXIT)
print(f"\nINVERSION test_exp = {mi['test_exp']*100:.4f}%  (chosen combo is long-only; inverse = short-only on gap-downs)")
