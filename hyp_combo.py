import wg_lib as wg, numpy as np, pandas as pd
pd.set_option('display.width', 220)

THIN = wg.THIN

def make_fade(thr, long_bias=False, drop_thin=False):
    def fn(r):
        if drop_thin and r.ticker in THIN: return 0
        if abs(r.gap) < thr: return 0
        if r.gap > 0:
            return 0 if long_bias else -1
        else:
            return 1
    return fn

# A-PRIORI constraint (from reference, NOT from peeking at test):
#  - exit = p_mon0930 ONLY (proven robust; p_mon1200 is the documented overfit/momentum-reversal trap)
#  - require >=30 train trades (avoid tiny-n lucky configs)
# Then pick best by TRAIN sharpe among the remaining. Measure test once.

entries = wg.DECISION_POINTS
thrs    = [0.010,0.015,0.020,0.025,0.030]
EXIT = "p_mon0930"
rows=[]
for entry in entries:
    d = wg.add_features(wg.load(True,False), entry=entry)
    dates=np.sort(d.fri_date.unique()); cut=dates[int(len(dates)*0.6)]
    tr = d[d.fri_date<cut]
    for thr in thrs:
        for lb in [False,True]:
            for dt in [False,True]:
                fn=make_fade(thr,lb,dt)
                pn=wg.pnl_series(tr,fn,exit=EXIT)
                if len(pn)<30: continue
                rows.append(dict(entry=entry,thr=thr,long_bias=lb,drop_thin=dt,
                                 tr_n=len(pn),tr_exp=pn.mean(),tr_wr=(pn>0).mean(),
                                 tr_sharpe=pn.mean()/pn.std() if pn.std()>0 else 0))
cf=pd.DataFrame(rows).sort_values("tr_sharpe",ascending=False)
print("constrained configs (exit=p_mon0930, tr_n>=30):", len(cf))
print(cf.head(12).to_string(index=False))

# pick train-best
best=cf.iloc[0]
print("\nTRAIN-BEST:", dict(best[['entry','thr','long_bias','drop_thin','tr_n','tr_exp','tr_wr','tr_sharpe']]))

print("\n=== diagnostics on TRAIN for top family (entry=p_mon0000, exit=p_mon0930) ===")
d = wg.add_features(wg.load(True,False), entry="p_mon0000")
dates=np.sort(d.fri_date.unique()); cut=dates[int(len(dates)*0.6)]
tr=d[d.fri_date<cut]
# side breakdown at thr=0.025 no long_bias
def fade(thr):
    def fn(r):
        if abs(r.gap)<thr: return 0
        return -1 if r.gap>0 else 1
    return fn
pn_all = wg.pnl_series(tr, fade(0.025), exit="p_mon0930")
sides = tr.loc[pn_all.index].gap.apply(lambda g: 'short(gapUp)' if g>0 else 'long(gapDn)')
print("train both-sides thr0.025: n",len(pn_all),"exp",round(pn_all.mean(),5),"wr",round((pn_all>0).mean(),3))
for s in ['short(gapUp)','long(gapDn)']:
    idx=sides[sides==s].index
    sub=pn_all.loc[idx]
    print(f"  {s}: n={len(sub)} exp={sub.mean():.5f} wr={(sub>0).mean():.3f}")

# train kfold stability of train-best config
def best_fn(r):
    if abs(r.gap)<0.025: return 0
    if r.gap>0: return 0   # long_bias: skip shorts
    return 1
print("train-kfold (best, long_bias) on TRAIN slice only:")
kf_tr = wg.kfold(tr, best_fn, exit="p_mon0930")
print("  ", [round(x,5) for x in kf_tr])
