import wg_lib as wg, numpy as np, pandas as pd

ENTRY="p_mon0000"; THR=0.025; EXIT="p_mon0930"
def cand(r):
    if abs(r.gap)<THR: return 0
    if r.gap>0: return 0
    return 1

d0=wg.add_features(wg.load(True,False),entry=ENTRY)
dates=np.sort(d0.fri_date.unique()); cut=dates[int(len(dates)*0.6)]
te=d0[d0.fri_date>=cut].copy()
pn=wg.pnl_series(te,cand,exit=EXIT)
tu=te.loc[pn.index].copy(); tu["_p"]=pn.values

print("=== CLUSTERING: trades per weekend in TEST ===")
g=tu.groupby("fri_date")["_p"].agg(["count","mean","sum"])
print(g.to_string())
print(f"\n  distinct weekends in test = {tu.fri_date.nunique()}  (trades={len(tu)})")
print("  -> events NOT independent: many winners come from the SAME down-weekends")

print("\n=== block bootstrap by WEEKEND (resample weekends, not trades) ===")
# each weekend = one event; trade-level p overstates significance
wk_mean=tu.groupby("fri_date")["_p"].mean().values   # avg pnl per weekend
print("  per-weekend mean pnl%:", [round(x*100,3) for x in wk_mean])
rng=np.random.default_rng(0); n=20000
boot=rng.choice(wk_mean,size=(n,len(wk_mean)),replace=True).mean(axis=1)
print(f"  weekend-level bootstrap_p (mean<=0) = {(boot<=0).mean():.4f}  (n_weekends={len(wk_mean)})")

print("\n=== LOO by WEEKEND: drop the single best weekend ===")
best_wk=tu.groupby("fri_date")["_p"].sum().idxmax()
print(f"  best weekend = {pd.Timestamp(best_wk).date()} sum={tu.groupby('fri_date')['_p'].sum().max()*100:.2f}%")
rest=tu[tu.fri_date!=best_wk]
print(f"  drop it: n={len(rest)} exp={rest._p.mean()*100:.4f}% wr={(rest._p>0).mean():.3f} p={wg.bootstrap_p(rest._p.values):.4f} weekends={rest.fri_date.nunique()}")

print("\n=== inversion p (one-sided that inverted>0) just to confirm symmetry ===")
mi=wg.pnl_series(te,lambda r:-cand(r),exit=EXIT)
print(f"  inverted test exp={mi.mean()*100:.3f}% (mirror of base, expected)")

print("\n=== gap-size sensitivity: candidate only longs gap-DOWNS >=2.5%. base rate of such weekends ===")
allf=wg.add_features(wg.load(True,False),entry=ENTRY)
down=allf[allf.gap<=-THR]
print(f"  total gap-down>=2.5% events in full panel: {len(down)} across {down.fri_date.nunique()} weekends, {down.ticker.nunique()} tickers")
print(f"  these cluster on macro down-weekends (crypto/risk selloffs). tickers: {sorted(down.ticker.unique())}")
