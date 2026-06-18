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

# ---- A. drop_thin=True (real cost concern: flat 0.19% understates thin weekend books) ----
print("=== A. drop_thin=True (remove GME,ZM,EBAY,BB,RKLB,DKNG,HIMS) ===")
dt = wg.load(only_regular_monday=True, drop_thin=True)
dt = wg.add_features(dt, entry=ENTRY)
mt = wg.walk_forward(dt, fade2, exit=EXIT, cost=wg.RT_COST, train_frac=0.6)
kft = wg.kfold(dt, fade2, exit=EXIT)
print(f"TRAIN exp={mt['train_exp']*100:.3f}% n={mt['train_n']}")
print(f"TEST  exp={mt['test_exp']*100:.3f}% n={mt['test_n']} wr={mt['test_wr']:.3f} p={mt['test_p']:.4f} kfold={[round(x*100,2) for x in kft]}")

# ---- B. concentration on weekends: per-weekend test PnL ----
print("\n=== B. per-weekend test concentration ===")
d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry=ENTRY)
cut = pd.Timestamp("2026-03-13")
te = d[d.fri_date >= cut].copy()
pn = wg.pnl_series(te, fade2, EXIT, wg.RT_COST)
te_t = te.loc[pn.index].copy(); te_t["_pnl"]=pn.values
wk = te_t.groupby("fri_date")["_pnl"].agg(["sum","count","mean"]).sort_values("sum", ascending=False)
print((wk.assign(sum=lambda x:x["sum"]*100, mean=lambda x:x["mean"]*100)).round(3).to_string())
print(f"\ntop-1 weekend sum = {wk['sum'].iloc[0]*100:.2f}% of total {pn.sum()*100:.2f}%")
# drop top weekend
top_wk = wk.index[0]
te_no_top = te[te.fri_date != top_wk]
pn2 = wg.pnl_series(te_no_top, fade2, EXIT, wg.RT_COST)
print(f"TEST without top weekend: exp={pn2.mean()*100:.3f}% n={len(pn2)} p={wg.bootstrap_p(pn2):.4f}")

# ---- C. higher cost on thin tickers (0.5%/side slip on THIN -> RT ~1.09%) ----
print("\n=== C. realistic thin-ticker cost (THIN: slip 0.5%/side) ===")
def pnl_real_cost(sub):
    sub = sub.dropna(subset=[EXIT]).copy()
    sides = sub.apply(fade2, axis=1)
    sub = sub.assign(_side=sides)
    sub = sub[sub._side != 0].copy()
    raw = sub._side*(sub[EXIT]/sub.entry_perp - 1)
    thin_rt = 2*(wg.TAKER_FEE + 0.005)   # 0.5% slip/side
    cost = sub.ticker.apply(lambda t: thin_rt if t in wg.THIN else wg.RT_COST)
    return (raw - cost.values)
pn_rc = pnl_real_cost(te)
print(f"TEST realistic-cost: exp={pn_rc.mean()*100:.3f}% n={len(pn_rc)} wr={(pn_rc>0).mean():.3f} p={wg.bootstrap_p(pn_rc):.4f}")

# ---- D. LOO p after dropping top-2 explicitly with bootstrap ----
print("\n=== D. LOO top-2 (GME,MSTR) bootstrap p on test ===")
te_loo = te[~te.ticker.isin(['GME','MSTR'])]
pn_loo = wg.pnl_series(te_loo, fade2, EXIT, wg.RT_COST)
print(f"TEST LOO: exp={pn_loo.mean()*100:.3f}% n={len(pn_loo)} p={wg.bootstrap_p(pn_loo):.4f}")
# also drop top-3
te_loo3 = te[~te.ticker.isin(['GME','MSTR','CRCL'])]
pn_loo3 = wg.pnl_series(te_loo3, fade2, EXIT, wg.RT_COST)
print(f"TEST LOO top-3 (+CRCL): exp={pn_loo3.mean()*100:.3f}% n={len(pn_loo3)} p={wg.bootstrap_p(pn_loo3):.4f}")
