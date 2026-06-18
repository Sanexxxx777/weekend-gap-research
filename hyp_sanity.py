import wg_lib as wg, numpy as np, pandas as pd
pd.set_option('display.width',220)
def make_fade(thr, long_bias=False):
    def fn(r):
        if abs(r.gap)<thr: return 0
        if r.gap>0: return 0 if long_bias else -1
        return 1
    return fn

# Look-ahead sanity: shift entry one decision-point earlier/later, edge should be similar not exploding
for ENTRY in ["p_sun2000","p_mon0000","p_mon0600","p_mon0630"]:
    d=wg.add_features(wg.load(True,False),entry=ENTRY)
    fn=make_fade(0.025,long_bias=True)
    m=wg.walk_forward(d,fn,exit="p_mon0930")
    print("entry=%-10s long_bias thr2.5%%: train_exp %.3f%% n%d | TEST %.3f%% n%d wr%.0f%% p%.4f"
          %(ENTRY,m['train_exp']*100,m['train_n'],m['test_exp']*100,m['test_n'],m['test_wr']*100,m['test_p']))

print()
for ENTRY in ["p_sun2000","p_mon0000","p_mon0600","p_mon0630"]:
    d=wg.add_features(wg.load(True,False),entry=ENTRY)
    fn=make_fade(0.025,long_bias=False)
    m=wg.walk_forward(d,fn,exit="p_mon0930")
    print("entry=%-10s both-side thr2.5%%: train_exp %.3f%% n%d | TEST %.3f%% n%d wr%.0f%% p%.4f"
          %(ENTRY,m['train_exp']*100,m['train_n'],m['test_exp']*100,m['test_n'],m['test_wr']*100,m['test_p']))

# Is long_bias edge just market drift? Compare to buy-and-hold same names over weekend (gap-dn perp -> mon0930)
# directional check: in TEST, mean raw long return on gap-down trades vs unconditional perp wknd ret
ENTRY="p_mon0000"
d=wg.add_features(wg.load(True,False),entry=ENTRY)
dates=np.sort(d.fri_date.unique()); cut=dates[int(len(dates)*0.6)]
te=d[d.fri_date>=cut]
raw_all = te.p_mon0930/te.entry_perp-1   # raw perp move entry->open for ALL names
print("\nTEST unconditional perp move (entry p_mon0000 -> p_mon0930): mean %.4f%% n%d (any directional drift?)"
      %(raw_all.mean()*100,len(raw_all)))
gdn = te[te.gap<=-0.025]
print("TEST gap<=-2.5%% subset raw long move: mean %.4f%% n%d wr%.0f%%"
      %((gdn.p_mon0930/gdn.entry_perp-1).mean()*100,len(gdn),((gdn.p_mon0930/gdn.entry_perp-1)>0).mean()*100))
