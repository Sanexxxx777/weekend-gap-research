import wg_lib as wg, numpy as np, pandas as pd
def make_fade(thr, long_bias=False):
    def fn(r):
        if abs(r.gap)<thr: return 0
        if r.gap>0: return 0 if long_bias else -1
        return 1
    return fn
d=wg.add_features(wg.load(True,False),entry="p_mon0000")

# CHOSEN combo final numbers
fn=make_fade(0.025,long_bias=True)
m=wg.walk_forward(d,fn,exit="p_mon0930"); kf=wg.kfold(d,fn,exit="p_mon0930")
print("CHOSEN long_bias: train %.4f%% TEST %.4f%% wr %.4f n %d p %.4f kf %s"
      %(m['train_exp']*100,m['test_exp']*100,m['test_wr']*100,m['test_n'],m['test_p'],[round(x*100,4) for x in kf]))
inv=lambda r:-fn(r); mi=wg.walk_forward(d,inv,exit="p_mon0930")
print("  inversion test_exp %.4f%%"%(mi['test_exp']*100))

# BOTH-SIDED combo final (the safer prod candidate)
fb=make_fade(0.025,long_bias=False)
mb=wg.walk_forward(d,fb,exit="p_mon0930"); kfb=wg.kfold(d,fb,exit="p_mon0930")
print("\nBOTH-SIDED: train %.4f%% TEST %.4f%% wr %.4f n %d p %.4f kf %s"
      %(mb['train_exp']*100,mb['test_exp']*100,mb['test_wr']*100,mb['test_n'],mb['test_p'],[round(x*100,4) for x in kfb]))
invb=lambda r:-fb(r); mib=wg.walk_forward(d,invb,exit="p_mon0930")
print("  inversion test_exp %.4f%%"%(mib['test_exp']*100))
# both-sided per ticker on test
dates=np.sort(d.fri_date.unique()); cut=dates[int(len(dates)*0.6)]; te=d[d.fri_date>=cut]
pn=wg.pnl_series(te,fb,exit="p_mon0930"); tk=te.loc[pn.index].ticker
c=pd.DataFrame({'pnl':pn.values,'t':tk.values}).groupby('t').agg(n=('pnl','size'),sum=('pnl','sum')).sort_values('sum',ascending=False)
print("  TEST top contributors:", c.head(3).to_dict()['sum'])
top2=c.index[:2].tolist(); keep=~tk.isin(top2)
print("  leave-out top2 %s: exp %.4f%% n%d"%(top2,pn[keep.values].mean()*100,keep.sum()))
