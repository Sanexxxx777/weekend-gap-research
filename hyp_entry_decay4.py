import wg_lib as wg
import numpy as np

EXIT = "p_mon0930"
def fade(r):
    return (-1 if r.gap > 0 else +1) if abs(r.gap) >= 0.02 else 0

print("="*90)
print("CHAMPION verification: entry-decay says EARLY entry best. Best honest entry = p_mon0000")
print("(p_sun2000 train_n=28 too small; p_mon0900 degenerate entry==exit due to 1h candle).")
print("="*90)

# kfold full-sample sign stability for each honest entry
print("\nkfold (full sample, 3 time blocks) expectancy %:")
for entry in ["p_sun2000","p_mon0000","p_mon0600","p_mon0630","p_mon0800"]:
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=entry)
    kf = wg.kfold(d, fade, exit=EXIT)
    m = wg.walk_forward(d, fade, exit=EXIT)
    signs = "all+" if all(x>0 for x in kf) else ("mixed" if any(x>0 for x in kf) else "all-")
    kfstr = str([round(x*100,2) for x in kf])
    print(f"  {entry:<12} kf={kfstr:<28} ({signs})  test_exp={m['test_exp']*100:+.3f}% p={m['test_p']:.3f}")

# Robustness of entry-decay to threshold choice (is monotone decay an artifact of 2% thr?)
print("\nEntry-decay robust to gap threshold? test_exp% by (entry x thr):")
print(f"{'entry':<12}" + "".join(f"thr{int(t*100)}%".rjust(10) for t in [0.015,0.02,0.025,0.03]))
for entry in ["p_sun2000","p_mon0000","p_mon0600","p_mon0800"]:
    row = f"{entry:<12}"
    for thr in [0.015,0.02,0.025,0.03]:
        d = wg.load(only_regular_monday=True, drop_thin=False)
        d = wg.add_features(d, entry=entry)
        sig = lambda r, t=thr: (-1 if r.gap>0 else +1) if abs(r.gap)>=t else 0
        m = wg.walk_forward(d, sig, exit=EXIT)
        row += f"{m['test_exp']*100:+.2f}".rjust(10)
    print(row)

# D2 honesty: is the 'fresh gap' filter robust or test-luck? Compare its kfold to baseline at mon0630.
print("\nD2 robustness (fade gap fresh, maxretr<=1.0 = a-priori same-sign filter) vs baseline @ mon0630:")
d0 = wg.load(only_regular_monday=True, drop_thin=False)
d0 = wg.add_features(d0, entry="p_mon0630")
d0["gap_sun"] = d0["p_sun2000"]/d0["fri_close_stock"]-1
def base_fade(r):
    return (-1 if r.gap>0 else +1) if abs(r.gap)>=0.02 else 0
def fresh_fade(r):
    if abs(r.gap)<0.02: return 0
    if not (r.gap*r.gap_sun>0): return 0   # gap kept its sign since sun (didn't flip/overshoot)
    return -1 if r.gap>0 else +1
for name,sig in [("baseline",base_fade),("fresh(maxretr<=1)",fresh_fade)]:
    m = wg.walk_forward(d0, sig, exit=EXIT)
    kf = wg.kfold(d0, sig, exit=EXIT)
    print(f"  {name:<20} train={m['train_exp']*100:+.3f}%(n{m['train_n']}) test={m['test_exp']*100:+.3f}%(n{m['test_n']}) wr={m['test_wr']*100:.0f}% p={m['test_p']:.3f} kf={[round(x*100,2) for x in kf]}")
