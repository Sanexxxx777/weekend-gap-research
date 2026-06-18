import wg_lib as wg
import numpy as np

d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry="p_mon0630")

print(f"total rows after features: {len(d)}, tickers: {d.ticker.nunique()}")

THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
EXIT = "p_mon0930"

def make_fade(th):
    t = th / 100.0
    def sig(r):
        if abs(r.gap) >= t:
            return -1 if r.gap > 0 else 1   # fade: short positive gap, long negative gap
        return 0
    return sig

def make_follow(th):
    t = th / 100.0
    def sig(r):
        if abs(r.gap) >= t:
            return 1 if r.gap > 0 else -1   # momentum: follow gap direction
        return 0
    return sig

print("\n=== FADE (short positive gap / long negative gap), exit=p_mon0930 ===")
print(f"{'th%':>5} {'tr_n':>5} {'tr_exp%':>8} {'te_n':>5} {'te_exp%':>8} {'te_wr%':>7} {'te_p':>6}  {'kfold%':>22}")
fade_res = {}
for th in THRESHOLDS:
    sig = make_fade(th)
    m = wg.walk_forward(d, sig, exit=EXIT)
    kf = wg.kfold(d, sig, exit=EXIT)
    fade_res[th] = (m, kf)
    kfs = " ".join(f"{x*100:+.2f}" for x in kf)
    print(f"{th:>5} {m['train_n']:>5} {m['train_exp']*100:>+8.3f} {m['test_n']:>5} {m['test_exp']*100:>+8.3f} {m['test_wr']*100:>7.1f} {m['test_p']:>6.3f}  [{kfs}]")

print("\n=== FOLLOW/MOMENTUM (long positive gap / short negative gap), exit=p_mon0930 ===")
print(f"{'th%':>5} {'tr_n':>5} {'tr_exp%':>8} {'te_n':>5} {'te_exp%':>8} {'te_wr%':>7} {'te_p':>6}  {'kfold%':>22}")
foll_res = {}
for th in THRESHOLDS:
    sig = make_follow(th)
    m = wg.walk_forward(d, sig, exit=EXIT)
    kf = wg.kfold(d, sig, exit=EXIT)
    foll_res[th] = (m, kf)
    kfs = " ".join(f"{x*100:+.2f}" for x in kf)
    print(f"{th:>5} {m['train_n']:>5} {m['train_exp']*100:>+8.3f} {m['test_n']:>5} {m['test_exp']*100:>+8.3f} {m['test_wr']*100:>7.1f} {m['test_p']:>6.3f}  [{kfs}]")

# momentum specifically for BIG gaps >3% as a band (3-inf)
print("\n=== MOMENTUM for BIG gaps only (band |gap|>=3%), follow direction ===")
sig = make_follow(3.0)
m = wg.walk_forward(d, sig, exit=EXIT)
kf = wg.kfold(d, sig, exit=EXIT)
print(f"train_n={m['train_n']} train_exp={m['train_exp']*100:+.3f}% test_n={m['test_n']} test_exp={m['test_exp']*100:+.3f}% test_wr={m['test_wr']*100:.1f}% p={m['test_p']:.3f} kfold={[round(x*100,3) for x in kf]}")
