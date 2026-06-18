import wg_lib as wg
import numpy as np

d = wg.load(only_regular_monday=True, drop_thin=False)
d = wg.add_features(d, entry="p_mon0630")

def make_fade(th):
    t = th / 100.0
    def sig(r):
        if abs(r.gap) >= t:
            return -1 if r.gap > 0 else 1
        return 0
    return sig

THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
EXIT = "p_mon0930"

# Honest threshold selection: pick threshold MAXIMIZING TRAIN exp, then report its TEST blindly.
print("Threshold selection on TRAIN only (fade, exit=p_mon0930):")
best_th, best_tr = None, -1e9
for th in THRESHOLDS:
    m = wg.walk_forward(d, make_fade(th), exit=EXIT)
    print(f"  th={th}% train_exp={m['train_exp']*100:+.3f}% (train_n={m['train_n']})")
    if m['train_exp'] > best_tr:
        best_tr, best_th = m['train_exp'], th
print(f"\n>> Train-best threshold = {best_th}% (train_exp={best_tr*100:+.3f}%)")

m = wg.walk_forward(d, make_fade(best_th), exit=EXIT)
kf = wg.kfold(d, make_fade(best_th), exit=EXIT)
print(f">> BLIND TEST at th={best_th}%: test_n={m['test_n']} test_exp={m['test_exp']*100:+.3f}% test_wr={m['test_wr']*100:.1f}% p={m['test_p']:.3f}")
print(f">> kfold = {[round(x*100,3) for x in kf]}")

# Inversion sanity test for the chosen fade signal (invert sides)
def make_anti(th):
    base = make_fade(th)
    def sig(r):
        return -base(r)
    return sig

mi = wg.walk_forward(d, make_anti(best_th), exit=EXIT)
print(f"\nINVERSION test (anti-fade) th={best_th}%: test_n={mi['test_n']} test_exp={mi['test_exp']*100:+.3f}% test_wr={mi['test_wr']*100:.1f}% p={mi['test_p']:.3f}")

# Also report the most ROBUST candidate: th=2.5 (kfold most sign-stable among liquid n)
print("\n--- Detail th=2.5% (best kfold sign-stability with decent n) ---")
m25 = wg.walk_forward(d, make_fade(2.5), exit=EXIT)
kf25 = wg.kfold(d, make_fade(2.5), exit=EXIT)
print(f"train_exp={m25['train_exp']*100:+.3f}% test_n={m25['test_n']} test_exp={m25['test_exp']*100:+.3f}% test_wr={m25['test_wr']*100:.1f}% p={m25['test_p']:.3f} kfold={[round(x*100,3) for x in kf25]}")
mi25 = wg.walk_forward(d, make_anti(2.5), exit=EXIT)
print(f"inversion th=2.5%: test_exp={mi25['test_exp']*100:+.3f}%")

print("\n--- Detail th=2.0% (reference-confirmed liquid baseline) ---")
m20 = wg.walk_forward(d, make_fade(2.0), exit=EXIT)
kf20 = wg.kfold(d, make_fade(2.0), exit=EXIT)
mi20 = wg.walk_forward(d, make_anti(2.0), exit=EXIT)
print(f"train_exp={m20['train_exp']*100:+.3f}% test_n={m20['test_n']} test_exp={m20['test_exp']*100:+.3f}% test_wr={m20['test_wr']*100:.1f}% p={m20['test_p']:.3f} kfold={[round(x*100,3) for x in kf20]} inversion_test_exp={mi20['test_exp']*100:+.3f}%")
