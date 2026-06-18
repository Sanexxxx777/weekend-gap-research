import wg_lib as wg
import numpy as np

EXIT = "p_mon0930"

print("="*95)
print("PART D: signal by CHANGE of perp between two decision points (does fade-in-progress continue?)")
print("Idea: if perp already retraced from sun2000 toward fri_close by mon0630, does it KEEP retracing to open?")
print("="*95)

# We enter at mon0630. Use delta = p_mon0630/p_sun2000 - 1 (perp move overnight) as the SIGNAL.
# Two variants:
#  D1: 'continuation' — bet the direction of the recent perp move continues (perp moved down -> short).
#  D2: 'fade the gap, only if perp has NOT yet fully retraced' (gap still wide AND was wider before).
# All decision-only: gap (at mon0630), p_sun2000, entry_perp(=p_mon0630). No future fields.

def add_delta(d):
    d = d.copy()
    d["perp_delta"] = d["p_mon0630"]/d["p_sun2000"] - 1   # overnight perp move, decision-only
    d["gap_sun"] = d["p_sun2000"]/d["fri_close_stock"] - 1 # gap at sun2000 (earlier), decision-only
    return d

d0 = wg.load(only_regular_monday=True, drop_thin=False)
d0 = wg.add_features(d0, entry="p_mon0630")
d0 = add_delta(d0)

# D1: pure continuation of recent perp move (momentum on overnight delta), threshold on |delta|
print("\nD1 continuation: side = sign(perp_delta) if |perp_delta|>=thr")
for thr in [0.005, 0.01, 0.015, 0.02]:
    def sig(r, thr=thr):
        if abs(r.perp_delta) >= thr:
            return +1 if r.perp_delta > 0 else -1
        return 0
    m = wg.walk_forward(d0, sig, exit=EXIT)
    kf = wg.kfold(d0, sig, exit=EXIT)
    print(f"  thr={thr:.3f}  train_exp={m['train_exp']*100:+.3f}% (n{m['train_n']})  test_exp={m['test_exp']*100:+.3f}% (n{m['test_n']}) wr={m['test_wr']*100:.0f}% p={m['test_p']:.3f}  kf={[round(x*100,2) for x in kf]}")

# D1b: contrarian to recent perp move (fade the overnight move)
print("\nD1b fade overnight move: side = -sign(perp_delta) if |perp_delta|>=thr")
for thr in [0.005, 0.01, 0.015, 0.02]:
    def sig(r, thr=thr):
        if abs(r.perp_delta) >= thr:
            return -1 if r.perp_delta > 0 else +1
        return 0
    m = wg.walk_forward(d0, sig, exit=EXIT)
    kf = wg.kfold(d0, sig, exit=EXIT)
    print(f"  thr={thr:.3f}  train_exp={m['train_exp']*100:+.3f}% (n{m['train_n']})  test_exp={m['test_exp']*100:+.3f}% (n{m['test_n']}) wr={m['test_wr']*100:.0f}% p={m['test_p']:.3f}  kf={[round(x*100,2) for x in kf]}")

# D2: fade gap (at mon0630) but condition on whether gap is STILL near its sun2000 level (not yet retraced)
print("\nD2 fade |gap|>=2% at mon0630, conditioned: only if perp has NOT already retraced >X of the way back")
print("   retrace_frac = 1 - gap/gap_sun  (how much of sun-gap already closed by mon0630)")
def base_fade(r):
    return (-1 if r.gap > 0 else +1) if abs(r.gap) >= 0.02 else 0
for maxretr in [1.0, 0.5, 0.3, 0.0, -10]:  # only enter if retraced LESS than maxretr (i.e. gap still fresh)
    def sig(r, maxretr=maxretr):
        if abs(r.gap) < 0.02: return 0
        if abs(r.gap_sun) < 1e-6: return 0
        retr = 1 - r.gap/r.gap_sun   # fraction of sun-gap already closed (same-sign assumption)
        # only trade if gap is "fresh": retraced less than maxretr of the way
        if not (r.gap*r.gap_sun > 0): return 0  # require same sign to define retrace meaningfully
        if retr > maxretr: return 0
        return -1 if r.gap > 0 else +1
    m = wg.walk_forward(d0, sig, exit=EXIT)
    kf = wg.kfold(d0, sig, exit=EXIT)
    print(f"  maxretr<={maxretr:>5}  train_exp={m['train_exp']*100:+.3f}% (n{m['train_n']})  test_exp={m['test_exp']*100:+.3f}% (n{m['test_n']}) wr={m['test_wr']*100:.0f}% p={m['test_p']:.3f}  kf={[round(x*100,2) for x in kf]}")

print("\nReference baseline (no delta condition): fade|gap|>=2% at mon0630")
m = wg.walk_forward(d0, base_fade, exit=EXIT)
print(f"  train_exp={m['train_exp']*100:+.3f}% (n{m['train_n']})  test_exp={m['test_exp']*100:+.3f}% (n{m['test_n']}) wr={m['test_wr']*100:.0f}% p={m['test_p']:.3f}")
