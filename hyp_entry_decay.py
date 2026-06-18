import wg_lib as wg
import numpy as np

DECISION = ["p_sun2000","p_mon0000","p_mon0600","p_mon0630","p_mon0800","p_mon0900"]
EXIT = "p_mon0930"

def fade_signal(thr):
    def f(r):
        if abs(r.gap) >= thr:
            return -1 if r.gap > 0 else +1
        return 0
    return f

print("="*100)
print("PART A: entry-decay — fade |gap|>=2%, exit=p_mon0930, scan entry from early to late")
print("="*100)
print(f"{'entry':<12}{'train_n':>8}{'train_exp%':>11}{'test_n':>8}{'test_exp%':>11}{'test_wr%':>10}{'test_p':>9}{'kfold%':>26}")
for entry in DECISION:
    d = wg.load(only_regular_monday=True, drop_thin=False)
    d = wg.add_features(d, entry=entry)
    sig = fade_signal(0.02)
    m = wg.walk_forward(d, sig, exit=EXIT)
    kf = wg.kfold(d, sig, exit=EXIT)
    kfs = "[" + ",".join(f"{x*100:+.2f}" for x in kf) + "]"
    print(f"{entry:<12}{m['train_n']:>8}{m['train_exp']*100:>11.3f}{m['test_n']:>8}{m['test_exp']*100:>11.3f}{m['test_wr']*100:>10.1f}{m['test_p']:>9.3f}{kfs:>26}")

print()
print("="*100)
print("PART A2: same but drop_thin=True (remove thin tickers)")
print("="*100)
print(f"{'entry':<12}{'train_n':>8}{'train_exp%':>11}{'test_n':>8}{'test_exp%':>11}{'test_wr%':>10}{'test_p':>9}{'kfold%':>26}")
for entry in DECISION:
    d = wg.load(only_regular_monday=True, drop_thin=True)
    d = wg.add_features(d, entry=entry)
    sig = fade_signal(0.02)
    m = wg.walk_forward(d, sig, exit=EXIT)
    kf = wg.kfold(d, sig, exit=EXIT)
    kfs = "[" + ",".join(f"{x*100:+.2f}" for x in kf) + "]"
    print(f"{entry:<12}{m['train_n']:>8}{m['train_exp']*100:>11.3f}{m['test_n']:>8}{m['test_exp']*100:>11.3f}{m['test_wr']*100:>10.1f}{m['test_p']:>9.3f}{kfs:>26}")
