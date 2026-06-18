import wg_lib as wg
import itertools

DECISION_POINTS = ["p_sun2000","p_mon0000","p_mon0600","p_mon0630","p_mon0800","p_mon0900"]
EXIT_POINTS     = ["p_mon0930","p_mon1100","p_mon1200","p_mon1600"]

# fade big gap: short when gap>=+2%, long when gap<=-2%, else skip. Threshold априорный.
def fade2(r):
    if r.gap >= 0.02:  return -1
    if r.gap <= -0.02: return +1
    return 0

base = wg.load(only_regular_monday=True, drop_thin=False)

rows = []
for entry, exit in itertools.product(DECISION_POINTS, EXIT_POINTS):
    # exit must be strictly after entry in time; skip degenerate pairs
    d = wg.add_features(base, entry=entry)
    m = wg.walk_forward(d, fade2, exit=exit, cost=wg.RT_COST, train_frac=0.6)
    kf = wg.kfold(d, fade2, exit=exit)
    kf_signs_consistent = all(x>0 for x in kf) or all(x<0 for x in kf)
    rows.append({
        "entry": entry, "exit": exit,
        "train_exp": m["train_exp"], "train_n": m["train_n"], "train_wr": m["train_wr"],
        "test_exp": m["test_exp"], "test_n": m["test_n"], "test_wr": m["test_wr"],
        "test_sharpe": m["test_sharpe"], "test_p": m["test_p"],
        "all_exp": m["all_exp"], "all_n": m["all_n"],
        "kf": [round(x*100,3) for x in kf], "kf_consistent": kf_signs_consistent,
        "cut": m["cut_date"],
    })

# print sorted by test_exp desc
rows_sorted = sorted(rows, key=lambda r: r["test_exp"], reverse=True)
print(f"{'entry':<11}{'exit':<11}{'tr_exp%':>8}{'tr_n':>5}{'te_exp%':>8}{'te_wr%':>7}{'te_n':>5}{'te_p':>7}{'te_shrp':>8}  kfold%(3blk)         kf_cons")
for r in rows_sorted:
    print(f"{r['entry']:<11}{r['exit']:<11}{r['train_exp']*100:>8.3f}{r['train_n']:>5}{r['test_exp']*100:>8.3f}{r['test_wr']*100:>7.1f}{r['test_n']:>5}{r['test_p']:>7.3f}{r['test_sharpe']:>8.3f}  {str(r['kf']):<20} {r['kf_consistent']}")

print("\ncut_date:", rows[0]["cut"])
