#!/usr/bin/env python3
"""
TP/SL path-бэктест weekend-gap FADE. Симуляция по 1h-свечам перпа от входа до открытия NASDAQ.
ИСТОЧНИКИ: weekend_panel (вход/gap/направление) + perp_hourly (high/low внутри хода для TP/SL).
Honest: TP/SL проверяются по high/low завершённых 1h-свечей; при двойном пересечении в свече -> SL первым (worst-case).
Уровни в ДОЛЯХ ГЭПА (адаптивны к овершуту): tp_k*|gap| профит, sl_m*|gap| против. Backstop-выход = открытие пн.
Walk-forward: параметры выбираем на train, мерим на test. costs 0.19% round-trip.
"""
import pandas as pd, numpy as np
from zoneinfo import ZoneInfo
import datetime
import wg_lib as wg

NY = ZoneInfo("America/New_York"); UTC = ZoneInfo("UTC")
COST = wg.RT_COST
GAP_THR = 0.02
ENTRY_ET = (0,0)     # Mon 00:00 ET вход (как в бэктесте)
EXIT_ET  = (9,30)    # backstop выход = открытие NASDAQ 09:30 ET

# --- данные ---
panel = wg.add_features(wg.load(only_regular_monday=True), entry="p_mon0000")
panel = panel[panel.gap.abs() >= GAP_THR].copy()
h = pd.read_csv("perp_hourly.csv")
# свечи перпа по тикеру: список (t_open, T_close, high, low, close), отсортировано
H = {}
for tic, g in h.groupby("ticker"):
    g = g.sort_values("t_open_ms")
    H[tic] = g[["t_open_ms","T_close_ms","high","low","close"]].values

def et_ms(date, hh, mm):
    d = pd.Timestamp(date)
    dt = datetime.datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY)
    return int(dt.timestamp()*1000)

def simulate(row, tp_k, sl_m):
    """Вернуть (pnl_net, reason). side=-1 short(gap>0)/+1 long(gap<0). Honest path по 1h-свечам."""
    side = -1 if row.gap > 0 else 1
    entry = row.entry_perp
    g = abs(row.gap)
    # TP = в сторону fade (профит), SL = против
    if side == -1:  # short: профит=перп вниз, лосс=перп вверх
        tp_price = entry*(1 - tp_k*g); sl_price = entry*(1 + sl_m*g)
    else:           # long: профит=перп вверх, лосс=перп вниз
        tp_price = entry*(1 + tp_k*g); sl_price = entry*(1 - sl_m*g)
    e_ms = et_ms(row.mon_date, *ENTRY_ET); x_ms = et_ms(row.mon_date, *EXIT_ET)
    cd = H.get(row.ticker)
    if cd is None:
        return None
    exit_px = None; reason = "time"
    for t_open, T_close, hi, lo, cl in cd:
        if T_close <= e_ms:   # свеча до входа
            continue
        if t_open >= x_ms:    # свеча после backstop-выхода
            break
        hit_tp = (lo <= tp_price) if side==-1 else (hi >= tp_price)
        hit_sl = (hi >= sl_price) if side==-1 else (lo <= sl_price)
        if hit_sl:            # worst-case: SL раньше TP при двойном пересечении
            exit_px = sl_price; reason = "SL"; break
        if hit_tp:
            exit_px = tp_price; reason = "TP"; break
    if exit_px is None:       # backstop: выход на открытии (перп на ~09:00-09:30)
        exit_px = row.p_mon0930 if not pd.isna(row.p_mon0930) else cl
    raw = side*(exit_px/entry - 1)
    return raw - COST, reason

def run(tp_k, sl_m, split="all"):
    dates = np.sort(panel.fri_date.unique()); cut = dates[int(len(dates)*0.6)]
    sub = panel if split=="all" else (panel[panel.fri_date<cut] if split=="train" else panel[panel.fri_date>=cut])
    res = [simulate(r, tp_k, sl_m) for r in sub.itertuples()]
    pnl = np.array([x[0] for x in res if x]); reasons = [x[1] for x in res if x]
    if len(pnl)==0: return None
    wkdf = pd.DataFrame({"fri":sub.iloc[:len(pnl)].fri_date.values, "pnl":pnl})
    wk = wkdf.groupby("fri").pnl.mean()
    return {"n":len(pnl), "exp":pnl.mean(), "wr":(pnl>0).mean(), "wk_exp":wk.mean(),
            "wk_pos":f"{(wk>0).sum()}/{len(wk)}", "p":wg.bootstrap_p(pnl),
            "tp%":sum(r=='TP' for r in reasons)/len(reasons), "sl%":sum(r=='SL' for r in reasons)/len(reasons),
            "tm%":sum(r=='time' for r in reasons)/len(reasons)}

print("="*78)
print("TP/SL PATH-БЭКТЕСТ (fade |gap|>=2%, вход Mon00:00, backstop=открытие NASDAQ)")
print("Уровни в долях гэпа. Honest path по 1h high/low. costs 0.19%.")
print("="*78)
print(f"позиций (|gap|>=2%): {len(panel)}, выходных: {panel.fri_date.nunique()}\n")

# бейзлайн: чистый time-exit (без TP/SL) = бесконечные уровни
base_tr = run(99,99,"train"); base_te = run(99,99,"test")
print(f"БЕЙЗЛАЙН time-exit (без TP/SL): TRAIN exp {base_tr['exp']*100:+.2f}% | TEST exp {base_te['exp']*100:+.2f}% "
      f"wk {base_te['wk_exp']*100:+.2f}% ({base_te['wk_pos']}) p={base_te['p']:.3f}\n")

print(f"{'tp_k':>5}{'sl_m':>5} | {'TR_exp':>8}{'TE_exp':>8}{'TE_wk':>8}{'TE_wr':>7}{'TE_p':>7} | {'TP%':>5}{'SL%':>5}{'tm%':>5}")
grid=[]
for tp_k in [0.5,0.75,1.0,1.5]:
    for sl_m in [1.0,1.5,2.0,3.0]:
        tr=run(tp_k,sl_m,"train"); te=run(tp_k,sl_m,"test")
        if not tr or not te: continue
        grid.append((tp_k,sl_m,tr,te))
        print(f"{tp_k:>5}{sl_m:>5} | {tr['exp']*100:>7.2f}%{te['exp']*100:>7.2f}%{te['wk_exp']*100:>7.2f}%"
              f"{te['wr']*100:>6.0f}%{te['p']:>7.3f} | {te['tp%']*100:>4.0f}%{te['sl%']*100:>4.0f}%{te['tm%']*100:>4.0f}%")

# выбор ЛУЧШЕГО по TRAIN (честно), затем показать его TEST
best = max(grid, key=lambda x: x[2]['exp'])
tp_k,sl_m,tr,te = best
print(f"\n>>> ЛУЧШИЙ по TRAIN: tp_k={tp_k} sl_m={sl_m} (TRAIN exp {tr['exp']*100:+.2f}%)")
print(f"    его TEST (вслепую): exp {te['exp']*100:+.2f}%/сделку, weekend {te['wk_exp']*100:+.2f}% ({te['wk_pos']}), "
      f"WR {te['wr']*100:.0f}%, p={te['p']:.3f}, исходы TP/SL/time={te['tp%']*100:.0f}/{te['sl%']*100:.0f}/{te['tm%']*100:.0f}%")
