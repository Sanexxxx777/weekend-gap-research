"""
wg_lib — единый honest-модуль для weekend-gap бэктеста (рой использует ТОЛЬКО его).
Гарантии против look-ahead и naive-fill:
  - signal_fn видит ТОЛЬКО decision-поля (entry-перп, fri_close, market на entry) — никаких mon_open/mon_close.
  - costs всегда вычитаются (taker fee + slippage), round-trip.
  - walk_forward: train = ранние выходные, test = свежие; параметры выбирать на train, мерить на test.
  - bootstrap_p: одностор. p-value что mean(test pnl) > 0.
Колонки панели см. COLS ниже. Все цены перпа honest (last завершённая 1h свеча <= момента).
"""
import pandas as pd, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "weekend_panel.csv")
PERP_HOURLY = os.path.join(HERE, "perp_hourly.csv")

# --- издержки (HL stock-перпы) ---
TAKER_FEE = 0.00045      # per side
SLIPPAGE  = 0.0005       # per side, консервативно для $1-3K (крупные перпы spread ~0.0003)
RT_COST   = 2*(TAKER_FEE+SLIPPAGE)   # round-trip ~0.19%

# decision-моменты (что доступно на принятии решения) и торговые точки выхода (перп ликвиден 24/7)
DECISION_POINTS = ["p_sun2000","p_mon0000","p_mon0600","p_mon0630","p_mon0800","p_mon0900"]
EXIT_POINTS     = ["p_mon0930","p_mon1100","p_mon1200","p_mon1600"]
# тонкие тикеры (мелкая книга в выходные -> большой slip); фильтровать в большинстве гипотез
THIN = {"GME","ZM","EBAY","BB","RKLB","DKNG","HIMS"}

def load(only_regular_monday=True, drop_thin=False):
    df = pd.read_csv(PANEL)
    df["fri_date"] = pd.to_datetime(df["fri_date"])
    if only_regular_monday:
        df = df[df.is_holiday_mon == 0].copy()
    if drop_thin:
        df = df[~df.ticker.isin(THIN)].copy()
    return df.sort_values("fri_date").reset_index(drop=True)

def add_features(df, entry="p_mon0630"):
    """gap на момент entry vs пятничный close акции + market/idio разложение."""
    d = df.dropna(subset=[entry, "fri_close_stock", "mon_open_stock", "mon_close_stock"]).copy()
    d["entry_perp"] = d[entry]
    d["gap"] = d.entry_perp / d.fri_close_stock - 1
    # market gap (SP500-перп от пт close до того же entry-момента)
    sx_entry = "spx_" + entry.split("p_")[1]
    if sx_entry in d.columns:
        d["mkt_gap"] = d[sx_entry] / d["spx_fri_close"] - 1
    else:
        d["mkt_gap"] = 0.0
    # beta из historical hourly (вычисляется один раз через compute_betas)
    betas = compute_betas()
    d["beta"] = d.ticker.map(betas).fillna(1.0)
    d["idio_gap"] = d["gap"] - d["beta"] * d["mkt_gap"]          # необъяснённая рынком часть
    d["explained_frac"] = 1 - (d["idio_gap"].abs() / d["gap"].abs().replace(0, np.nan))
    # реальные исходы (для PnL/таргетов — это БУДУЩЕЕ, не feature)
    d["open_ret"] = d.mon_open_stock / d.fri_close_stock - 1
    d["mon_ret"]  = d.mon_close_stock / d.fri_close_stock - 1
    return d

_BETAS = None
def compute_betas():
    """beta тикера vs SP500-перп на 1h ln-доходностях (один раз, кэш)."""
    global _BETAS
    if _BETAS is not None:
        return _BETAS
    h = pd.read_csv(PERP_HOURLY)
    piv = h.pivot_table(index="t_open_ms", columns="ticker", values="close")
    ret = np.log(piv).diff()
    if "SP500" not in ret.columns:
        _BETAS = {}; return _BETAS
    m = ret["SP500"]
    out = {}
    for c in ret.columns:
        if c == "SP500": continue
        xy = pd.concat([ret[c], m], axis=1).dropna()
        if len(xy) > 50 and xy.iloc[:,1].var() > 0:
            out[c] = float(np.cov(xy.iloc[:,0], xy.iloc[:,1])[0,1] / xy.iloc[:,1].var())
    _BETAS = out
    return out

def pnl_series(d, signal_fn, exit="p_mon0930", cost=RT_COST):
    """
    signal_fn(row)-> -1 short / +1 long / 0 skip. Видит только decision-поля.
    Возвращает Series чистого per-trade PnL (доля), индекс = индекс d.
    """
    d = d.dropna(subset=[exit]).copy()
    sides = d.apply(lambda r: signal_fn(r), axis=1)
    d = d.assign(_side=sides)
    d = d[d._side != 0].copy()
    if len(d) == 0:
        return pd.Series([], dtype=float)
    raw = d._side * (d[exit] / d.entry_perp - 1)   # long: (exit-entry)/entry; short: обратное
    net = raw - cost
    net.index = d.index
    return net

def walk_forward(d, signal_fn, exit="p_mon0930", cost=RT_COST, train_frac=0.6):
    """train = ранние выходные, test = свежие. Возвращает метрики обоих + общий."""
    dates = np.sort(d.fri_date.unique())
    cut = dates[int(len(dates)*train_frac)]
    tr = d[d.fri_date < cut]; te = d[d.fri_date >= cut]
    def metr(sub, label):
        pn = pnl_series(sub, signal_fn, exit, cost)
        if len(pn)==0:
            return {f"{label}_n":0, f"{label}_exp":0.0, f"{label}_wr":0.0, f"{label}_sum":0.0, f"{label}_sharpe":0.0}
        return {f"{label}_n":int(len(pn)), f"{label}_exp":float(pn.mean()),
                f"{label}_wr":float((pn>0).mean()), f"{label}_sum":float(pn.sum()),
                f"{label}_sharpe":float(pn.mean()/pn.std()) if pn.std()>0 else 0.0}
    out = {}; out.update(metr(tr,"train")); out.update(metr(te,"test")); out.update(metr(d,"all"))
    out["test_p"] = bootstrap_p(pnl_series(te, signal_fn, exit, cost))
    out["cut_date"] = str(pd.Timestamp(cut).date())
    return out

def bootstrap_p(pnl, n=2000, seed=0):
    """Одностор. p: доля бутстрэп-выборок где mean<=0 (хотим p<0.05 для +ev)."""
    pnl = np.asarray(pnl, float)
    if len(pnl) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(pnl, size=(n, len(pnl)), replace=True).mean(axis=1)
    return float((means <= 0).mean())

def kfold(d, signal_fn, exit="p_mon0930", cost=RT_COST, k=3):
    """k последовательных блоков по времени -> expectancy каждого (стабильность знака)."""
    dates = np.sort(d.fri_date.unique())
    blocks = np.array_split(dates, k)
    res = []
    for b in blocks:
        sub = d[d.fri_date.isin(b)]
        pn = pnl_series(sub, signal_fn, exit, cost)
        res.append(float(pn.mean()) if len(pn) else 0.0)
    return res
