"""ens_alerts_union.py -- deaths in cell-months warned only by the ensemble, counted once across targets (CPU).
(Forecasting with a pretrained encoding; not the LORO transfer design.)

Same alert rule as ens_alerts.py: for each target, ens_mean and base (XGBoost) each flag k = number of test events;
gained_t = event cell-months flagged by ens_mean only, lost_t = flagged by base only.
Union over targets T (default incidence, civilian, onset; fatal25 reported separately because it is dominated by
Ukraine battle deaths): a cell-month (grid_id, forecast month) is
  ens_only  if it is in gained_t for some t and in lost_t for no t
  base_only if it is in lost_t for some t and in gained_t for no t
  mixed     if both (reported, excluded from net)
Deaths are counted once per cell-month at the outcome month (t+1): civ_deaths = ACLED deaths_civilians summed over
events; fatalities = panel count_fatalities. These are deaths that occurred where only one model warned; they are
not lives saved.
Breakdowns: ALL, region, country (all countries with any ens_only or base_only cell-month).
Intervals: paired 5x5-degree block bootstrap stratified by region, B=1000, seed 123 (flags held fixed).
Output (never overwritten): outputs/ens/alerts_union_mt{RTAG}_{targets}.csv
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
OD = f"{R}/outputs/ens"
RTAG = os.environ.get("ENS_RTAG", "_pilot_s123-456-789")
B = int(os.environ.get("ENS_BOOT_B", "1000"))
SEED = 123
TARGETS = os.environ.get("ENS_UNION_TARGETS", "incidence,civilian,onset").split(",")
assert set(TARGETS) <= {"incidence", "civilian", "onset", "fatal25"} and len(TARGETS) == len(set(TARGETS))
OUT_CSV = f"{OD}/alerts_union_mt{RTAG}_{'-'.join(TARGETS)}.csv"
assert not os.path.exists(OUT_CSV), f"refusing to overwrite {OUT_CSV}"


def next_month(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}"


panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "count_fatalities", "country"])
panel["year_month"] = panel["year_month"].astype(str)
assert not panel.duplicated(["grid_id", "year_month"]).any()
cur = panel.set_index(["grid_id", "year_month"])
ac = pd.read_parquet(f"{R}/data_intermediate/acled_with_grid.parquet", columns=["grid_id", "year_month", "fatalities", "deaths_civilians"])
ac["year_month"] = ac["year_month"].astype(str)
ac = ac[ac["year_month"] >= "2022-01"]
ac = ac.assign(deaths_civilians=pd.to_numeric(ac["deaths_civilians"], errors="coerce"), fatalities=pd.to_numeric(ac["fatalities"], errors="coerce"))
assert ac["deaths_civilians"].notna().all() and not (ac["deaths_civilians"] > ac["fatalities"]).any()
civ = ac.groupby(["grid_id", "year_month"])["deaths_civilians"].sum()

gained, lost, region_of = set(), set(), {}
for t in TARGETS:
    z = np.load(f"{OD}/preds_ens_mt_{t}{RTAG}.npz")
    te = z["window"].astype(str) == "test"
    g = z["grid_id"][te].astype(np.int64); ym = z["year_month"][te].astype(str); y = z["y"][te].astype(np.int64)
    rg = z["region"][te].astype(str)
    p_e, p_b = z["p_ens_mean"][te].astype(np.float64), z["p_base"][te].astype(np.float64)
    k = int(y.sum())
    fe, fb = np.zeros(len(y), bool), np.zeros(len(y), bool)
    fe[np.argsort(-p_e, kind="stable")[:k]] = True
    fb[np.argsort(-p_b, kind="stable")[:k]] = True
    ga, lo = fe & ~fb & (y == 1), fb & ~fe & (y == 1)
    gained |= set(zip(g[ga].tolist(), ym[ga].tolist()))
    lost |= set(zip(g[lo].tolist(), ym[lo].tolist()))
    for a, b_, r in zip(g.tolist(), ym.tolist(), rg.tolist()):
        if region_of.setdefault((a, b_), r) != r:
            raise AssertionError(f"region differs across targets for {(a, b_)}")
    print(f"[{t}] gained {int(ga.sum())} lost {int(lo.sum())}", flush=True)

mixed = gained & lost
ens_only, base_only = gained - mixed, lost - mixed
cells = sorted(ens_only | base_only)
df = pd.DataFrame(cells, columns=["grid_id", "year_month"])
df["side"] = np.where([c in ens_only for c in cells], 1, -1)
df["region"] = [region_of[c] for c in cells]
df["ym_next"] = [next_month(m) for m in df["year_month"]]
kn = pd.MultiIndex.from_arrays([df["grid_id"], df["ym_next"]])
nx = cur.reindex(kn)
assert nx["count_fatalities"].notna().all(), "outcome month missing from panel"
df["fatalities"] = nx["count_fatalities"].to_numpy(float)
df["civ_deaths"] = civ.reindex(kn).fillna(0).to_numpy(float)
df["country"] = cur.reindex(pd.MultiIndex.from_arrays([df["grid_id"], df["year_month"]]))["country"].astype(str).to_numpy()
df["cellmonths"] = 1.0
rowi, coli = (df["grid_id"] - 1) // 720, (df["grid_id"] - 1) % 720
df["cl"] = (rowi // 10) * 72 + (coli // 10)
codes, uniq = pd.factorize(pd.Series(list(zip(df["region"], df["cl"]))))
reg_of = np.array([u[0] for u in uniq])
regs = sorted(set(df["region"]))
rng = np.random.default_rng([SEED, 7])
W = np.empty((B, len(uniq)))
for b in range(B):
    cnt = np.zeros(len(uniq))
    for r in regs:
        c = np.flatnonzero(reg_of == r)
        cnt += np.bincount(rng.choice(c, size=len(c), replace=True), minlength=len(uniq))
    W[b] = cnt

rows = []
groups = [("ALL", "ALL", np.ones(len(df), bool))] + [("region", r, (df["region"] == r).to_numpy()) for r in regs]
groups += [("country", c, (df["country"] == c).to_numpy()) for c in sorted(set(df["country"]))]
for gt, gn, s in groups:
    rec = dict(targets="-".join(TARGETS), group_type=gt, group=gn, n_mixed_all=len(mixed))
    for v in ("cellmonths", "civ_deaths", "fatalities"):
        x = df[v].to_numpy() * s
        e, bse = (x * (df["side"] == 1)).sum(), (x * (df["side"] == -1)).sum()
        rec[f"{v}_ens_only"], rec[f"{v}_base_only"], rec[f"{v}_net"] = e, bse, e - bse
        d = W @ np.bincount(codes, weights=x * df["side"].to_numpy(), minlength=len(uniq))
        rec[f"{v}_net_lo"], rec[f"{v}_net_hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
    rows.append(rec)
out = pd.DataFrame(rows)
chk = out[out.group_type == "region"][["cellmonths_net", "civ_deaths_net", "fatalities_net"]].sum()
assert np.allclose(chk.values, out[out.group_type == "ALL"][["cellmonths_net", "civ_deaths_net", "fatalities_net"]].values[0])
out.to_csv(OUT_CSV + ".part", index=False)
os.replace(OUT_CSV + ".part", OUT_CSV)
print(f"mixed cell-months (excluded): {len(mixed)}")
print(out[out.group_type != "country"].round(0).to_string(index=False))
print(f"[done] {OUT_CSV}")
