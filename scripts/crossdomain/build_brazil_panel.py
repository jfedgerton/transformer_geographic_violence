"""build_brazil_panel.py (2026-10-23) -- cross-domain test C3: Brazil SIM homicides -> PRIO-GRID cell-month panel
in the conflict encoder's own input format (scripts/phase4_scaling/build_sequences_super.py).

Construction rules (frozen before any result; crossdomain_plan_2026-10-23.md):
  unit      PRIO-GRID 0.5 deg cell (municipality of occurrence -> IBGE centroid -> cell; residence used only if
            occurrence is missing, flagged); fixed cell set = every cell holding at least one municipality centroid
  time      month; panel 2000-01 .. 2024-12 (12 history months needed before the 2001+ rows used downstream)
  events    one event per death. ICD-10 X85-Y09 (assault) and Y35 (legal intervention)
  mapping   MINIMAL: assault -> Violence against civilians / Attack / Unidentified Armed Group (Brazil) [political_militia]
                     -> Civilians (Brazil) [civilians]
                     police (Y35) -> Battles / Armed clash / Military Forces of Brazil (<period>) Military Police [state]
                     -> Unidentified Gang (Brazil) [political_militia]  (ACLED's coding of Brazilian police shootouts)
            RICH:    as MINIMAL, but assault with firearm (X93-X95) on a male victim aged 15-29 (the modal gang
                     homicide) -> actor1 Unidentified Gang (Brazil)
            fatalities bin = "1-2" (one death); civilian-death bin = "1-2" for assault, "0" for police events
  sequence  events in months t-12..t-1, oldest first, month_offset = t - month (1..12); at most CAP_PER_MONTH events
            per cell-month, sampled with seed 123 (keeps all 12 months under the encoder's 200-token limit)
  targets   deaths_next = deaths in t+1 (count; primary); surge_next = deaths_next > mean(deaths t-11..t), defined
            only where that trailing mean >= SURGE_FLOOR (surge_eligible); persistence inputs deaths_t, surge_t
  base      lags 1..12 of total, police, firearm, gang-pattern deaths (t-1..t-12, as in the conflict base),
            deaths_t is NOT a base feature (the conflict base stops at t-1); calendar month, year; train-period cell
            mean (2001-2020) -- statics are merged at evaluation time from priogrid_static.parquet
Outputs: data_intermediate/xd_brazil_panel.parquet, outputs/crossdomain/xd_brazil_audit.json
Env XD_TEST=1 runs on synthetic inputs passed via XD_EVENTS / XD_CENTROIDS / XD_VOCAB / XD_OUTDIR.
"""
import os, json
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
TEST = os.environ.get("XD_TEST", "0") == "1"
EVENTS = os.environ.get("XD_EVENTS", f"{R}/data_intermediate/sim_homicide_events.parquet")
CENTROIDS = os.environ.get("XD_CENTROIDS", f"{R}/data_raw/br_centroids/municipios.csv")
VOCAB = os.environ.get("XD_VOCAB", f"{R}/data_intermediate/vocab_super.json")
OUTDIR = os.environ.get("XD_OUTDIR", R)
OUT = f"{OUTDIR}/data_intermediate/xd_brazil_panel.parquet"
AUD = f"{OUTDIR}/outputs/crossdomain/xd_brazil_audit.json"
PANEL_START, PANEL_END = "2000-01", "2024-12"
TRAIN_MEAN_START, TRAIN_MEAN_END = "2001-01", "2020-12"
H, CAP_PER_MONTH, SURGE_FLOOR, SEED = 12, 16, 1.0, 123
os.makedirs(os.path.dirname(OUT), exist_ok=True); os.makedirs(os.path.dirname(AUD), exist_ok=True)
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
rng = np.random.default_rng(SEED)
audit = {}


def gid_of(lat, lon):
    """PRIO-GRID id: row from the south pole, col from -180, 720 cols (matches bootstrap_ci.py block code)."""
    row = np.floor((np.asarray(lat) + 90.0) / 0.5).astype(int)
    col = np.floor((np.asarray(lon) + 180.0) / 0.5).astype(int)
    return row * 720 + col + 1


# ---- PRIO id formula check against a file that stores cell centers ----
if not TEST:
    tg = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet",
                         columns=["grid_id", "grid_lat_center", "grid_lon_center"]).drop_duplicates("grid_id")
    assert (gid_of(tg.grid_lat_center, tg.grid_lon_center) == tg.grid_id.values).all(), "PRIO gid formula mismatch"
    audit["gid_formula_checked_cells"] = int(len(tg))

# ---- vocabulary ids (must exist; fail loudly) ----
V = json.load(open(VOCAB))
def vid(field, key):
    assert key in V[field], f"vocab {field} lacks '{key}'"
    return int(V[field][key])
ET_VAC, ET_BAT = vid("event_type", "Violence against civilians"), vid("event_type", "Battles")
SE_ATT, SE_CLASH = vid("sub_event_type", "Attack"), vid("sub_event_type", "Armed clash")
A1_UAG = vid("actor1", "Unidentified Armed Group (Brazil)")
A1_GANG = vid("actor1", "Unidentified Gang (Brazil)")
A2_CIV = vid("actor2", "Civilians (Brazil)")
A2_GANG = vid("actor2", "Unidentified Gang (Brazil)")
MP = {"2016": vid("actor1", "Military Forces of Brazil (2016-2018) Military Police"),
      "2019": vid("actor1", "Military Forces of Brazil (2019-2022) Military Police"),
      "2023": vid("actor1", "Military Forces of Brazil (2023-) Military Police")}
T_STATE, T_MIL, T_CIV = vid("actor1_type", "state"), vid("actor1_type", "political_militia"), vid("actor2_type", "civilians")
T2_MIL = vid("actor2_type", "political_militia")
assert V["fatalities"]["1-2"] == 2 and V["deaths_civilians"]["0"] == 1 and V["deaths_civilians"]["1-2"] == 2
assert [int(V["month_offset"][str(m)]) for m in range(1, 13)] == list(range(1, 13))

# ---- events -> cell, month, tokens ----
ev = pd.read_parquet(EVENTS)
n0 = len(ev)
cen = pd.read_csv(CENTROIDS, dtype={"codigo_ibge": str})
cen["mun6"] = cen["codigo_ibge"].str[:6]
assert cen["mun6"].is_unique, "6-digit IBGE codes not unique"
cen["grid_id"] = gid_of(cen["latitude"], cen["longitude"])
mun2gid = dict(zip(cen["mun6"], cen["grid_id"]))
occ = ev["CODMUNOCOR"].astype(str).str.strip().str[:6]
res = ev["CODMUNRES"].astype(str).str.strip().str[:6] if "CODMUNRES" in ev else pd.Series("", index=ev.index)
g_occ, g_res = occ.map(mun2gid), res.map(mun2gid)
ev["grid_id"] = g_occ.fillna(g_res)
ev["geo_from_residence"] = g_occ.isna() & g_res.notna()
audit["events_in"] = int(n0)
audit["share_geo_from_residence"] = float(ev["geo_from_residence"].mean())
audit["share_unlocated_dropped"] = float(ev["grid_id"].isna().mean())
assert audit["share_unlocated_dropped"] < 0.01, f"too many unlocated deaths: {audit['share_unlocated_dropped']:.4f}"
ev = ev[ev["grid_id"].notna() & ev["date"].notna()].copy()
ev["grid_id"] = ev["grid_id"].astype(np.int64)
ev["ym"] = ev["date"].dt.to_period("M").astype(str)
ev = ev[(ev["ym"] >= PANEL_START) & (ev["ym"] <= PANEL_END)].copy()

cause = ev["cause3"].astype(str)
police = ev["is_police"].astype(bool).values
firearm = cause.between("X93", "X95").values
sex = ev["SEXO"].astype(str).str.strip() if "SEXO" in ev else pd.Series("", index=ev.index)
male = sex.isin(["1", "M"]).values
idade = ev["IDADE"].astype(str).str.strip() if "IDADE" in ev else pd.Series("", index=ev.index)
age = np.where(idade.str.len().eq(3) & idade.str[0].eq("4"), pd.to_numeric(idade.str[1:], errors="coerce"), np.nan)
gang_pattern = (~police) & firearm & male & (age >= 15) & (age <= 29)
yr = ev["date"].dt.year.values
mp_id = np.where(yr >= 2023, MP["2023"], np.where(yr >= 2019, MP["2019"], MP["2016"]))

et = np.where(police, ET_BAT, ET_VAC)
se = np.where(police, SE_CLASH, SE_ATT)
a1_min = np.where(police, mp_id, A1_UAG)
a1_rich = np.where(police, mp_id, np.where(gang_pattern, A1_GANG, A1_UAG))
a2 = np.where(police, A2_GANG, A2_CIV)
a1t = np.where(police, T_STATE, T_MIL)
a2t = np.where(police, T2_MIL, T_CIV)
fat = np.full(len(ev), 2)
civ = np.where(police, 1, 2)
ev = ev.assign(et=et, se=se, a1_min=a1_min, a1_rich=a1_rich, a2=a2, a1t=a1t, a2t=a2t, fat=fat, civ=civ,
               police=police.astype(int), firearm=firearm.astype(int), gang=gang_pattern.astype(int))
audit["events_kept"] = int(len(ev))
audit["share_police"] = float(police.mean()); audit["share_firearm"] = float(firearm.mean())
audit["share_gang_pattern"] = float(gang_pattern.mean())
if not TEST:
    assert 0.05 < audit["share_gang_pattern"] < 0.8, f"gang-pattern share {audit['share_gang_pattern']:.3f}: SEXO/IDADE parsing failed?"
    assert audit["share_police"] > 0, "no police (Y35) deaths"

# ---- balanced panel over the fixed cell set ----
months = pd.period_range(PANEL_START, PANEL_END, freq="M").astype(str)
m_idx = {m: i for i, m in enumerate(months)}
cells = np.sort(cen["grid_id"].unique())
ev = ev[ev["grid_id"].isin(cells)].copy()
ev["t"] = ev["ym"].map(m_idx).astype(int)
cnt = ev.groupby(["grid_id", "t"]).agg(deaths=("t", "size"), police=("police", "sum"),
                                      firearm=("firearm", "sum"), gang=("gang", "sum"))
full = pd.MultiIndex.from_product([cells, range(len(months))], names=["grid_id", "t"])
P = cnt.reindex(full, fill_value=0).reset_index()
P["year_month"] = np.asarray(months)[P["t"].values]
audit["cells"] = int(len(cells)); audit["months"] = int(len(months)); audit["rows"] = int(len(P))

# ---- per-cell-month cap, then sequences t-12..t-1 ----
ev = ev.sort_values(["grid_id", "t"]).reset_index(drop=True)
keep = np.ones(len(ev), bool)
over = 0
for (_, _), idx in ev.groupby(["grid_id", "t"]).indices.items():
    if len(idx) > CAP_PER_MONTH:
        over += 1
        drop = rng.choice(idx, size=len(idx) - CAP_PER_MONTH, replace=False)
        keep[drop] = False
audit["cell_months_over_cap"] = int(over)
audit["share_event_cell_months_over_cap"] = float(over / max(1, len(cnt)))
evk = ev[keep]
cols_min = ["et", "se", "a1_min", "a2", "a1t", "a2t"]
cols_rich = ["et", "se", "a1_rich", "a2", "a1t", "a2t"]
by_cell = {g: d for g, d in evk.groupby("grid_id")}
seq_min, seq_rich, seq_len = [], [], []
for g, t in zip(P["grid_id"].values, P["t"].values):
    d = by_cell.get(g)
    if d is None or t - H < 0:
        seq_min.append("[]"); seq_rich.append("[]"); seq_len.append(0); continue
    w = d[(d["t"] >= t - H) & (d["t"] <= t - 1)]
    if len(w) == 0:
        seq_min.append("[]"); seq_rich.append("[]"); seq_len.append(0); continue
    off = (t - w["t"].values).astype(int)
    A = w[cols_min].values; B = w[cols_rich].values; F = w[["fat", "civ"]].values
    seq_min.append(json.dumps([[*map(int, A[j]), int(off[j]), *map(int, F[j])] for j in range(len(w))]))
    seq_rich.append(json.dumps([[*map(int, B[j]), int(off[j]), *map(int, F[j])] for j in range(len(w))]))
    seq_len.append(len(w))
P["seq_min"], P["seq_rich"], P["seq_len"] = seq_min, seq_rich, seq_len
assert P["seq_len"].max() <= H * CAP_PER_MONTH <= 200, "sequence longer than the encoder limit"

# ---- targets, persistence inputs, base features ----
P = P.sort_values(["grid_id", "t"]).reset_index(drop=True)
gb = P.groupby("grid_id", sort=False)
P["deaths_t"] = P["deaths"]
P["deaths_next"] = gb["deaths"].shift(-1)
tm = gb["deaths"].transform(lambda s: s.rolling(12, min_periods=12).mean())     # mean of t-11..t
P["trail_mean"] = tm
P["surge_eligible"] = (tm >= SURGE_FLOOR).astype(int)
P["surge_next"] = np.where(P["surge_eligible"].eq(1) & P["deaths_next"].notna(),
                           (P["deaths_next"] > tm).astype(float), np.nan)
tm_prev = gb["deaths"].transform(lambda s: s.shift(1).rolling(12, min_periods=12).mean())  # mean of t-12..t-1
P["surge_t"] = np.where(tm_prev >= SURGE_FLOOR, (P["deaths"] > tm_prev).astype(float), np.nan)
for v in ["deaths", "police", "firearm", "gang"]:
    for k in range(1, H + 1):
        P[f"lag{k}_{v}"] = gb[v].shift(k)
P["cal_month"] = P["year_month"].str[5:7].astype(int)
P["year"] = P["year_month"].str[:4].astype(int)
trn = (P["year_month"] >= TRAIN_MEAN_START) & (P["year_month"] <= TRAIN_MEAN_END)
P["cell_train_mean"] = P["grid_id"].map(P.loc[trn].groupby("grid_id")["deaths"].mean())
P = P.drop(columns=["deaths", "police", "firearm", "gang"])
P = P[P["deaths_next"].notna()].copy()                                       # last month has no target
P["deaths_next"] = P["deaths_next"].astype(int)
assert not P.duplicated(["grid_id", "year_month"]).any()
# timing checks on a sample: lag1 == deaths at t-1, target == deaths at t+1
s = P.sample(min(2000, len(P)), random_state=SEED)
look = P.set_index(["grid_id", "t"])["deaths_t"]
for r in s.itertuples():
    if r.t >= 1 and (r.grid_id, r.t - 1) in look.index:
        assert r.lag1_deaths == look[(r.grid_id, r.t - 1)], "lag1 misaligned"
    if (r.grid_id, r.t + 1) in look.index:
        assert r.deaths_next == look[(r.grid_id, r.t + 1)], "target misaligned"

for lo, hi, name in [("2001-01", "2020-12", "train"), ("2021-01", "2021-12", "val"), ("2022-01", "2024-11", "test")]:
    m = (P["year_month"] >= lo) & (P["year_month"] <= hi)
    audit[f"{name}_rows"] = int(m.sum())
    audit[f"{name}_mean_deaths_next"] = float(P.loc[m, "deaths_next"].mean())
    audit[f"{name}_share_any_death_next"] = float((P.loc[m, "deaths_next"] > 0).mean())
    audit[f"{name}_surge_eligible_share"] = float(P.loc[m, "surge_eligible"].mean())
    audit[f"{name}_surge_rate"] = float(np.nanmean(P.loc[m, "surge_next"]))
P.to_parquet(OUT, index=False)
json.dump(audit, open(AUD, "w"), indent=2)
print(json.dumps(audit, indent=2)); print(f"[done] {len(P):,} rows -> {OUT}", flush=True)
