"""country_slice.py -- Ukraine and Sudan results at every training share, from the saved share-grid forecasts. CPU only.

No model is refit. The share grid already forecasts every test cell-month of the held-out region (Jan 2022 .. Apr 2025
forecast months), which includes Ukraine (Europe) and Sudan (Africa). This script scores those forecasts on the rows
whose own country label is the target country (the same row-level label the region filter uses).
Arms
  finetune  ft_pretrained vs ft_scratch           preds_finetune_{region}_s{seed}_f{share}{TSUF}.npz
  nested    emb_base vs base (frozen embedding)   preds_nested_{region}_s{seed}.npz  (incidence only; no civilian nested grid)
Every arm is checked first: the full-region PR-AUC recomputed from the saved forecasts must equal the reported one.
Training data are the region's shares (not country-specific); this is a country slice of the regional design, not an
onset-adaptation test. Rows are country-labelled grid-months (border cells follow their row label), and the slice covers
all test months 2022-2025 (for Sudan this includes the months before the April 2023 war).
Env: SLICE_TARGET (incidence | civilian). Output: outputs/country_slice_{target}.csv (one row per country x arm x
model x share x seed x draw) and a printed summary (means over seeds and draws). Never overwritten.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("SLICE_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
TARGET = os.environ.get("SLICE_TARGET", "incidence"); assert TARGET in ("incidence", "civilian"), TARGET
TSUF = "" if TARGET == "incidence" else "_civ"
CASES = {"Ukraine": "europe", "Sudan": "africa"}
SEEDS = [123, 456, 789]
DRAWS = [int(x) for x in os.environ.get("SLICE_DRAWS", "0,1,2,3,4").split(",")]   # all five required on Roar; the synthetic test uses fewer
SHARES = [float(x) for x in os.environ.get("SLICE_SHARES", "0.0025,0.005,0.0075,0.01,0.025,0.05,0.10").split(",")]   # all seven on Roar
OUT = f"{O}/country_slice_{TARGET}.csv"
ALLOW_MISSING = os.environ.get("SLICE_ALLOW_MISSING", "0") == "1"   # synthetic test only; on Roar every run must exist
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"

cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
cm["grid_id"] = cm["grid_id"].astype(np.int64); cm["year_month"] = cm["year_month"].astype(str).str[:7]
assert not cm.duplicated(["grid_id", "year_month"]).any()
ctry = cm.set_index(["grid_id", "year_month"])["country"]

rows = []
cohort = {}                                          # country -> (grid_id, year_month, y): identical for every file used


def load_checked(tag, country):
    """Forecast file + reported metrics, with key, label-coverage and cohort checks (Codex 9, C02)."""
    fp, fm = f"{O}/share_grid/preds_{tag}.npz", f"{O}/share_grid/metrics_{tag}.csv"
    if not os.path.exists(fp):
        assert ALLOW_MISSING, f"missing run {tag}"
        print(f"[missing] {tag}", flush=True); return None
    z = np.load(fp); met = pd.read_csv(fm)
    gid = z["__grid_id"].astype(np.int64); ym = z["__year_month"].astype(str); y = z["__y"].astype(int)
    key = pd.MultiIndex.from_arrays([gid, ym])
    assert key.is_unique, f"{tag}: duplicate forecast keys"
    lab = ctry.reindex(key).to_numpy()
    assert not pd.isna(lab).any(), f"{tag}: {int(pd.isna(lab).sum())} test rows without a country label"
    if country in cohort:
        g0, m0, y0 = cohort[country]
        assert np.array_equal(gid, g0) and np.array_equal(ym, m0) and np.array_equal(y, y0), f"{tag}: test rows differ"
    else:
        cohort[country] = (gid, ym, y)
    mask = lab == country
    assert mask.sum() > 0 and 0 < y[mask].sum() < mask.sum(), f"{tag}: {country} slice has no rows or one class"
    return z, met, y, mask


def ne_shares(rg, arm):
    """Shares that the jobs' availability files mark not estimable for every seed and draw (civilian floors)."""
    out = set()
    for sh in SHARES:
        st = []
        for sd in SEEDS:
            fa = f"{O}/share_grid/avail_{arm}_{rg}_s{sd}_f{sh:g}{TSUF}.csv"
            if os.path.exists(fa): st += pd.read_csv(fa).status.tolist()
            else: st += ["ok"]
        if all(x != "ok" for x in st): out.add(sh)
        else: assert all(x == "ok" for x in st), f"{rg} {sh:g}: mixed availability"
    return out


def score(country, arm, model, share, seed, draw, y, p, mask, full_pr):
    assert np.isfinite(p).all() and len(p) == len(y)
    ys = y[mask]
    rows.append(dict(target=TARGET, country=country, arm=arm, model=model, share=share, seed=seed, draw=draw,
                     n=int(mask.sum()), pos=int(ys.sum()), pr=average_precision_score(ys, p[mask]), region_pr=full_pr))


expected = []                                         # (country, arm, share, seed, draw) that must have both models
for country, rg in CASES.items():
    ne = ne_shares(rg, "finetune")
    for sd in SEEDS:
        # ---- fine-tune arm ----
        for sh in SHARES:
            if sh in ne: print(f"[not estimable] {country} finetune share={sh:g}", flush=True); continue
            r_ = load_checked(f"finetune_{rg}_s{sd}_f{sh:g}{TSUF}", country)
            if r_ is None: continue
            z, met, y, mask = r_
            for k in DRAWS:
                expected.append((country, "finetune", sh, sd, k))
                for m in ("ft_pretrained", "ft_scratch"):
                    key = f"{m}|{sh:g}|{k}"
                    assert key in z.files, f"finetune {rg} s{sd} {key} missing"
                    p = z[key].astype(np.float64)
                    rep = met[(met.model == m) & np.isclose(met.share, sh) & (met.draw == k)]["pr"]
                    assert len(rep) == 1 and abs(average_precision_score(y, p) - rep.iloc[0]) < 1e-5, f"{rg} s{sd} {key}"
                    score(country, "finetune", m, sh, sd, k, y, p, mask, rep.iloc[0])
        # ---- nested arm (frozen embedding); only incidence has a nested share grid ----
        if TARGET != "incidence": continue
        r_ = load_checked(f"nested_{rg}_s{sd}", country)
        if r_ is None: continue
        z, met, y, mask = r_
        for sh in SHARES:
            for k in DRAWS:
                expected.append((country, "nested", sh, sd, k))
                for m in ("base", "emb_base"):
                    key = f"{m}|{sh:g}|{k}"
                    assert key in z.files, f"nested {rg} s{sd} {key} missing"          # Codex 9, C01: never skipped
                    p = z[key].astype(np.float64)
                    rep = met[(met.model == m) & np.isclose(met.share, sh) & (met.draw == k)]["pr"]
                    assert len(rep) == 1 and abs(average_precision_score(y, p) - rep.iloc[0]) < 1e-5, f"{rg} s{sd} {key}"
                    score(country, "nested", m, sh, sd, k, y, p, mask, rep.iloc[0])

D = pd.DataFrame(rows)
assert len(D), "no forecasts found"
w = D.pivot_table(index=["country", "arm", "share", "seed", "draw"], columns="model", values="pr")
for c, a, sh, sd, k in expected:                     # every expected pair complete before anything is published
    hi, lo = ("ft_pretrained", "ft_scratch") if a == "finetune" else ("emb_base", "base")
    assert np.isfinite(w.loc[(c, a, sh, sd, k), [hi, lo]]).all(), f"incomplete pair {c} {a} {sh} s{sd} d{k}"
assert len(w) == len(expected), f"{len(w)} pairs scored, {len(expected)} expected"
D.to_csv(OUT + ".part", index=False); os.replace(OUT + ".part", OUT)
out = []
for (c, a), g in w.groupby(level=["country", "arm"]):
    hi, lo = ("ft_pretrained", "ft_scratch") if a == "finetune" else ("emb_base", "base")
    gg = (g[hi] - g[lo]).groupby(level="share")
    out.append(pd.DataFrame({"country": c, "arm": a, "gain": gg.mean(), "share_positive": gg.apply(lambda x: (x > 0).mean()),
                             "n_pairs": gg.size(), hi: g[hi].groupby(level="share").mean(), lo: g[lo].groupby(level="share").mean()}))
pd.set_option("display.width", 180)
print(pd.concat(out).round(4).to_string(), flush=True)
print(f"[done] {OUT}", flush=True)
