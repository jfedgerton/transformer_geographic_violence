"""make_share_tables.py -- builds every limited-data table from outputs/share_grid/metrics_*.csv.

No model is run here. Each table cell = mean over 5 draws x 3 encoder seeds of a within-run difference
(same rows, same test set), with min, max and the share of runs where the difference is positive.
Checks (fail loudly):
  * every expected (unit, seed, share, draw) run exists for each requested arm family, none unexpected, no duplicates
  * paired arms share n_train, pos_train, n_test, test_pos
  * every metrics file has its preds file, and all jobs for a unit used identical test rows (__grid_id/__year_month or __t/__row)
  * the base arm is identical across encoder seeds (it never sees the encoder)
  * persistence PR-AUC matches across all jobs for a unit
Outputs (outputs/share_grid/tables/):
  nested_{domain}.csv / finetune_{domain}.csv   long summaries
  tab_nested_{domain}.tex / tab_finetune_{domain}.tex   region x share means (with positive-run share)
Vocabulary: by default only main-specification runs (vocab remap on) are read; --vorig reads only the original-vocabulary
  runs (job names containing "_vorig") and writes tables with a "_vorig" suffix. The two are never mixed.
Usage: python make_share_tables.py [conflict|crime|dengue|ces] [--families nested,finetune] [--allow-missing] [--vorig]
"""
import sys, glob, os
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OD = f"{R}/outputs/share_grid"; TD = f"{OD}/tables"; os.makedirs(TD, exist_ok=True)
DOM = sys.argv[1] if len(sys.argv) > 1 else "conflict"; ALLOW_MISSING = "--allow-missing" in sys.argv
VORIG = "--vorig" in sys.argv; VSUF = "_vorig" if VORIG else ""
FAMS = sys.argv[sys.argv.index("--families") + 1].split(",") if "--families" in sys.argv else ["nested", "finetune"]
assert set(FAMS) <= {"nested", "finetune"} and FAMS, FAMS
SHARES = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.10]; DRAWS = [0, 1, 2, 3, 4]; SEEDS = [123, 456, 789]
if DOM == "conflict":
    UNITS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
    LABEL = {"africa": "Africa", "mena": "MENA", "europe": "Europe", "americas": "Americas",
             "south_asia": "South Asia", "asia_pacific": "Asia-Pacific"}
    files = [f for f in glob.glob(f"{OD}/metrics_*.csv") if "_smoke" not in f
             and not os.path.basename(f).startswith(("metrics_crime_", "metrics_dengue_", "metrics_ces_"))]
    UCOL = "region"
else:
    UNITS = {"crime": ["Boston", "St Louis", "Seattle", "Austin", "Los Angeles"],
             "dengue": [f"region{i}" for i in range(1, 6)],
             "ces": ["New England", "Middle Atlantic", "East North Central", "West North Central", "South Atlantic",
                     "East South Central", "West South Central", "Mountain", "Pacific"]}[DOM]
    LABEL = {u: u for u in UNITS}
    files = [f for f in glob.glob(f"{OD}/metrics_{DOM}_*.csv") if "_smoke" not in f]
    UCOL = "unit"
files = [f for f in files if ("_vorig" in os.path.basename(f)) == VORIG]
assert files, f"no metrics files for {DOM} (vorig={VORIG})"
assert not [f for f in files if f.endswith(".part")], "unfinished .part files matched the glob"
# every finished metrics file must have its forecasts, and test rows must be identical within a unit
ids = {}
for f in files:
    pf = f.replace("/metrics_", "/preds_").replace(".csv", ".npz")
    assert os.path.exists(pf), f"metrics without preds (interrupted publish?): {f}"
    z = np.load(pf)
    keyset = {"conflict": ("__grid_id", "__year_month"), "ces": ("__t", "__case_id")}.get(DOM, ("__t", "__row"))
    unit = pd.read_csv(f, usecols=[UCOL])[UCOL].iloc[0]
    sig = tuple(z[k].tobytes() for k in keyset) + (z["__y"].tobytes(),)
    if unit in ids: assert ids[unit] == sig, f"test rows differ across jobs for {unit}: {f}"
    else: ids[unit] = sig
m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
m = m.rename(columns={UCOL: "unit"})
print(f"[load] {len(files)} files, {len(m):,} rows", flush=True)

# persistence identical within unit
p = m[m["model"] == "persistence"].groupby("unit")["pr"].agg(["min", "max"])
assert ((p["max"] - p["min"]) < 1e-9).all(), f"persistence differs across jobs:\n{p}"

def paired(a, b, name):
    x = m[m["model"].isin([a, b])].copy()
    key = ["unit", "enc_seed", "share", "draw"]
    assert not x.duplicated(key + ["model"]).any(), f"duplicate rows for {a}/{b}"
    for col in ("n_train", "pos_train", "n_test", "test_pos"):          # paired runs must share rows
        pv = x.pivot_table(index=key, columns="model", values=col, aggfunc="first")
        both = pv.dropna()
        assert np.allclose(both[a], both[b]), f"{name}: {col} differs between {a} and {b} within a run"
    w = x.pivot_table(index=key, columns="model", values="pr", aggfunc="first")
    one_sided = w[w[a].isna() ^ w[b].isna()]
    assert one_sided.empty, f"{name}: runs with only one arm, e.g. {one_sided.index[:5].tolist()}"
    w = w.dropna().reset_index()
    nt = x[x["model"] == a].set_index(key)["n_train"]
    w = w.join(nt, on=key)
    expect = pd.MultiIndex.from_product([UNITS, SEEDS, SHARES, DRAWS], names=key)
    got = pd.MultiIndex.from_frame(w[key])
    extra = got.difference(expect)
    assert not len(extra), f"{name}: {len(extra)} unexpected runs (unit/seed/share/draw outside the design), e.g. {list(extra[:5])}"
    missing = expect.difference(got)
    if len(missing):
        msg = f"{name}: {len(missing)} of {len(expect)} runs missing, e.g. {list(missing[:5])}"
        if ALLOW_MISSING: print("[warn]", msg)
        else: raise AssertionError(msg + "  (skipped draws with <5 positives also appear here; rerun with --allow-missing to inspect)")
    w["delta"] = w[a] - w[b]
    s = w.groupby(["unit", "share"]).agg(mean=("delta", "mean"), lo=("delta", "min"), hi=("delta", "max"),
                                           pos=("delta", lambda v: float((v > 0).mean())), runs=("delta", "size"),
                                           n_train=("n_train", "median"), arm_a=(a, "mean"), arm_b=(b, "mean")).reset_index()
    s.to_csv(f"{TD}/{name}_{DOM}{VSUF}.csv", index=False)
    return s

def tex(s, name, caption):
    lines = [r"\begin{tabular}{@{}l" + "c" * len(SHARES) + "@{}}", r"\toprule",
             " & \\multicolumn{%d}{c}{Share of target training data} \\\\" % len(SHARES),
             r"\cmidrule(l){2-%d}" % (len(SHARES) + 1),
             "Held-out unit & " + " & ".join(f"{100*x:g}\\%" for x in SHARES) + r" \\", r"\midrule"]
    for u in UNITS:
        cells = []
        for x in SHARES:
            r = s[(s["unit"] == u) & np.isclose(s["share"], x)]
            cells.append("--" if r.empty else f"${r['mean'].iloc[0]:+.3f}$")
        lines.append(f"{LABEL[u]} & " + " & ".join(cells) + r" \\")
    pos = []
    for x in SHARES:
        r = s[np.isclose(s["share"], x)]
        pos.append("--" if r.empty else f"{int((r['mean'] > 0).sum())}/{len(r)}")
    lines += [r"\midrule", "Positive (mean) & " + " & ".join(pos) + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(f"{TD}/tab_{name}_{DOM}{VSUF}.tex", "w").write(f"% {caption}\n% generated by make_share_tables.py -- do not edit by hand\n" + "\n".join(lines) + "\n")

present = set(m["model"])
for fam, arms in (("nested", {"base", "emb_base"}), ("finetune", {"ft_pretrained", "ft_scratch"})):
    if fam in FAMS: assert arms <= present, f"{fam}: arms missing entirely: {sorted(arms - present)} (use --families to build one family)"
if "nested" in FAMS:
    b = m[m["model"] == "base"].groupby(["unit", "share", "draw"])["pr"].agg(["min", "max"])
    assert ((b["max"] - b["min"]) < 1e-6).all(), "base arm differs across encoder seeds (should be identical)"
    s = paired("emb_base", "base", "nested"); tex(s, "nested", "PR-AUC(emb_base) - PR-AUC(base), mean over draws x seeds")
    print(s.pivot(index="unit", columns="share", values="mean").round(3).to_string(), flush=True)
if "finetune" in FAMS:
    s = paired("ft_pretrained", "ft_scratch", "finetune"); tex(s, "finetune", "PR-AUC(fine-tuned pretrained) - PR-AUC(scratch), mean over draws x seeds")
    print(s.pivot(index="unit", columns="share", values="mean").round(3).to_string(), flush=True)
print(f"[done] tables in {TD}", flush=True)
