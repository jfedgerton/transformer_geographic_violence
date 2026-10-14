"""cmp_random.py -- random-encoder control summary (2026-10-13; revised after Codex review 4).
For each region x seed, reads the main-spec eval (eval_region_{TAG}.csv) and the random-encoder eval
(eval_region_{TAG}_rand.csv) and reports, by target and mode:
  pre_gain  = emb_base(pretrained) - base
  rand_gain = emb_base(random)     - base
  transfer  = emb_base(pretrained) - emb_base(random)     (gain attributable to pretraining)
Validation BEFORE any averaging (fold level):
  - each file has unique (target, mode, model) rows, finite scores, and exactly the modes A_pooled_f0..f4 + B_temporal
  - both files cover the same (target, mode, model) cells and the same row counts n per cell (identical inputs/folds)
  - base and persistence scores agree within ABS_TOL (they do not use the encoder); the largest gap is always printed
Pooled scores are then averaged over the five folds. Reported targets: incidence and civilian; all targets are validated.
Output: outputs/random_control_summary.csv (per region x seed x target x mode).
"""
import os
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
SEEDS = [123, 456, 789]
MODES = {f"A_pooled_f{k}" for k in range(5)} | {"B_temporal"}
REPORT = ["incidence", "civilian"]
ABS_TOL = 1e-6   # XGBoost heads are seeded; separate jobs on the same node type should agree far below this (Codex 4, #3)

def load(f, tag):
    d = pd.read_csv(f)
    key = ["target", "mode", "model"]
    assert not d.duplicated(key).any(), f"{tag}: duplicate (target, mode, model) rows in {f}"
    assert np.isfinite(d["score"]).all(), f"{tag}: non-finite scores in {f}"
    got = set(d["mode"])
    assert got == MODES, f"{tag}: modes {sorted(got)} != expected {sorted(MODES)} in {f}"
    return d.set_index(key).sort_index()

rows, missing, worst = [], [], 0.0
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        fp, fr = f"{R}/outputs/eval_region_{tag}.csv", f"{R}/outputs/eval_region_{tag}_rand.csv"
        if not (os.path.exists(fp) and os.path.exists(fr)):
            missing.append(tag); continue
        p, r = load(fp, tag), load(fr, tag)
        assert p.index.equals(r.index), f"{tag}: (target, mode, model) cells differ between arms"
        assert (p["n"].values == r["n"].values).all(), f"{tag}: row counts n differ between arms (different inputs/folds)"
        for mdl in ["base", "persistence"]:
            gap = (p.xs(mdl, level="model")["score"] - r.xs(mdl, level="model")["score"]).abs()
            # base/persistence must match exactly on the reported targets; severity_crps is not
            # deterministic across jobs (max gap 7e-4, checked Oct 14) and is not reported
            gap = gap[gap.index.get_level_values("target").isin(REPORT)]
            worst = max(worst, float(gap.max()))
            assert gap.max() <= ABS_TOL, f"{tag}: {mdl} differs between arms by {gap.max():.3g} at {gap.idxmax()}"
        # fold-level checks passed -> average pooled folds
        def collapse(x):
            x = x.reset_index(); x["m"] = np.where(x["mode"] == "B_temporal", "temporal", "pooled")
            return x.pivot_table(index=["target", "m"], columns="model", values="score", aggfunc="mean")
        pc, rc = collapse(p), collapse(r)
        out = pd.DataFrame({"region": rg, "seed": sd, "base": pc["base"],
                            "emb_base_pre": pc["emb_base"], "emb_base_rand": rc["emb_base"]})
        out["pre_gain"] = out.emb_base_pre - out.base
        out["rand_gain"] = out.emb_base_rand - out.base
        out["transfer"] = out.emb_base_pre - out.emb_base_rand
        rows.append(out.reset_index())

print(f"[cmp] pairs found: {len(rows)} of {len(REGIONS) * len(SEEDS)}; missing: {missing}")
print(f"[cmp] largest base/persistence gap across arms: {worst:.3g} (tolerance {ABS_TOL})")
assert rows, "no complete pretrained/random pairs yet"
res = pd.concat(rows, ignore_index=True)
res.to_csv(f"{R}/outputs/random_control_summary.csv", index=False)
for tg in REPORT:
    s = res[res.target == tg]
    g = s.groupby(["m", "region"]).agg(k=("transfer", "size"), pre_gain=("pre_gain", "mean"),
                                        rand_gain=("rand_gain", "mean"), transfer=("transfer", "mean"),
                                        transfer_pos=("transfer", lambda v: int((v > 0).sum())))
    print(f"\n== {tg} ==\n{g.round(4).to_string()}")
    print(s.groupby("m")[["pre_gain", "rand_gain", "transfer"]].mean().round(4).to_string())
