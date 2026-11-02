"""xd_placebo_boot_v3M.py (2026-10-26) -- v3 with the M Americas-held-out encoder (_capM outputs from launch_xd_placeboM.sbatch);
only CONFIGS, a homicide self-comparison and the output name changed. Writes outputs/crossdomain/xd_placebo_boot_v3M.csv.
xd_placebo_boot_v3.py (2026-10-25) -- v2 (6464b919) + suicide/drowning compared with BOTH full homicide and homicide
thinned to the placebo's death rate (sparsity control). "homicide" columns hold the reference domain (see column ref).
Writes outputs/crossdomain/xd_placebo_boot_v3.csv. v2 docstring follows.
xd_placebo_boot_v2.py (2026-10-24) -- placebo bootstrap with the Codex 2026-10-24 fix (v1 = xd_placebo_boot.py c7032962).
Fix: a comparison is computed ONLY when the exact manifest is present in BOTH domains: seeds {123,456,789} x draws
{0..4} (draw 0 at share 1.0) for both models, finite predictions, identical (grid_id, year_month) test rows across seeds
and domains. Homicide and placebo use the SAME (seed, draw) pair list and the SAME resampling indices (cells W and pairs
PI), so the difference CI is properly paired. Otherwise status = INCOMPLETE and no placebo verdict.
Placebo domains: traffic (braziltraffic_), suicide, drowning, dengue (brazil{cause}_) -- each skipped if absent.
Configs (C0 construction, same as the traffic placebo): finetune 1% rich (top2 vs scratch, top2 vs top2_rand) for
americas / all; nested C1 paper base (emb_base vs base, emb_base vs rand_emb_base) for americas / all, every share.
Homicide is compared with its rich mapping (traffic-style placebos have one token type, so placebo rich == min).
Effects: relative (dev_a - dev_b)/dev_a and absolute (per cell-month), mean over the 15 (or 3) pairs; two-way bootstrap
B=1000, seed 123. Pre-specified reading (unchanged from v1): violence-specific iff placebo CI includes 0 AND the
homicide-minus-placebo CI > 0. Writes outputs/crossdomain/xd_placebo_boot_v2.csv. Env XP_DIR."""
import os
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
D = os.environ.get("XP_DIR", f"{R}/outputs/crossdomain")
B, SEED = 1000, 123
SEEDS, DRAWS = [123, 456, 789], [0, 1, 2, 3, 4]
COMPARISONS = [("homicide_self", "brazil", "brazil"),  # v3M: gives homicide-only rows for configs with no placebo run (C4, geo)
               ("traffic", "brazil", "braziltraffic"), ("suicide", "brazil", "brazilsuicide"), ("drowning", "brazil", "brazildrowning"),
               ("suicide_vs_thinned", "brazilhomthinsuicide", "brazilsuicide"), ("drowning_vs_thinned", "brazilhomthindrowning", "brazildrowning"),
               ("thinned_suicide_vs_hom", "brazil", "brazilhomthinsuicide"), ("thinned_drowning_vs_hom", "brazil", "brazilhomthindrowning")]


def unit_dev(y, mu):
    mu = np.clip(mu, 1e-8, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (t - (y - mu))


FT, NE = [("scratch", "top2"), ("top2_rand", "top2")], [("base", "emb_base"), ("rand_emb_base", "emb_base")]
NSH = [0.0025, 0.01, 0.05, 0.1, 1.0]
CONFIGS = [("nested", "nested_americas_s{S}_rich_bpaper_capM", NSH, NE),          # 2026-10-26 v3M: M encoder (Americas held out)
           ("nested", "nested_americas_s{S}_rich_bpaper_c4_capM", NSH, NE),
           ("nested", "nested_americas_s{S}_rich_bpaper_c4_geo_capM", NSH, NE)]
q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))


def load(prefix, stem):
    fs = {s: f"{D}/preds_{prefix}_{stem.replace('{S}', str(s))}.npz" for s in SEEDS}
    if not all(os.path.exists(f) for f in fs.values()): return None
    return {s: np.load(f) for s, f in fs.items()}


rows = []
for pname, rpre, ppre in COMPARISONS:
    for arm, stem, shares, contrasts in CONFIGS:
        H, T = load(rpre, stem), load(ppre, stem)
        base = dict(placebo=pname, ref=rpre, plac=ppre, arm=arm, config=stem.replace("_s{S}", ""))
        if H is None or T is None:
            rows.append(dict(base, status="INCOMPLETE seeds")); print(f"[incomplete seeds] {pname} {stem}", flush=True); continue
        gid0, ym0 = H[SEEDS[0]]["__grid_id"], H[SEEDS[0]]["__year_month"]
        for Z in (H, T):
            for s in SEEDS:
                assert np.array_equal(Z[s]["__grid_id"], gid0) and np.array_equal(Z[s]["__year_month"], ym0), f"{pname} {stem}: rows not aligned"
            assert all(np.array_equal(Z[s]["__y"], Z[SEEDS[0]]["__y"]) for s in SEEDS), f"{pname} {stem}: y differs across seeds"
        yH, yT = H[SEEDS[0]]["__y"].astype(np.float64), T[SEEDS[0]]["__y"].astype(np.float64)
        cells, inv = np.unique(gid0, return_inverse=True); nC = len(cells); NC = np.bincount(inv, minlength=nC).astype(np.float64)
        rng = np.random.default_rng(SEED)
        W = np.stack([np.bincount(rng.integers(0, nC, nC), minlength=nC) for _ in range(B)]).astype(np.float64)
        for sh in shares:
            dr = [0] if sh >= 1.0 else DRAWS
            pairs = [(s, k) for s in SEEDS for k in dr]
            PI = np.random.default_rng([SEED, 7, len(pairs)]).integers(0, len(pairs), size=(B, len(pairs)))  # shared by both domains
            for a, b in contrasts:
                keys = {m: f"{m}|{sh:g}" for m in (a, b)}
                ok = all(f"{keys[m]}|{k}" in Z[s].files and np.isfinite(Z[s][f"{keys[m]}|{k}"]).all()
                         for Z in (H, T) for m in (a, b) for s, k in pairs)
                if not ok:
                    rows.append(dict(base, share=sh, baseline=a, model=b, status="INCOMPLETE pairs")); continue
                def cs(Z, y, m, s, k): return np.bincount(inv, weights=unit_dev(y, Z[s][f"{keys[m]}|{k}"].astype(np.float64)), minlength=nC)
                SH = {(m, s, k): cs(H, yH, m, s, k) for m in (a, b) for s, k in pairs}
                ST = {(m, s, k): cs(T, yT, m, s, k) for m in (a, b) for s, k in pairs}
                for kind in ("relative", "absolute"):
                    def point(S):
                        if kind == "absolute": return float(np.mean([(S[(a, s, k)].sum() - S[(b, s, k)].sum()) / NC.sum() for s, k in pairs]))
                        return float(np.mean([(S[(a, s, k)].sum() - S[(b, s, k)].sum()) / S[(a, s, k)].sum() for s, k in pairs]))
                    def boot(S):
                        if kind == "absolute": per = np.stack([(W @ S[(a, s, k)] - W @ S[(b, s, k)]) / (W @ NC) for s, k in pairs], 1)
                        else: per = np.stack([(W @ S[(a, s, k)] - W @ S[(b, s, k)]) / (W @ S[(a, s, k)]) for s, k in pairs], 1)
                        return np.take_along_axis(per, PI, axis=1).mean(1)
                    eh, et = point(SH), point(ST); bh, bt = boot(SH), boot(ST); dd = bh - bt
                    rows.append(dict(base, share=sh, baseline=a, model=b, effect_kind=kind, status="complete", n_pairs=len(pairs),
                                     effect_homicide=eh, ci_hom_lo=q(bh)[0], ci_hom_hi=q(bh)[1],
                                     effect_placebo=et, ci_plac_lo=q(bt)[0], ci_plac_hi=q(bt)[1],
                                     diff=eh - et, ci_diff_lo=q(dd)[0], ci_diff_hi=q(dd)[1],
                                     placebo_pass=bool(q(bt)[0] <= 0 <= q(bt)[1] and q(dd)[0] > 0)))
out = pd.DataFrame(rows); out.to_csv(f"{D}/xd_placebo_boot_v3M.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out[out.get("effect_kind", pd.Series(dtype=str)).eq("relative") | out.status.ne("complete")].round(4).to_string(index=False))
print("XP3M_DONE")
