"""parity_phase2.py -- output parity for Phase 2 script merges. For each case, the OLD scripts (taken from git tag
v5-as-run, commit b3ed0e0c28e3, so the test survives their removal) and the NEW merged script are run on the SAME real
input files and must write identical outputs. Where the paper uses a published copy in outputs/, it is compared too
(reported, not required: a rerun may differ in the last digits, see README).

Comparison: byte-identical, except PDFs, which are compared after removing their /CreationDate and /ModDate stamps.
stdout is compared where it carries no run-directory paths.

SAFE: every run gets its own scratch directory under $TMPDIR; inputs are symlinked into it (read-only use) at the same
relative paths, outputs are written only there. Nothing in the project tree is written, moved or deleted. Needs the real
inputs, so run it on Roar, e.g.
  sbatch --account=open --mem=16G --time=01:00:00 -o logs/parity_p2_%j.out \\
         --wrap "cd <root> && module load anaconda && module load r && venv/bin/python pipeline/tests/parity_phase2.py figures"
Cases: xd_boot, sudan, figures.  Exit 0 only if every case run is PARITY2_OK."""
import os, re, sys, glob, shutil, hashlib, subprocess, tempfile, difflib

ROOT = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
V5_COMMIT = "b3ed0e0c28e34fd49dc431d93b45ea1e4a3e7743"
PY = f"{ROOT}/venv/bin/python"
P, X, V = "scripts/phase4_scaling", "scripts/crossdomain", "scripts/replication_v5"
NE_ZIP = "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"
FO = "figures_tables/v5"

# A case is a list of runs. Run fields:
#   label; mode "envdir" (inputs linked flat into the run dir, whose path goes in env var `dirvar`; cwd = ROOT) or
#   "cwd" (inputs linked at their relative paths, cwd = run dir); inputs (globs relative to ROOT); mkdirs (relative,
#   created before the run); old: list of commands ({OLD} = tag copy of scripts, {D} = run dir); new: same with {NEW} =
#   project scripts; env: extra environment for both; outputs: relative files, or a directory to compare recursively;
#   published: True if outputs/<same path> in ROOT is the copy the paper uses; stdout: compare stdout.
CASES = {
    "xd_boot": [
        dict(label="M", mode="envdir", dirvar="XC_DIR", inputs=[f"outputs/crossdomain/preds_brazil_*.npz"], mkdirs=[],
             old=[[PY, f"{{OLD}}/{X}/xd_m_boot.py"]], new=[[PY, f"{{NEW}}/{X}/xd_boot.py"]], env_new={"XD_CAP": "M"},
             outputs=["xd_m_boot.csv"], published="outputs/crossdomain", stdout=True),
        dict(label="S", mode="envdir", dirvar="XC_DIR", inputs=[f"outputs/crossdomain/preds_brazil_*.npz"], mkdirs=[],
             old=[[PY, f"{{OLD}}/{X}/xd_s_boot.py"]], new=[[PY, f"{{NEW}}/{X}/xd_boot.py"]], env_new={"XD_CAP": "S"},
             outputs=["xd_s_boot.csv"], published="outputs/crossdomain", stdout=True),
    ],
    "sudan": [
        dict(label="sudan", mode="cwd", inputs=["data_intermediate/baseline_features.parquet", NE_ZIP], mkdirs=["outputs"],
             old=[[PY, f"{{OLD}}/{P}/audit_cells_export.py"], ["Rscript", f"{{OLD}}/{P}/audit_cell_country.R"],
                  [PY, f"{{OLD}}/{P}/sudan_cells_fix.py"]],
             new=[[PY, f"{{NEW}}/{P}/sudan_cells.py"]],
             outputs=["outputs/audit_cells_in.csv", "outputs/audit_cell_country.csv", "outputs/sudan_cells_keep.csv",
                      "outputs/sudan_exclude_cells.csv"], published=".", stdout=True),
    ],
    "figures": [
        dict(label="figures", mode="cwd",
             inputs=["outputs/bootstrap_ci_M_fullsrc_block5.csv", "outputs/bootstrap_ci_M_fullsrc_cell.csv",
                     "outputs/bootstrap_ci_M_fullsrc_block5_shares.csv", "outputs/bootstrap_ci_M_fullsrc_cell_shares.csv",
                     "outputs/bootstrap_ft_fullsrc_M_full_block5.csv", "outputs/crossdomain/xd_placebo_boot_v3M.csv",
                     "outputs/cell_alerts_persistence_M_fullsrc.csv.gz", "outputs/cell_alerts_incidence_M_fullsrc.csv.gz",
                     "outputs/eval_region_M_contrastive_e_fullsrc_loro_*.csv", NE_ZIP],
             # old: the nine Rscript calls of the v5/Phase 1 figures job, in order, each output folder created first
             # (the old job created all of them except fig3_loro_gains; see the fig3 diagnostic below)
             mkdirs=[FO] + [f"{FO}/{d}" for d in ("fig3_loro_gains", "fig4_cell_maps", "fig5_finetune_gains", "fig6_brazil",
                                                   "fig9_random_encoder", "fig10_share_curve", "fig16_brazil_random",
                                                   "fig17_share_regions", "fig18_persistence_gap")],
             old=[["Rscript", f"{{OLD}}/{P}/fig_loro_gains_v2.R", "outputs/bootstrap_ci_M_fullsrc_block5.csv", f"{FO}/fig3_loro_gains"],
                  ["Rscript", f"{{OLD}}/{V}/fig_cell_maps.R", "outputs/cell_alerts_persistence_M_fullsrc.csv.gz",
                   "outputs/cell_alerts_incidence_M_fullsrc.csv.gz", NE_ZIP, f"{FO}/fig4_cell_maps"],
                  ["Rscript", f"{{OLD}}/{P}/fig_finetune_gains.R", "outputs/bootstrap_ft_fullsrc_M_full_block5.csv", f"{FO}/fig5_finetune_gains"],
                  ["Rscript", f"{{OLD}}/{P}/fig_placebo_gradient_a_only_nolabel.R", "outputs/crossdomain/xd_placebo_boot_v3M.csv", f"{FO}/fig6_brazil"],
                  ["Rscript", f"{{OLD}}/{V}/fig_random_encoder.R", f"{FO}/fig9_random_encoder", "{D}"],
                  ["Rscript", f"{{OLD}}/{V}/fig_share_curve.R", f"{FO}/fig10_share_curve", "{D}"],
                  ["Rscript", f"{{OLD}}/{P}/fig_placebo_gradient.R", "outputs/crossdomain/xd_placebo_boot_v3M.csv", f"{FO}/fig16_brazil_random"],
                  ["Rscript", f"{{OLD}}/{V}/fig_share_regions.R", f"{FO}/fig17_share_regions", "{D}"],
                  ["Rscript", f"{{OLD}}/{V}/fig_persistence_gap.R", f"{FO}/fig18_persistence_gap", "{D}"]],
             new=[["Rscript", f"{{NEW}}/{V}/figures.R", FO, "{D}"]],
             new_mkdirs_only=[FO],          # figures.R creates its own sub-folders
             outputs=[FO], published=None, stdout=False),
    ],
}
ENV_R = {"R_PROFILE_USER": f"{ROOT}/{P}/Rprofile_cairo"}   # as the old wrapped CPU jobs set it


def sha(b): return hashlib.sha1(b).hexdigest()[:12]
def norm(path):
    b = open(path, "rb").read()
    if path.endswith(".pdf"):
        b = re.sub(rb"/(CreationDate|ModDate)\s*\(D:[^)]*\)", b"", b)
    return b


def old_copy(W, rel):
    dst = os.path.join(W, "old", rel)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(subprocess.run(["git", "-C", ROOT, "show", f"v5-as-run:{rel}"], capture_output=True, check=True).stdout)
    return dst


def link_inputs(d, globs, flat):
    n = 0
    for g in globs:
        hits = sorted(glob.glob(os.path.join(ROOT, g)))
        assert hits, f"no input matches {g}"
        for f in hits:
            rel = os.path.basename(f) if flat else os.path.relpath(f, ROOT)
            os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
            os.symlink(f, os.path.join(d, rel)); n += 1
    return n


def files_under(d, outs):
    got = {}
    for o in outs:
        p = os.path.join(d, o)
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for f in fs: got[os.path.relpath(os.path.join(r, f), d)] = os.path.join(r, f)
        elif os.path.isfile(p):
            got[o] = p
    return got


def run_side(W, run, side):
    d = os.path.join(W, f"{run['label']}_{side}"); os.makedirs(d)
    n = link_inputs(d, run["inputs"], flat=(run["mode"] == "envdir"))
    for m in (run.get("new_mkdirs_only") if side == "new" and run.get("new_mkdirs_only") is not None else run["mkdirs"]):
        os.makedirs(os.path.join(d, m), exist_ok=True)
    env = dict(os.environ); env.update(ENV_R); env.update(run.get("env_new", {}) if side == "new" else {})
    cwd = d if run["mode"] == "cwd" else ROOT
    if run["mode"] == "envdir": env[run["dirvar"]] = d
    out, rc = "", 0
    for cmd in run[side]:
        c = []
        for a in cmd:
            if "{OLD}" in a: a = old_copy(W, a.replace("{OLD}/", ""))
            a = a.replace("{NEW}", ROOT).replace("{D}", d)
            c.append(a)
        p = subprocess.run(c, cwd=cwd, env=env, capture_output=True, text=True)
        out += p.stdout
        print(f"  [{run['label']}/{side}] {os.path.basename(c[1])} rc={p.returncode}", flush=True)
        if p.returncode:
            print(p.stdout[-1500:], p.stderr[-3000:], flush=True); rc = p.returncode; break
    return d, rc, out, n


def fig3_diagnostic(W):
    """Does the old figures job's first call (fig_loro_gains_v2.R into a folder it never created) succeed?"""
    d = os.path.join(W, "fig3_diag"); os.makedirs(d)
    link_inputs(d, ["outputs/bootstrap_ci_M_fullsrc_block5.csv"], flat=False)
    os.makedirs(os.path.join(d, FO))
    env = dict(os.environ); env.update(ENV_R)
    p = subprocess.run(["Rscript", old_copy(W, f"{P}/fig_loro_gains_v2.R"), "outputs/bootstrap_ci_M_fullsrc_block5.csv",
                        f"{FO}/fig3_loro_gains"], cwd=d, env=env, capture_output=True, text=True)
    print(f"[fig3 diagnostic] old job's fig3 call without its folder: rc={p.returncode}"
          + ("" if p.returncode == 0 else f" -> {p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else ''}"), flush=True)


def main(cases):
    for c in cases: assert c in CASES, f"unknown case {c}; known: {sorted(CASES)}"
    c = subprocess.run(["git", "-C", ROOT, "rev-parse", "v5-as-run^{commit}"], capture_output=True, text=True).stdout.strip()
    assert c == V5_COMMIT, f"tag v5-as-run resolves to '{c}', expected {V5_COMMIT}"
    W = tempfile.mkdtemp(prefix="parity_p2_", dir=os.environ.get("TMPDIR", "/tmp"))
    print("scratch:", W, flush=True)
    ok_all = True
    try:
        for case in cases:
            print(f"== case {case}", flush=True)
            ok = True
            if case == "figures": fig3_diagnostic(W)
            for run in CASES[case]:
                do, ro, so, n = run_side(W, run, "old")
                dn, rn, sn, _ = run_side(W, run, "new")
                print(f"[{run['label']}] inputs linked: {n}", flush=True)
                if ro or rn: ok = False; continue
                if run["stdout"]:
                    if so == sn: print(f"[{run['label']}] stdout identical ({len(so.splitlines())} lines)")
                    else:
                        ok = False; print(f"[{run['label']}] STDOUT DIFFERS")
                        print("".join(list(difflib.unified_diff(so.splitlines(1), sn.splitlines(1), "old", "new"))[:40]))
                fo, fn = files_under(do, run["outputs"]), files_under(dn, run["outputs"])
                if set(fo) != set(fn) or not fo:
                    ok = False; print(f"[{run['label']}] OUTPUT SETS DIFFER: only old {sorted(set(fo)-set(fn))}, only new {sorted(set(fn)-set(fo))}")
                for rel in sorted(set(fo) & set(fn)):
                    a, b = norm(fo[rel]), norm(fn[rel])
                    same = a == b
                    msg = f"[{run['label']}] {rel}: old==new {same} ({sha(a)} / {sha(b)})"
                    if run["published"] is not None:
                        pp = os.path.join(ROOT, run["published"], rel)
                        msg += f"; old==published {os.path.isfile(pp) and norm(pp) == a}" if os.path.isfile(pp) else "; no published copy"
                    print(msg, flush=True)
                    ok = ok and same
            print(f"== case {case}: {'PARITY2_OK' if ok else 'PARITY2_FAILED'}", flush=True)
            ok_all = ok_all and ok
    finally:
        shutil.rmtree(W, ignore_errors=True)
    print("PARITY2_OK" if ok_all else "PARITY2_FAILED", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(CASES)))
