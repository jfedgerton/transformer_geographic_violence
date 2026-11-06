"""parity_phase2.py -- output parity for Phase 2 script merges. For each case, the OLD scripts (taken from git tag
v5-as-run, commit b3ed0e0c28e3, so the test survives their removal) and the NEW merged script are run on the SAME real
input files and must write identical outputs. Where the paper uses a published copy in outputs/, it is compared too
(reported, not required: a rerun may differ in the last digits, see README).

Comparison: byte-identical, except PDFs, which are compared after removing their /CreationDate and /ModDate stamps.
Inventory (Codex Phase 2 review, P2): every file a run writes into its run directory (regular files, linked inputs
excluded) must be exactly the declared set (outputs, the pinned `expected` list for a directory output, and any pinned
`byproducts`), on both sides. A declared file missing on either side, or any undeclared file, fails the case. A built-in
self-test (also run on its own with --selftest) checks that these failures are caught before any real run.
stdout is compared where it carries no run-directory paths.

SAFE: every run gets its own scratch directory under $TMPDIR; inputs are symlinked into it (read-only use) at the same
relative paths, outputs are written only there. Nothing in the project tree is written, moved or deleted. Needs the real
inputs, so run it on Roar, e.g.
  sbatch --account=open --mem=16G --time=01:00:00 -o logs/parity_p2_%j.out \\
         --wrap "cd <root> && module load anaconda && module load r && venv/bin/python pipeline/tests/parity_phase2.py figures"
Cases: xd_boot, sudan, figures, figures_one, sim_prep, panel; panel_build runs separately (see panel_build_side); a single run is selected with case:label (e.g. sim_prep:traffic).  Exit 0 only if every case run is PARITY2_OK."""
import os, re, sys, glob, time, shutil, hashlib, subprocess, tempfile, difflib

ROOT = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
V5_COMMIT = "b3ed0e0c28e34fd49dc431d93b45ea1e4a3e7743"
PY = f"{ROOT}/venv/bin/python"
PYXD = f"{ROOT}/venv_xd/bin/python"   # adds pyreaddbc for the SIM .dbc files
P, X, V = "scripts/phase4_scaling", "scripts/crossdomain", "scripts/replication_v5"
NE_ZIP = "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"
FO = "figures_tables/v5"

# A case is a list of runs. Run fields:
#   label; mode "envdir" (inputs linked flat into the run dir, whose path goes in env var `dirvar`; cwd = ROOT) or
#   "cwd" (inputs linked at their relative paths, cwd = run dir); "rootrw" (inputs linked at their relative paths, and
#   both the old and the new script are run from copies whose hard-coded `R = "<project root>"` line points at the run
#   dir, the only change, so outputs land there); "outdir" (env XD_OUTDIR = run dir, inputs read from the project); inputs (globs relative to ROOT); mkdirs (relative,
#   created before the run); old: list of commands ({OLD} = tag copy of scripts, {D} = run dir); new: same with {NEW} =
#   project scripts; env: extra environment for both; outputs: relative files, or a directory to compare recursively;
#   published: True if outputs/<same path> in ROOT is the copy the paper uses; stdout: compare stdout.
FIG_FILES = ["fig3_loro_gains/fig_loro_gains.pdf", "fig3_loro_gains/fig_loro_gains.png", "fig3_loro_gains/tab_loro_gains_ci.tex",
             "fig4_cell_maps/fig_cell_alerts_maps_pers_nested_M_fullsrc.pdf", "fig4_cell_maps/fig_cell_alerts_maps_pers_nested_M_fullsrc.png",
             "fig5_finetune_gains/fig_finetune_gains.pdf", "fig5_finetune_gains/fig_finetune_gains.png",
             "fig6_brazil/fig_placebo_gradient_ab_M.pdf", "fig6_brazil/fig_placebo_gradient_ab_M.png",
             "fig9_random_encoder/fig_random_encoder.pdf", "fig9_random_encoder/fig_random_encoder.png",
             "fig10_share_curve/fig_boot_shares.pdf", "fig10_share_curve/fig_boot_shares.png",
             "fig16_brazil_random/fig_placebo_gradient_M.pdf", "fig16_brazil_random/fig_placebo_gradient_M.png",
             "fig17_share_regions/fig_boot_share_regions.pdf", "fig17_share_regions/fig_boot_share_regions.png",
             "fig18_persistence_gap/fig_boot_persistence.pdf", "fig18_persistence_gap/fig_boot_persistence.png"]
assert len(FIG_FILES) == 19
FIG19_ARGS = ["outputs/cell_alerts_incidence_M_fullsrc.csv.gz", NE_ZIP, "incidence",
              "outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc.png"]   # as tasks.sbatch alert_chain passes them (SIZE=M)
FIG19_VARIANTS = ["print_countries_all_count", "print_countries_all_rate", "print_opt1", "print_opt4_countries", "print_opt4_maps",
                  "print_opt4b_countries_wide"]
UKR_SETS = ["1_invasion", "2_counteroffensives", "3_southern_offensive", "4_2024_2025"]
def ukr_gain_args(arm):
    return [f"outputs/ukraine_maps_M_f0025/ukraine_cells_f0.0025_M_fullsrc_{arm}.csv",
            f"outputs/ukraine_maps_M_f01/ukraine_cells_f0.1_M_fullsrc_{arm}.csv", f"outputs/ukraine_alert_gain_M_{arm}"]
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
             outputs=[FO], expected=[f"{FO}/{x}" for x in FIG_FILES],
             # R's default-device file in the working directory (the project root in production). The old nine scripts write
             # it too. It is a blank page: ggplotGrob()/ggplot_build() open the default device and record only empty q/Q
             # pairs. One R session accumulates more pairs than nine separate ones, so the bytes differ; it is required on
             # both sides and checked to be blank instead of byte-compared (Roar diagnostic job 56992072).
             blank_byproducts=["Rplots.pdf"],
             published=None, stdout=False),
    ],
    "sim_prep": [
        dict(label=c, mode="rootrw", inputs=["data_raw/sim_brazil/dores/DOBR*.dbc"], mkdirs=["data_intermediate"],
             old=[[PYXD, f"{{OLD}}/{X}/prep_sim_{c}.py"]], new=[[PYXD, f"{{NEW}}/{X}/prep_sim.py"]], env_new={"XD_CAUSE": c},
             outputs=[f"data_intermediate/sim_{c}_events.parquet", "outputs/crossdomain/" + aud], published=".", stdout=True)
        for c, aud in (("homicide", "sim_year_audit.csv"), ("traffic", "sim_traffic_year_audit.csv"))
    ] + [
        dict(label="placebos", mode="rootrw", inputs=["data_raw/sim_brazil/dores/DOBR*.dbc"], mkdirs=["data_intermediate", "outputs/crossdomain"],
             old=[[PYXD, f"{{OLD}}/{X}/prep_sim_placebos_v2.py"]], new=[[PYXD, f"{{NEW}}/{X}/prep_sim.py"]], env_new={"XD_CAUSE": "placebos"},
             outputs=[f"data_intermediate/sim_{c}_events.parquet" for c in ("suicide", "drowning", "dengue")]
                     + ["outputs/crossdomain/sim_placebos_year_audit_v2.csv"], published=".", stdout=True)
    ],
    "panel": [
        dict(label=c, mode="outdir", inputs=[], mkdirs=[],
             old=[[PY, f"{{OLD}}/{X}/{old}"]], new=[[PY, f"{{NEW}}/{X}/brazil_panel.py"]], env_new={"XD_CAUSE": c},
             outputs=[f"data_intermediate/{stem}_panel.parquet", f"outputs/crossdomain/{stem}_audit.json"], published=".", stdout=True)
        for c, old, stem in (("homicide", "build_brazil_panel.py", "xd_brazil"), ("traffic", "build_brazil_traffic_panel.py", "xd_brazil_traffic"))
    ] + [
        dict(label=c, mode="outdir", inputs=[], mkdirs=[],
             old=[[PY, f"{{OLD}}/{X}/build_brazil_placebo_panel.py"]], env_old={"XD_PLACEBO": c},
             new=[[PY, f"{{NEW}}/{X}/brazil_panel.py"]], env_new={"XD_CAUSE": c},
             outputs=[f"data_intermediate/xd_brazil_{c}_panel.parquet", f"outputs/crossdomain/xd_brazil_{c}_audit.json"], published=".", stdout=True)
        for c in ("suicide", "drowning", "dengue")
    ],
    # 2026-11-05: figures.R --one <id> (single-figure mode) vs the three old single-figure R scripts of stages 06-07
    "figures_one": [
        dict(label="fig19", mode="cwd", inputs=["outputs/cell_alerts_incidence_M_fullsrc.csv.gz", NE_ZIP],
             mkdirs=["outputs/cell_alerts_fig_M_fullsrc"],
             old=[["Rscript", "{OLD}/figures_tables/fig_cell_alerts.R", *FIG19_ARGS]],
             new=[["Rscript", f"{{NEW}}/{V}/figures.R", "--one", "fig19", *FIG19_ARGS]],
             outputs=["outputs/cell_alerts_fig_M_fullsrc"],
             expected=[f"outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_{x}.png" for x in FIG19_VARIANTS],
             published=".", stdout=False),
    ] + [
        dict(label=f"ukr_map_{fdir}_{arm}", mode="cwd", inputs=[f"outputs/ukraine_maps_M_{fdir}/ukraine_cells_f{share}_M_fullsrc_{arm}.csv"],
             mkdirs=[],
             old=[["Rscript", f"{{OLD}}/{P}/fig_ukraine_map.R", f"outputs/ukraine_maps_M_{fdir}/ukraine_cells_f{share}_M_fullsrc_{arm}.csv"]],
             new=[["Rscript", f"{{NEW}}/{V}/figures.R", "--one", "ukr_map", f"outputs/ukraine_maps_M_{fdir}/ukraine_cells_f{share}_M_fullsrc_{arm}.csv"]],
             outputs=[f"outputs/ukraine_maps_M_{fdir}"],
             expected=[f"outputs/ukraine_maps_M_{fdir}/fig_ukraine_f{share}_M_fullsrc_{arm}_{st}.{ext}" for st in UKR_SETS for ext in ("pdf", "png")],
             published=".", stdout=False)
        for fdir, share, arm in (("f0025", "0.0025", "top2"), ("f01", "0.1", "full"))
    ] + [
        dict(label=f"ukr_gain_{arm}", mode="cwd", env_both={"UKR_ARM": arm},
             inputs=[f"outputs/ukraine_maps_M_f0025/ukraine_cells_f0.0025_M_fullsrc_{arm}.csv", f"outputs/ukraine_maps_M_f01/ukraine_cells_f0.1_M_fullsrc_{arm}.csv"],
             mkdirs=[f"outputs/ukraine_alert_gain_M_{arm}"],
             old=[["Rscript", f"{{OLD}}/{V}/fig_ukraine_alert_gain_arm.R", *ukr_gain_args(arm)]],
             new=[["Rscript", f"{{NEW}}/{V}/figures.R", "--one", "ukr_gain", *ukr_gain_args(arm)]],
             outputs=[f"outputs/ukraine_alert_gain_M_{arm}/fig_ukraine_alert_gain_M_fullsrc_{arm}.pdf",
                      f"outputs/ukraine_alert_gain_M_{arm}/fig_ukraine_alert_gain_M_fullsrc_{arm}.png",
                      f"outputs/ukraine_alert_gain_M_{arm}/ukraine_alert_gain_cells_M_fullsrc_{arm}.csv"],
             published=".", stdout=False)
        for arm in ("top2", "full")
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


def required_files(run):
    """the exact set of files a run must write: declared file outputs, the pinned list for a directory output, byproducts"""
    req = set(run.get("byproducts", [])) | set(run.get("blank_byproducts", []))
    for o in run["outputs"]:
        if "expected" in run and any(e.startswith(o.rstrip("/") + "/") for e in run["expected"]):
            req |= {e for e in run["expected"] if e.startswith(o.rstrip("/") + "/")}
        else:
            req.add(o)
    return req


def inventory(d):
    """every regular file written under the run dir; linked inputs (symlinks) are not outputs"""
    got = set()
    for r, dirs, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            if not os.path.islink(p) and os.path.isfile(p): got.add(os.path.relpath(p, d))
    return got


def blank_pdf(path):
    """True if every page content stream of an R pdf() file holds only line settings and empty save/restore pairs"""
    import zlib
    b = open(path, "rb").read()
    if not b.startswith(b"%PDF"): return False
    pages = 0
    for m in re.finditer(rb"<<([^<>]*?)>>\s*stream\r?\n(.*?)endstream", b, re.S):
        if b"/Length" not in m.group(1) or b"/FlateDecode" not in m.group(1): continue
        try: t = zlib.decompress(m.group(2))
        except zlib.error: continue
        if t.startswith(b"\x00\x00"): continue            # the embedded sRGB ICC profile, not page content
        pages += 1
        if re.sub(rb"[\sqQJj01]", b"", t): return False     # anything beyond '1 J 1 j' and q/Q is a drawing
    return pages >= 1


def compare_outputs(do, dn, run, root_for_published=None):
    """inventory gate, then byte comparison of every required file. Returns (ok, report lines)."""
    lab, req, lines, ok = run["label"], required_files(run), [], True
    assert req, f"[{lab}] no declared outputs"
    for side, d in (("old", do), ("new", dn)):
        inv = inventory(d)
        missing, extra = sorted(req - inv), sorted(inv - req)
        if missing: ok = False; lines.append(f"[{lab}] {side}: DECLARED OUTPUT MISSING {missing}")
        if extra: ok = False; lines.append(f"[{lab}] {side}: UNDECLARED FILE WRITTEN {extra}")
    lines.append(f"[{lab}] inventory: {len(req)} declared files")
    for rel in sorted(run.get("blank_byproducts", [])):
        bo = os.path.isfile(os.path.join(do, rel)) and blank_pdf(os.path.join(do, rel))
        bn = os.path.isfile(os.path.join(dn, rel)) and blank_pdf(os.path.join(dn, rel))
        lines.append(f"[{lab}] {rel}: blank page old {bo}, new {bn} (byproduct; not byte-compared)")
        ok = ok and bo and bn
    for rel in sorted(req - set(run.get("blank_byproducts", []))):
        fo, fn = os.path.join(do, rel), os.path.join(dn, rel)
        if not (os.path.isfile(fo) and os.path.isfile(fn)): continue
        a, b = norm(fo), norm(fn)
        same = a == b
        msg = f"[{lab}] {rel}: old==new {same} ({sha(a)} / {sha(b)})"
        if root_for_published is not None and run["published"] is not None:
            pp = os.path.join(root_for_published, run["published"], rel)
            msg += f"; old==published {norm(pp) == a}" if os.path.isfile(pp) else "; no published copy"
        lines.append(msg)
        ok = ok and same
    return ok, lines


def selftest():
    """negative controls (Codex P2): the gate must fail on a missing declared output and on an undeclared file"""
    run = dict(label="selftest", outputs=["outputs/a.csv", "outputs/b.csv"], published=None)
    figrun = dict(label="selftest_dir", outputs=["figs"], expected=["figs/x/p.pdf", "figs/x/p.png"], published=None)
    pdf0 = b"%PDF-1.4\n/CreationDate (D:20260101000000)\n/ModDate (D:20260101000000)\nBODY\n"
    pdf1 = pdf0.replace(b"20260101", b"20261231")
    def mk(base, files):
        os.makedirs(base)
        for rel, data in files.items():
            os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True); open(os.path.join(base, rel), "wb").write(data)
    full = {"outputs/a.csv": b"1\n", "outputs/b.csv": b"2\n"}
    brun = dict(run, label="selftest_blank", blank_byproducts=["Rplots.pdf"])
    def rp(content):
        import zlib
        z = zlib.compress(content)
        return (b"%PDF-1.4\n/CreationDate (D:20260101000000)\n<<\n/Length " + str(len(z)).encode() + b" /Filter /FlateDecode\n>>\nstream\n"
                + z + b"\nendstream\n")
    cases = [  # (name, run, old files, new files, expected ok)
        ("identical full sets pass", run, full, full, True),
        ("declared output missing on both sides fails", run, {"outputs/a.csv": b"1\n"}, {"outputs/a.csv": b"1\n"}, False),
        ("undeclared extra file on new side fails", run, full, dict(full, **{"outputs/UNEXPECTED.csv": b"x"}), False),
        ("content difference fails", run, full, dict(full, **{"outputs/b.csv": b"3\n"}), False),
        ("pdf differing only in dates passes", figrun, {"figs/x/p.pdf": pdf0, "figs/x/p.png": b"P"}, {"figs/x/p.pdf": pdf1, "figs/x/p.png": b"P"}, True),
        ("pdf body change fails", figrun, {"figs/x/p.pdf": pdf0, "figs/x/p.png": b"P"}, {"figs/x/p.pdf": pdf1.replace(b"BODY", b"BODZ"), "figs/x/p.png": b"P"}, False),
        ("file missing from a pinned directory inventory fails", figrun, {"figs/x/p.pdf": pdf0}, {"figs/x/p.pdf": pdf0}, False),
        ("blank byproducts differing in q/Q count pass", brun, dict(full, **{"Rplots.pdf": rp(b"1 J 1 j q\nQ\n")}),
         dict(full, **{"Rplots.pdf": rp(b"1 J 1 j q\nQ q\nQ q\nQ\n")}), True),
        ("byproduct with a drawing fails", brun, dict(full, **{"Rplots.pdf": rp(b"1 J 1 j q\nQ\n")}),
         dict(full, **{"Rplots.pdf": rp(b"1 J 1 j q\n0 0 m 10 10 l S\nQ\n")}), False),
        ("missing blank byproduct fails", brun, full, dict(full, **{"Rplots.pdf": rp(b"q\nQ\n")}), False),
    ]
    T = tempfile.mkdtemp(prefix="parity_p2_selftest_", dir=os.environ.get("TMPDIR", "/tmp"))
    good = True
    try:
        for i, (name, r, fo, fn, want) in enumerate(cases):
            do, dn = os.path.join(T, f"{i}_old"), os.path.join(T, f"{i}_new"); mk(do, fo); mk(dn, fn)
            got, _ = compare_outputs(do, dn, r)
            print(f"[selftest] {name}: {'ok' if got == want else 'WRONG'} (gate says {'pass' if got else 'fail'})", flush=True)
            good = good and got == want
    finally:
        shutil.rmtree(T, ignore_errors=True)
    return good


def run_side(W, run, side):
    d = os.path.join(W, f"{run['label']}_{side}"); os.makedirs(d)
    n = link_inputs(d, run["inputs"], flat=(run["mode"] == "envdir")) if run["inputs"] else 0
    for m in (run.get("new_mkdirs_only") if side == "new" and run.get("new_mkdirs_only") is not None else run["mkdirs"]):
        os.makedirs(os.path.join(d, m), exist_ok=True)
    env = dict(os.environ); env.update(ENV_R); env.update(run.get("env_both", {}))
    env.update(run.get("env_new", {}) if side == "new" else run.get("env_old", {}))
    cwd = d if run["mode"] in ("cwd", "rootrw") else ROOT
    if run["mode"] == "envdir": env[run["dirvar"]] = d
    if run["mode"] == "outdir": env["XD_OUTDIR"] = d
    out, rc = "", 0
    for cmd in run[side]:
        c = []
        for a in cmd:
            if "{OLD}" in a: a = old_copy(W, a.replace("{OLD}/", ""))
            a = a.replace("{NEW}", ROOT).replace("{D}", d)
            if run["mode"] == "rootrw" and a.endswith(".py"):
                t = open(a).read(); line = f'R = "{ROOT}"\n'
                assert t.count(line) >= 1, f"{a}: hard-coded root line not found"
                a2 = os.path.join(W, f"rw_{run['label']}_{side}_" + os.path.basename(a))
                open(a2, "w").write(t.replace(line, f'R = "{d}"\n')); a = a2
            c.append(a)
        p = subprocess.run(c, cwd=cwd, env=env, capture_output=True, text=True)
        out += p.stdout.replace(d, "<RUNDIR>")
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


# ======================= panel_build: stage 01 (old eleven scripts vs scripts/build_panel.py) =======================
# Too long for one job, so each side runs as its own job into a persistent work dir, then a third call compares:
#   parity_phase2.py panel_build --side old --workdir W    (and --side new, in parallel)
#   parity_phase2.py panel_build --compare W
# Each side gets its own project root W/<side> holding: scripts/ (the old scripts from tag v5-as-run, or
# scripts/build_panel.py from this tree), links to the raw inputs, and nothing else. The stage-01 code derives every path
# from its own location (00_master.py PROJECT_ROOT), except build_pgm_fatalities_target.py, whose hard-coded root line is
# rewritten to W/<side> in both the old and the new copy; afterwards no copy may still name the project root. The steps
# run in the pipeline order with the pipeline environment (rebuild: PHASE_MODE=global; featbuild: super, then baseline
# with REPR_SEQ_SUFFIX=_super; prio; pgm), one fresh process each, as pipeline/tasks.sbatch and run.sh do.
PB_OLD = [("acled", "01_ingest_acled.py"), ("ucdp", "01b_ingest_ucdp.py"), ("merge", "01c_merge_sources.py"),
          ("grid", "02_build_grid.py"), ("aggregate", "03_aggregate_location_month.py"), ("sequences", "04_build_sequences.py"),
          ("baseline", "05_build_baseline_features.py"), ("super", "build_sequences_super.py"),
          ("prio", "build_priogrid_static.py"), ("pgm", "build_pgm_fatalities_target.py")]
PB_RUNS = [("acled", {"PHASE_MODE": "global"}), ("ucdp", {"PHASE_MODE": "global"}), ("merge", {"PHASE_MODE": "global"}),
           ("grid", {"PHASE_MODE": "global"}), ("aggregate", {"PHASE_MODE": "global"}), ("sequences", {"PHASE_MODE": "global"}),
           ("baseline", {"PHASE_MODE": "global"}), ("super", {}), ("baseline", {"REPR_SEQ_SUFFIX": "_super"}),
           ("prio", {}), ("pgm", {})]
PB_RAW = ["data_raw/acled_full.csv", "data_raw/ucdp/GEDEvent_v25_1.csv", "data_raw/priogrid/static/*.json"]
PB_SKIP = ("scripts/", "stdout_")

# Required stage-01 artifacts (GPT review, 2026-11-06): both sides must write every one of these; any other file is still
# compared, and a file on one side only still fails. Paths are relative to the side root.
PB_REQUIRED = sorted(["data_intermediate/acled_cleaned.parquet", "data_intermediate/ucdp_cleaned.parquet",
                      "data_intermediate/acled_only_backup.parquet", "data_intermediate/events_unified.parquet",
                      "data_intermediate/acled_with_grid.parquet", "data_intermediate/grid_month_panel.parquet",
                      "data_intermediate/vocab.json", "data_intermediate/splits.json", "data_intermediate/sequences.parquet",
                      "data_intermediate/baseline_feature_cols.json", "data_intermediate/baseline_features.parquet",
                      "data_intermediate/vocab_super.json", "data_intermediate/sequences_super.parquet",
                      "data_intermediate/baseline_feature_cols_super.json", "data_intermediate/baseline_features_super.parquet",
                      "data_intermediate/priogrid_static.parquet", "data_intermediate/pgm_fatalities_target.parquet",
                      "outputs/grid_activity_distribution.png", "outputs/panel_diagnostics.png"])
PB_STDOUT = [f"stdout_{i:02d}_{step}.txt" for i, (step, _) in enumerate(PB_RUNS)]   # the eleven captures, exactly
# Timing fields printed by the sequence builders (04_build_sequences.py, build_sequences_super.py). Only these are
# masked before stdout is compared; counts, percentages and every other line must match exactly.
PB_TIMING = [(re.compile(r"rate=\d+/s eta=-?[\d.]+ min"), "rate=<T>/s eta=<T> min"),
             (re.compile(r"Sequence build complete in [\d.]+s"), "Sequence build complete in <T>s"),
             (re.compile(r"(\(\d+%\)) [\d.]+min"), r"\1 <T>min")]


def pb_norm_stdout(t):
    for rx, rep in PB_TIMING: t = rx.sub(rep, t)
    return t
      # not outputs: the code under test and the captured stdout


def panel_build_side(side, W):
    assert side in ("old", "new")
    c = subprocess.run(["git", "-C", ROOT, "rev-parse", "v5-as-run^{commit}"], capture_output=True, text=True).stdout.strip()
    assert c == V5_COMMIT, f"tag v5-as-run resolves to '{c}', expected {V5_COMMIT}"
    d = os.path.join(os.path.abspath(W), side)
    assert not os.path.exists(d), f"{d} exists; use a fresh work dir"
    os.makedirs(os.path.join(d, "scripts"))
    if side == "old":
        files = ["00_master.py"] + [fn for _, fn in PB_OLD]
        for fn in files:
            open(os.path.join(d, "scripts", fn), "wb").write(subprocess.run(
                ["git", "-C", ROOT, "show", f"v5-as-run:scripts/{fn}"], capture_output=True, check=True).stdout)
    else:
        files = ["build_panel.py"]; shutil.copy(os.path.join(ROOT, "scripts", "build_panel.py"), os.path.join(d, "scripts"))
    hard = f'R="{ROOT}"\n'
    tgt = "build_pgm_fatalities_target.py" if side == "old" else "build_panel.py"
    t = open(os.path.join(d, "scripts", tgt)).read()
    assert t.count(hard) == 1, f"{tgt}: hard-coded root line not found exactly once"
    open(os.path.join(d, "scripts", tgt), "w").write(t.replace(hard, f'R="{d}"\n'))
    for fn in files:
        assert ROOT not in open(os.path.join(d, "scripts", fn)).read(), f"{fn} still names the project root"
    n = link_inputs(d, PB_RAW, flat=False)
    print(f"[panel_build/{side}] root {d}; {n} raw inputs linked", flush=True)
    env0 = dict(os.environ); env0.pop("PHASE_MODE", None); env0.pop("REPR_SEQ_SUFFIX", None)
    for i, (step, extra) in enumerate(PB_RUNS):
        env = dict(env0); env.update(extra)
        script = dict(PB_OLD)[step] if side == "old" else "build_panel.py"
        cmd = [PY, f"scripts/{script}"] + ([] if side == "old" else [step])
        t0 = time.time()
        p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
        open(os.path.join(d, f"stdout_{i:02d}_{step}.txt"), "w").write(p.stdout.replace(d, "<RUNDIR>"))
        print(f"[panel_build/{side}] {i:02d} {step} {extra or ''} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-4000:], flush=True)
            print("PANEL_BUILD_SIDE_FAILED", flush=True); return 1
    open(os.path.join(d, "stdout_DONE"), "w").write("done\n")
    print("PANEL_BUILD_SIDE_DONE", flush=True)
    return 0


def pb_compare_sides(do, dn, published=True):
    """compare two finished side roots; returns (ok, lines). Used by panel_build_compare and by its self-test."""
    lines, ok = [], True
    io = {f for f in inventory(do) if not f.startswith(PB_SKIP)}
    inn = {f for f in inventory(dn) if not f.startswith(PB_SKIP)}
    for side, inv in (("old", io), ("new", inn)):
        miss = [f for f in PB_REQUIRED if f not in inv]
        if miss: ok = False; lines.append(f"[panel_build] REQUIRED OUTPUTS MISSING on {side}: {miss}")
    if io != inn:
        ok = False; lines.append(f"[panel_build] OUTPUT SETS DIFFER: only old {sorted(io - inn)}, only new {sorted(inn - io)}")
    lines.append(f"[panel_build] inventory: {len(io)} files written by the old chain ({len(PB_REQUIRED)} required)")
    content_only = []
    for rel in sorted(io & inn):
        a, b = norm(os.path.join(do, rel)), norm(os.path.join(dn, rel))
        same = a == b
        msg = f"[panel_build] {rel}: old==new {same} ({sha(a)} / {sha(b)})"
        if not same and rel.endswith(".parquet"):
            try:
                import pandas as pd
                fa, fb = pd.read_parquet(os.path.join(do, rel)), pd.read_parquet(os.path.join(dn, rel))
                ce = fa.shape == fb.shape and list(fa.columns) == list(fb.columns) and fa.equals(fb)
            except Exception as e:
                ce = False; msg += f"; content check unreadable ({type(e).__name__})"
            msg += f"; content equal {ce} (diagnostic only, not a pass)"
            if ce: content_only.append(rel)
        if published:
            pp = os.path.join(ROOT, rel)
            msg += (f"; old==published {norm(pp) == a}" if os.path.isfile(pp) else "; no published copy")
        lines.append(msg)
        ok = ok and same
    for side, d in (("old", do), ("new", dn)):
        got = sorted(f for f in os.listdir(d) if f.startswith("stdout_") and f != "stdout_DONE")
        if got != sorted(PB_STDOUT): ok = False; lines.append(f"[panel_build] STDOUT CAPTURES WRONG on {side}: {got}")
    for f in PB_STDOUT:
        fo, fn = os.path.join(do, f), os.path.join(dn, f)
        if not (os.path.isfile(fo) and os.path.isfile(fn)): continue
        so, sn = pb_norm_stdout(open(fo).read()), pb_norm_stdout(open(fn).read())
        lines.append(f"[panel_build] {f}: stdout identical apart from timing fields {so == sn}")
        ok = ok and so == sn
    if content_only: lines.append(f"[panel_build] NOTE bytes differ but content is equal for {content_only}; still a failure")
    return ok, lines


def panel_build_selftest():
    """negative and positive controls for pb_compare_sides (GPT review 2026-11-06, P2 findings 1 and 2)"""
    T = tempfile.mkdtemp(prefix="pb_selftest_"); good = True
    def side(base, files, logs):
        os.makedirs(base)
        for rel, data in files.items():
            os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True); open(os.path.join(base, rel), "wb").write(data)
        for f, txt in logs.items(): open(os.path.join(base, f), "w").write(txt)
        open(os.path.join(base, "stdout_DONE"), "w").write("done\n")
    full = {f: b"x" for f in PB_REQUIRED}
    logs = {f: "ok\n" for f in PB_STDOUT}
    t1 = dict(logs); t1[PB_STDOUT[5]] = " 100,000 / 9,000,000 (1.1%) rate=4100/s eta=36.2 min\n\nSequence build complete in 12.3s\nProcessed 9,000,000 grid-months\n"
    t2 = dict(t1); t2[PB_STDOUT[5]] = t1[PB_STDOUT[5]].replace("4100/s eta=36.2", "3900/s eta=38.0").replace("12.3s", "13.4s")
    t3 = dict(t1); t3[PB_STDOUT[5]] = t1[PB_STDOUT[5]].replace("9,000,000 grid-months", "8,999,999 grid-months")
    s1 = dict(logs); s1[PB_STDOUT[7]] = " 500,000/9,000,000 (6%) 1.2min\n"
    s2 = dict(logs); s2[PB_STDOUT[7]] = " 500,000/9,000,000 (6%) 1.9min\n"
    miss = {f: b"x" for f in PB_REQUIRED[1:]}
    extra = dict(full); extra["data_intermediate/extra.parquet"] = b"x"
    diff = dict(full); diff[PB_REQUIRED[0]] = b"y"
    short = dict(logs); short.pop(PB_STDOUT[-1])
    cases = [("identical outputs and logs pass", full, logs, full, logs, True),
             ("empty inventory fails", {}, logs, {}, logs, False),
             ("required file missing on both sides fails", miss, logs, miss, logs, False),
             ("extra output on one side fails", full, logs, extra, logs, False),
             ("content difference fails", full, logs, diff, logs, False),
             ("timing-only stdout difference passes (sequences)", full, t1, full, t2, True),
             ("timing-only stdout difference passes (super)", full, s1, full, s2, True),
             ("count difference in stdout fails", full, t1, full, t3, False),
             ("missing stdout capture on both sides fails", full, short, full, short, False)]
    try:
        for k, (name, fo, lo, fn, ln, want) in enumerate(cases):
            do, dn = os.path.join(T, f"{k}_old"), os.path.join(T, f"{k}_new"); side(do, fo, lo); side(dn, fn, ln)
            got, _ = pb_compare_sides(do, dn, published=False)
            print(f"[pb-selftest] {name}: {'ok' if got == want else 'WRONG'} (says {'pass' if got else 'fail'})", flush=True)
            good = good and got == want
    finally:
        shutil.rmtree(T, ignore_errors=True)
    return good


def panel_build_compare(W):
    assert panel_build_selftest(), "panel_build self-test failed: the comparison does not catch what it must"
    do, dn = os.path.join(W, "old"), os.path.join(W, "new")
    for x in (do, dn): assert os.path.isfile(os.path.join(x, "stdout_DONE")), f"{x} did not finish"
    ok, lines = pb_compare_sides(do, dn)
    for l in lines: print(l, flush=True)
    print("PARITY2_OK" if ok else "PARITY2_FAILED", flush=True)
    return 0 if ok else 1


def main(cases):
    only = {}
    for i, c in enumerate(cases):
        if ":" in c: c, lab = c.split(":"); only[c] = lab; cases[i] = c
        assert c in CASES, f"unknown case {c}; known: {sorted(CASES)}"
        assert c not in only or only[c] in [r["label"] for r in CASES[c]], f"unknown label {only.get(c)} for case {c}"
    assert selftest(), "parity_phase2 self-test failed: the output gate does not catch what it must"
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
                if case in only and run["label"] != only[case]: continue
                do, ro, so, n = run_side(W, run, "old")
                dn, rn, sn, _ = run_side(W, run, "new")
                print(f"[{run['label']}] inputs linked: {n}", flush=True)
                if ro or rn: ok = False; continue
                if run["stdout"]:
                    if so == sn: print(f"[{run['label']}] stdout identical ({len(so.splitlines())} lines)")
                    else:
                        ok = False; print(f"[{run['label']}] STDOUT DIFFERS")
                        print("".join(list(difflib.unified_diff(so.splitlines(1), sn.splitlines(1), "old", "new"))[:40]))
                good, lines = compare_outputs(do, dn, run, ROOT)
                for l in lines: print(l, flush=True)
                ok = ok and good
            print(f"== case {case}: {'PARITY2_OK' if ok else 'PARITY2_FAILED'}", flush=True)
            ok_all = ok_all and ok
    finally:
        shutil.rmtree(W, ignore_errors=True)
    print("PARITY2_OK" if ok_all else "PARITY2_FAILED", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]: sys.exit(0 if (selftest() & panel_build_selftest()) else 1)
    if sys.argv[1:2] == ["panel_build"]:
        a = sys.argv[2:]
        if len(a) == 4 and a[0] == "--side" and a[2] == "--workdir": sys.exit(panel_build_side(a[1], a[3]))
        if len(a) == 2 and a[0] == "--compare": sys.exit(panel_build_compare(a[1]))
        sys.exit("usage: parity_phase2.py panel_build --side old|new --workdir W  |  panel_build --compare W")
    sys.exit(main(sys.argv[1:] or sorted(CASES)))
