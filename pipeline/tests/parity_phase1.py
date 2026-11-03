"""parity_phase1.py -- proves that pipeline/run.sh + pipeline/tasks.sbatch submit and run exactly what the v5 drivers
(replication_v5/) + the original launchers did at git tag v5-as-run (commit b3ed0e0c28e3).

SAFE ANYWHERE. Everything happens in a fresh temporary directory: code is copied there (no data), the hard-coded project
path is rewritten to that directory in the copies, empty fake inputs are created, and every external command (sbatch,
python, Rscript, module, md5sum, nvidia-smi) is a logging stub. Nothing is submitted; nothing outside the temp dir is written.

BASELINE. The v5 files are NOT read from the working tree. They come from the git tag v5-as-run (default: the repo at the
project root; or --baseline-git REPO), checked against the pinned commit, or from a directory given with --baseline-dir.
Either way every baseline file must match the git blob hash in pipeline/tests/v5_baseline_blobs.txt, whose SHA-1 is pinned
below. So the test still works after replication_v5/ and the launchers are deleted from the tree.

Usage (from the project root, Python >= 3.7, bash >= 4):
  venv/bin/python pipeline/tests/parity_phase1.py [--baseline-git REPO | --baseline-dir DIR] [--skip-selftest]

Checks (exit 0 only if all pass):
 (1) plan: both drivers' run_01..run_09 are executed for real against a stub sbatch that records every argument, the full
     environment and the working directory. Same number of submissions, no duplicate names or ids, same job names, same
     effective sbatch options (launcher #SBATCH header, shell-parsed, overridden by command-line options), same dependency
     graph by job name, identical --wrap commands, identical environment at submission (minus a short excluded list).
 (2) preflight: `declare -f pre_01 .. pre_09` identical between the old stage files and pipeline/run.sh.
 (3) bodies: every submission, including the wrapped CPU jobs, is executed in the sandbox with the environment it would
     get on the cluster (submission environment + --export), for every array index. The stubs log every command with
     its arguments, working directory and full environment (minus the excluded list), plus every `module` call and
     `source venv/bin/activate`. Old and new must give identical exit codes and identical logs. Repeated for the first and
     last array index with (a) `module` failing and (b) every python/Rscript call failing.
 (4) self-test: the same checks are rerun on deliberately broken copies of the new pipeline and must FAIL for each:
     a duplicate tables submission, an exported CRIME_TARGET, a global `module load anaconda` (affects the Brazil jobs),
     a changed learning rate, a changed memory request, a #SBATCH line in tasks.sbatch, an --export=NONE submission, a merged script called without its selector.
Accepted difference (pinned, see SETE_ADDED): 11 jobs whose old launchers had no `set -e` now stop when `module load`
fails instead of carrying on. Known limits: login-shell profile files are not simulated (the wrapped jobs' `bash -lc` is run as `bash -c` on both sides);
analysis scripts are not executed (they are unchanged byte for byte, see pipeline/manifest.csv)."""
import os, re, sys, glob, json, shlex, shutil, hashlib, argparse, subprocess, tempfile, collections, difflib

REAL = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
V5_COMMIT = "b3ed0e0c28e34fd49dc431d93b45ea1e4a3e7743"
BLOB_LIST = "pipeline/tests/v5_baseline_blobs.txt"
BLOB_LIST_SHA1 = "ec6ae67afbeb506493cd05b3b1e680ba37092f5f"
EXCLUDE_ENV = {"_", "SHLVL", "PWD", "OLDPWD", "TRACE", "TASK", "PIPE_ROOT", "V5_ROOT", "SBATCH_LOG", "SBATCH_N",
               "MODE_MODULE_FAIL", "MODE_PY_FAIL"}   # TASK = new dispatch variable; the rest are test plumbing
# Accepted behavior change: these jobs' old launchers had no `set -e`, so when `module load anaconda` failed they carried
# on; tasks.sbatch runs with `set -eo pipefail` and stops. Under the module-failure mode the old run must exit 0 and the new
# run must exit 23 having logged only the module call. Everywhere else they must match exactly.
SETE_ADDED = {"data_seen", "ces_prep", "ces_pretrain", "ces_nested", "ces_finetune", "dengue_ingest", "dom_pretrain",
              "crime_nested", "crime_finetune", "dengue_nested", "dengue_finetune"}
DENGUE_EXPECTED = {"--job-name": (None, "dengue_ingest"), "--output": ("slurm-%j.out", "logs/dengue_ingest_%j.out"),
                   "--error": (None, "logs/dengue_ingest_%j.err")}   # the only allowed option changes (old: no job name, log in root)

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default=".")
ap.add_argument("--baseline-git", default=None)
ap.add_argument("--baseline-dir", default=None)
ap.add_argument("--skip-selftest", action="store_true")
A = ap.parse_args()
SRC = os.path.abspath(A.root)
assert os.path.isfile(os.path.join(SRC, "pipeline/run.sh")), f"{SRC} has no pipeline/run.sh (run from the project root)"

def blob(data): return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
blist_raw = open(os.path.join(SRC, BLOB_LIST), "rb").read()
assert hashlib.sha1(blist_raw).hexdigest() == BLOB_LIST_SHA1, f"{BLOB_LIST} does not match its pinned SHA-1"
BASE = [l.split("\t") for l in blist_raw.decode().splitlines() if l.strip()]

W = tempfile.mkdtemp(prefix="parity_p1_")
R = os.path.join(W, "root"); STUB = os.path.join(W, "stubbin"); TRACE = os.path.join(W, "trace.txt")
SBLOG = os.path.join(W, "sbatch.jsonl")
EXCL_RE = "^(" + "|".join(re.escape(e) for e in sorted(EXCLUDE_ENV)) + ")="

def build_tree():
    """new code from SRC + baseline v5 files from the tag (or pinned dir), path rewritten, fake inputs, stubs."""
    for d in ("pipeline", "scripts", "figures_tables", "env"):
        s = os.path.join(SRC, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(R, d), ignore=shutil.ignore_patterns(
                "*.parquet", "*.csv", "*.gz", "*.pt", "*.npz", "*.pdf", "*.png", "__pycache__"))
    if os.path.isdir(os.path.join(R, "replication_v5")): shutil.rmtree(os.path.join(R, "replication_v5"))
    if A.baseline_dir:
        src = lambda p: open(os.path.join(A.baseline_dir, p), "rb").read()
        print("baseline: directory", A.baseline_dir)
    else:
        repo = A.baseline_git or SRC
        c = subprocess.run(["git", "-C", repo, "rev-parse", "v5-as-run^{commit}"], capture_output=True, text=True).stdout.strip()
        assert c == V5_COMMIT, f"tag v5-as-run in {repo} resolves to '{c}', expected {V5_COMMIT}"
        src = lambda p: subprocess.run(["git", "-C", repo, "show", f"v5-as-run:{p}"], capture_output=True, check=True).stdout
        print("baseline: git tag v5-as-run =", c[:12], "in", repo)
    for h, p in BASE:
        data = src(p)
        assert blob(data) == h, f"baseline file {p} does not match the pinned blob {h[:12]}"
        os.makedirs(os.path.dirname(os.path.join(R, p)), exist_ok=True)
        open(os.path.join(R, p), "wb").write(data)
    print("baseline files verified:", len(BASE))
    for f in glob.glob(os.path.join(R, "**", "*"), recursive=True):
        if os.path.isfile(f) and f.endswith((".sh", ".sbatch")):
            t = open(f, errors="replace").read()
            if REAL in t: open(f, "w").write(t.replace(REAL, R))
    for d in ("venv/bin", "venv_xd/bin", "data_model", "data_intermediate", "outputs/crossdomain", "logs",
              "data_raw/shapefiles/ne_50m_admin_0"):
        os.makedirs(os.path.join(R, d), exist_ok=True)
    LOGENV = f"env | LC_ALL=C sort | grep -vE '{EXCL_RE}' | sed 's/^/  ENV /'"
    py = f"""#!/bin/bash
{{ echo "CMD $0 $*"; echo "  CWD $PWD"; {LOGENV}; }} >> "$TRACE"
[ "${{MODE_PY_FAIL:-0}}" = 1 ] && exit 7
case "$*" in
  *export_ukraine_map*) sh=$(echo "$*" | sed -E 's/.*--share ([^ ]+).*/\\1/'); echo "map uses share $sh" ;;
  *alert_counts_fullsrc.py*) touch outputs/alert_counts_persistence_${{SIZE}}_fullsrc${{OUTSFX}}.csv ;;
  *cell_alerts_agg_fullsrc.py*) touch outputs/cell_alerts_incidence_${{SIZE}}_fullsrc${{OUTSFX}}.csv.gz ;;
  *cell_alerts_persistence_fullsrc.py*) touch outputs/cell_alerts_persistence_${{SIZE}}_fullsrc${{OUTSFX}}.csv.gz ;;
  *cell_month_decompose_fullsrc.py*) touch outputs/cell_month_decompose_${{SIZE}}_fullsrc${{OUTSFX}} ;;
  *alert_profile_fullsrc.py*) touch outputs/alert_profile_incidence_${{SIZE}}_fullsrc${{OUTSFX}}.DONE ;;
  *alert_budget_sweep_fullsrc.py*) touch outputs/alert_budget_sweep_${{SIZE}}_fullsrc${{OUTSFX}}.csv ;;
esac
exit 0
"""
    rs = f"""#!/bin/bash
{{ echo "CMD Rscript $*"; echo "  CWD $PWD"; {LOGENV}; }} >> "$TRACE"
[ "${{MODE_PY_FAIL:-0}}" = 1 ] && exit 7
exit 0
"""
    sb = """#!/usr/bin/env python3
import os, sys, json
rec = {"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ)}
with open(os.environ["SBATCH_LOG"], "a") as f: f.write(json.dumps(rec) + "\\n")
n = int(open(os.environ["SBATCH_N"]).read() or 0) + 1 if os.path.exists(os.environ["SBATCH_N"]) else 1
open(os.environ["SBATCH_N"], "w").write(str(n))
print(5000000 + n)
"""
    os.makedirs(STUB)
    stubs = {"python": py, "Rscript": rs, "sbatch": sb,
             "md5sum": '#!/bin/bash\necho "CMD md5sum $*" >> "$TRACE"\n',
             "module": '#!/bin/bash\necho "CMD module $*" >> "$TRACE"\n[ "${MODE_MODULE_FAIL:-0}" = 1 ] && exit 23\nexit 0\n',
             "nvidia-smi": "#!/bin/bash\necho GPU\n"}
    for n, t in stubs.items():
        open(os.path.join(STUB, n), "w").write(t); os.chmod(os.path.join(STUB, n), 0o755)
    for v in ("venv/bin/python", "venv_xd/bin/python"):
        shutil.copy(os.path.join(STUB, "python"), os.path.join(R, v))
    open(os.path.join(R, "venv/bin/activate"), "w").write('echo "CMD source venv/bin/activate" >> "$TRACE"\n')
    RGS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
    with open(os.path.join(R, "logs/fullsrc_fake.out"), "w") as lg:
        for S in "SML":
            for rg in RGS:
                for sd in (123, 456, 789):
                    t = f"{S}_contrastive_e_fullsrc_loro_{rg}_s{sd}"
                    open(os.path.join(R, f"data_model/repr_model_{t}.pt"), "w").close(); lg.write(f"FULLSRC_DONE_{t}\n")
    for rg in RGS: open(os.path.join(R, f"data_intermediate/seen_tokens_fullsrc_loro_{rg}.json"), "w").close()
    open(os.path.join(R, "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"), "w").close()

OLD = r'''set -euo pipefail
source replication_v5/_common.sh
declare -A RUN=(); for s in 00 01 02 03 04 05 06 07 08 09; do RUN[$s]=1; done
on() { [ -n "${RUN[$1]:-}" ]; }
for f in 00_setup 01_build_panel 02_pretrain 03_frozen_eval 04_finetune_grids 05_bootstrap 06_alerts 07_appendix 08_brazil 09_figures_tables; do source replication_v5/$f.sh; done
'''
NEW = r'''set -euo pipefail
source pipeline/run.sh
for s in 00 01 02 03 04 05 06 07 08 09; do RUN[$s]=1; done
'''
RUNS = 'for s in 01 02 03 04 05 06 07 08 09; do run_$s; done\n'
PRES = 'for s in 01 02 03 04 05 06 07 08 09; do declare -f pre_$s; done\n'

def base_env(extra):
    e = {"PATH": STUB + ":/usr/bin:/bin", "HOME": W, "LC_ALL": "C", "SBATCH_LOG": SBLOG, "SBATCH_N": SBLOG + ".n"}
    e.update(extra); return e

def drive(src, rootvar, tail):
    for f in (SBLOG, SBLOG + ".n"):
        if os.path.exists(f): os.remove(f)
    p = subprocess.run(["bash", "-c", src + tail], capture_output=True, text=True, cwd=R, env=base_env({rootvar: R}))
    if p.returncode: print(p.stderr[-2000:]); raise SystemExit("driver failed")
    return p.stdout

VALUE_OPTS = {"-o", "-e", "-J", "-c", "-A", "-p", "-t", "-a", "-N", "-n", "--output", "--error", "--job-name", "--account",
              "--partition", "--time", "--array", "--mem", "--gres", "--cpus-per-task", "--dependency", "--export", "--wrap"}
ALIAS = {"-o": "--output", "-e": "--error", "-J": "--job-name", "-c": "--cpus-per-task", "-A": "--account",
         "-p": "--partition", "-t": "--time", "-a": "--array"}
def parse_opts(tokens):
    o, pos, k = {}, [], 0
    while k < len(tokens):
        t = tokens[k]
        if t.startswith("--") and "=" in t: key, v = t.split("=", 1)
        elif t in VALUE_OPTS: key, v = t, tokens[k + 1]; k += 1
        elif t.startswith("-"): key, v = t, ""
        else: pos.append(t); k += 1; continue
        o[ALIAS.get(key, key)] = v; k += 1
    return o, pos
def header(path):
    toks = []
    for l in open(path):
        if l.startswith("#SBATCH"): toks += shlex.split(l[len("#SBATCH"):], comments=True)
    return parse_opts(toks)[0]

def clear_driver_logs():
    for g in ("logs/run_all_v5_*", "logs/pipeline_*"):
        for f in glob.glob(os.path.join(R, g)): os.remove(f)

def plan(src, rootvar):
    clear_driver_logs()
    drive(src, rootvar, RUNS)
    rows = [json.loads(l) for l in open(SBLOG)] if os.path.exists(SBLOG) else []
    out = []
    for i, r in enumerate(rows):
        o, pos = parse_opts(r["argv"])
        script = pos[-1] if pos else None
        wrap = o.pop("--wrap", None)
        eff = header(os.path.join(R, script)) if script else {}   # every job script's #SBATCH lines, tasks.sbatch included
        eff.update(o); eff.pop("--parsable", None)
        exp = eff.pop("--export", "")
        # export mode (first comma field). sbatch's default is ALL. The body environment below is built as ALL
        # (submission environment + explicit assignments, explicit wins), so any other mode is compared AND rejected.
        mode = exp.split(",")[0] if exp else ""
        eff["--export(mode)"] = "ALL" if mode in ("", "ALL") else mode
        exports = dict(kv.split("=", 1) for kv in exp.split(",")[1:]) if exp else {}
        env = {k: v for k, v in r["env"].items() if k not in EXCLUDE_ENV}
        out.append(dict(id=str(5000001 + i), opts=eff, exports=exports, script=script, wrap=wrap, env=env, cwd=r["cwd"],
                        name=None))
    return out

def attach_names(rows, logglob):
    log = sorted(glob.glob(os.path.join(R, logglob)), key=os.path.getmtime)[-1]
    sub = [l.split() for l in open(log) if l.startswith("SUBMIT ")]
    assert len(sub) == len(rows), f"driver log has {len(sub)} SUBMIT lines for {len(rows)} sbatch calls"
    for (tag, nm, jid), r in zip(sub, rows):
        assert jid == r["id"], (nm, jid, r["id"]); r["name"] = nm
    for f in glob.glob(os.path.join(R, logglob)): os.remove(f)
    return rows

def compare_plans(Orows, Nrows):
    fails = []
    for side, rows in (("old", Orows), ("new", Nrows)):
        names = [r["name"] for r in rows]; ids = [r["id"] for r in rows]
        if len(set(names)) != len(names): fails.append((side, "duplicate job names", [n for n, c in collections.Counter(names).items() if c > 1]))
        if len(set(ids)) != len(ids): fails.append((side, "duplicate job ids"))
    if len(Orows) != len(Nrows): fails.append(("submission count", len(Orows), len(Nrows)))
    for side, rows in (("old", Orows), ("new", Nrows)):
        for r in rows:
            if r["opts"].get("--export(mode)") != "ALL":
                fails.append((side, r["name"], "unsupported --export mode (test models ALL only)", r["opts"].get("--export(mode)")))
    # resources belong in run.sh: the single job script must carry no #SBATCH directive (Slurm would apply it)
    sb = [l.rstrip() for l in open(os.path.join(R, "pipeline/tasks.sbatch")) if l.startswith("#SBATCH")]
    if sb: fails.append(("pipeline/tasks.sbatch contains #SBATCH directives", sb))
    O = {r["name"]: r for r in Orows}; N = {r["name"]: r for r in Nrows}
    for T in (O, N):
        i2n = {r["id"]: n for n, r in T.items()}
        for r in T.values():
            d = r["opts"].pop("--dependency", "")
            r["deps"] = sorted(i2n.get(x, "?" + x) for x in d.replace("afterok:", "").split(":") if x)
    if set(O) != set(N): fails.append(("job set", sorted(set(O) ^ set(N))))
    for n in sorted(set(O) & set(N)):
        a, b = O[n], N[n]
        for k in sorted(set(a["opts"]) | set(b["opts"])):
            va, vb = a["opts"].get(k), b["opts"].get(k)
            if va != vb and not (n == "dengue_ingest" and DENGUE_EXPECTED.get(k) == (va, vb)):
                fails.append((n, k, va, vb))
        if n == "dengue_ingest":
            for k, (va, vb) in DENGUE_EXPECTED.items():
                if (a["opts"].get(k), b["opts"].get(k)) != (va, vb): fails.append((n, "dengue exception not as pinned", k))
        if a["deps"] != b["deps"]: fails.append((n, "deps", a["deps"], b["deps"]))
        if a["wrap"] != chain_wrap(n, b["wrap"]): fails.append((n, "wrap differs"))
        if a["env"] != b["env"]:
            ks = sorted(k for k in set(a["env"]) | set(b["env"]) if a["env"].get(k) != b["env"].get(k))
            fails.append((n, "environment at submission differs", ks))
        if a["cwd"] != b["cwd"]: fails.append((n, "cwd", a["cwd"], b["cwd"]))
    return O, N, fails

def indices(spec):
    if not spec: return [None]
    out = []
    for part in spec.split(","):
        if "-" in part: lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(part))
    return out

def clean():
    for g in ("outputs/alert_*", "outputs/cell_*", "outputs/ukraine_maps_*", "outputs/eval_region_*", "logs/ukraine_mapfs_export_*",
              "outputs/ukraine_alert_gain_*", "logs/domain_tp_fp_*", "figures_tables/v5", "figures_tables/tables_v5"):
        for f in glob.glob(os.path.join(R, g)): shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)

def run_body(row, idx, mode, jobname):
    if os.path.exists(TRACE): os.remove(TRACE)
    clean()
    if jobname == "figures":                      # the figures job requires the Fig 19 png written by stage 06
        os.makedirs(os.path.join(R, "outputs/cell_alerts_fig_M_fullsrc"), exist_ok=True)
        open(os.path.join(R, "outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png"), "w").close()
    if jobname.startswith("rand_"):
        for f in glob.glob(os.path.join(R, "data_model/repr_model_*_contrastive_e_fullsrc_loro_*.pt")):
            open(os.path.join(R, f"outputs/eval_region_{os.path.basename(f)[11:-3]}.csv"), "w").close()
    hidden = []
    if jobname.startswith("pretrain_"):
        for f in glob.glob(os.path.join(R, "data_model/*.pt")): os.rename(f, f + ".hide"); hidden.append(f)
    env = dict(row["env"]); env.update(row["exports"])
    env.update({"SLURM_JOB_ID": "777", "TRACE": TRACE, "MODE_MODULE_FAIL": "1" if mode == "module_fail" else "0",
                "MODE_PY_FAIL": "1" if mode == "py_fail" else "0"})
    if idx is not None: env["SLURM_ARRAY_TASK_ID"] = str(idx)
    if row["wrap"] is not None:
        cmd = ["sh", "-c", row["wrap"].replace("bash -lc '", "bash -c '", 1)]
    else:
        cmd = ["bash", os.path.join(R, row["script"])]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=row["cwd"])
    for f in hidden: os.rename(f + ".hide", f)
    tr = open(TRACE).read() if os.path.exists(TRACE) else ""
    return p.returncode, tr

# Phase 2 merges: jobs whose old script(s) were merged into one new script. In the NEW trace the merged script's name is
# mapped back to the old one and the selector variable the old script did not receive is dropped; everything else must
# match exactly. Output equivalence of each merge is proven separately by pipeline/tests/parity_phase2.py.
PHASE2_MAP = {
    "xd_boot_M": [("scripts/crossdomain/xd_boot.py", "scripts/crossdomain/xd_m_boot.py"), ("  ENV XD_CAP=M\n", "")],
    "xd_boot_S": [("scripts/crossdomain/xd_boot.py", "scripts/crossdomain/xd_s_boot.py"), ("  ENV XD_CAP=S\n", "")],
    "br_sim_hom": [("scripts/crossdomain/prep_sim.py", "scripts/crossdomain/prep_sim_homicide.py"), ("  ENV XD_CAUSE=homicide\n", "")],
    "br_traffic": [("scripts/crossdomain/prep_sim.py", "scripts/crossdomain/prep_sim_traffic.py"), ("  ENV XD_CAUSE=traffic\n", "")],
    "br_placebos": [("scripts/crossdomain/prep_sim.py", "scripts/crossdomain/prep_sim_placebos_v2.py"), ("  ENV XD_CAUSE=placebos\n", "")],
}
def phase2_normalize(job, trace, mode):
    for a, b in PHASE2_MAP.get(job, []):
        if mode == "ok" and a not in trace:   # the merged script must really be called with its selector
            return trace + f"\nPHASE2_MAP_ITEM_NOT_FOUND {a!r}\n"
        trace = trace.replace(a, b)
    return trace


# Phase 2 merges of a chain of calls in a wrapped job into one script. Plan: the new --wrap with the merged call replaced
# by the old calls must equal the old --wrap exactly. Bodies: in the OLD trace the run of old call records (all with the
# same working directory and environment block) is collapsed into one record with the merged call; the result must equal
# the NEW trace exactly. Under py_fail the chain stops at its first call, so only that record is present.
_FO, _P, _V = "figures_tables/v5", "scripts/phase4_scaling", "scripts/replication_v5"
_NE = "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"
PHASE2_CHAIN = {
    "sudan_cells_keep": dict(
        new_wrap=f"venv/bin/python {_P}/sudan_cells.py",
        old_wrap=f"venv/bin/python {_P}/audit_cells_export.py; Rscript {_P}/audit_cell_country.R; venv/bin/python {_P}/sudan_cells_fix.py",
        new_cmd=f"CMD venv/bin/python {_P}/sudan_cells.py",
        old_cmds=[f"CMD venv/bin/python {_P}/audit_cells_export.py", f"CMD Rscript {_P}/audit_cell_country.R",
                  f"CMD venv/bin/python {_P}/sudan_cells_fix.py"]),
    "figures": dict(
        new_wrap=f"Rscript {_V}/figures.R {_FO} {{R}}",
        old_wrap=("; " + " " * 5).join([   # old run.sh lines: inside double quotes bash drops each backslash-newline, keeping the indent
            f"Rscript {_P}/fig_loro_gains_v2.R outputs/bootstrap_ci_M_fullsrc_block5.csv {_FO}/fig3_loro_gains",
            f"mkdir -p {_FO}/fig4_cell_maps; Rscript {_V}/fig_cell_maps.R outputs/cell_alerts_persistence_M_fullsrc.csv.gz outputs/cell_alerts_incidence_M_fullsrc.csv.gz {_NE} {_FO}/fig4_cell_maps",
            f"mkdir -p {_FO}/fig5_finetune_gains; Rscript {_P}/fig_finetune_gains.R outputs/bootstrap_ft_fullsrc_M_full_block5.csv {_FO}/fig5_finetune_gains",
            f"mkdir -p {_FO}/fig6_brazil; Rscript {_P}/fig_placebo_gradient_a_only_nolabel.R outputs/crossdomain/xd_placebo_boot_v3M.csv {_FO}/fig6_brazil",
            f"mkdir -p {_FO}/fig9_random_encoder {_FO}/fig10_share_curve {_FO}/fig17_share_regions {_FO}/fig18_persistence_gap {_FO}/fig16_brazil_random",
            f"Rscript {_V}/fig_random_encoder.R {_FO}/fig9_random_encoder {{R}}",
            f"Rscript {_V}/fig_share_curve.R {_FO}/fig10_share_curve {{R}}",
            f"Rscript {_P}/fig_placebo_gradient.R outputs/crossdomain/xd_placebo_boot_v3M.csv {_FO}/fig16_brazil_random",
            f"Rscript {_V}/fig_share_regions.R {_FO}/fig17_share_regions {{R}}",
            f"Rscript {_V}/fig_persistence_gap.R {_FO}/fig18_persistence_gap {{R}}"]),
        new_cmd=f"CMD Rscript {_V}/figures.R {_FO} {{R}}",
        old_cmds=[f"CMD Rscript {_P}/fig_loro_gains_v2.R outputs/bootstrap_ci_M_fullsrc_block5.csv {_FO}/fig3_loro_gains",
                  f"CMD Rscript {_V}/fig_cell_maps.R outputs/cell_alerts_persistence_M_fullsrc.csv.gz outputs/cell_alerts_incidence_M_fullsrc.csv.gz {_NE} {_FO}/fig4_cell_maps",
                  f"CMD Rscript {_P}/fig_finetune_gains.R outputs/bootstrap_ft_fullsrc_M_full_block5.csv {_FO}/fig5_finetune_gains",
                  f"CMD Rscript {_P}/fig_placebo_gradient_a_only_nolabel.R outputs/crossdomain/xd_placebo_boot_v3M.csv {_FO}/fig6_brazil",
                  f"CMD Rscript {_V}/fig_random_encoder.R {_FO}/fig9_random_encoder {{R}}",
                  f"CMD Rscript {_V}/fig_share_curve.R {_FO}/fig10_share_curve {{R}}",
                  f"CMD Rscript {_P}/fig_placebo_gradient.R outputs/crossdomain/xd_placebo_boot_v3M.csv {_FO}/fig16_brazil_random",
                  f"CMD Rscript {_V}/fig_share_regions.R {_FO}/fig17_share_regions {{R}}",
                  f"CMD Rscript {_V}/fig_persistence_gap.R {_FO}/fig18_persistence_gap {{R}}"]),
}
def chain_wrap(job, wrap):
    """new --wrap -> what the old --wrap should be (unchanged if the job has no chain merge)."""
    c = PHASE2_CHAIN.get(job)
    if c is None or wrap is None: return wrap
    nw, ow = c["new_wrap"].replace("{R}", R), c["old_wrap"].replace("{R}", R)
    if wrap.count(nw) != 1: return wrap + " PHASE2_CHAIN_NEW_CALL_NOT_FOUND"
    return wrap.replace(nw, ow)
def chain_collapse(job, trace, mode):
    """old trace -> what the new trace should be."""
    c = PHASE2_CHAIN.get(job)
    if c is None: return trace
    olds, new = [x.replace("{R}", R) for x in c["old_cmds"]], c["new_cmd"].replace("{R}", R)
    recs, cur = [], None
    for line in trace.splitlines(keepends=True):
        if line.startswith("CMD ") or cur is None: cur = [line]; recs.append(cur)
        else: cur.append(line)
    heads = [r[0].rstrip("\n") for r in recs]
    if olds[0] not in heads: return trace
    i = heads.index(olds[0])
    k = len(olds) if mode == "ok" else 1
    if heads[i:i + k] != olds[:k] or any(recs[j][1:] != recs[i][1:] for j in range(i, i + k)):
        return trace + "\nPHASE2_CHAIN_OLD_CALLS_NOT_AS_PINNED\n"
    merged = [new + "\n"] + recs[i][1:]
    return "".join("".join(r) for r in recs[:i]) + "".join(merged) + "".join("".join(r) for r in recs[i + k:])

accepted = [0]
def compare_bodies(O, N, only=None):
    fails, nb = [], 0
    for n in sorted(set(O) & set(N)):
        if only and n not in only: continue
        a, b = O[n], N[n]
        idxs = indices(a["opts"].get("--array"))
        plan_ = [(i, "ok") for i in idxs] + [(i, m) for i in sorted({idxs[0], idxs[-1]}, key=lambda x: (x is None, x)) for m in ("module_fail", "py_fail")]
        for idx, mode in plan_:
            (ro, to), (rn, tn) = run_body(a, idx, mode, n), run_body(b, idx, mode, n)
            tn = phase2_normalize(n, tn, mode)
            to = chain_collapse(n, to, mode)
            nb += 1
            if mode == "module_fail" and n in SETE_ADDED:
                bad = not (ro == 0 and rn == 23 and tn == "CMD module load anaconda\n" and to.startswith(tn))
                accepted[0] += 0 if bad else 1
            else:
                bad = ro != rn or to != tn or (mode == "ok" and (ro != 0 or not to))
            if bad: fails.append((n, idx, mode, ro, rn, to, tn))
    clean()
    return nb, fails

def compare_pre():
    po = drive(OLD, "V5_ROOT", PRES); pn = drive(NEW, "PIPE_ROOT", PRES); clear_driver_logs()
    return [] if po == pn else [("preflight functions differ", "".join(difflib.unified_diff(po.splitlines(1), pn.splitlines(1), n=0))[:3000])]

def full_check(only_bodies=None, bodies=True):
    Orows = attach_names(plan(OLD, "V5_ROOT"), "logs/run_all_v5_*.log*")
    Nrows = attach_names(plan(NEW, "PIPE_ROOT"), "logs/pipeline_*.log*")
    O, N, pf = compare_plans(Orows, Nrows)
    pre = compare_pre()
    nb, bf = compare_bodies(O, N, only_bodies) if bodies else (0, [])
    return dict(nold=len(Orows), nnew=len(Nrows), plan=pf, pre=pre, nbodies=nb, bodies=bf)

def report(r, label):
    print(f"[{label}] submissions old {r['nold']} new {r['nnew']}; plan differences {len(r['plan'])}; "
          f"preflight differences {len(r['pre'])}; bodies run {r['nbodies']} differences {len(r['bodies'])}; "
          f"accepted set -e stops {accepted[0]}")
    for f in r["plan"][:10]: print("   PLAN", f)
    for f in r["pre"]: print("   PRE", f[0]); print(f[1])
    if r["bodies"]: print("   failing jobs:", dict(collections.Counter((f[0], f[2]) for f in r["bodies"])))
    for n, idx, mode, ro, rn, to, tn in r["bodies"][:4]:
        print(f"   BODY {n}[{idx}] {mode} rc old={ro} new={rn}")
        for l in list(difflib.unified_diff(to.splitlines(), tn.splitlines(), "old", "new", lineterm="", n=0))[:12]: print("     " + l)
    return not (r["plan"] or r["pre"] or r["bodies"])

MUTATIONS = [  # (name, file, old text, new text, bodies to run)
    ("duplicate tables submission", "pipeline/run.sh",
     '  wrapcpu tables 8G 00:30:00 "$D" "TABLES_ROOT=$ROOT venv/bin/python $V/make_tables.py --ukraine-full" >/dev/null\n',
     '  wrapcpu tables 8G 00:30:00 "$D" "TABLES_ROOT=$ROOT venv/bin/python $V/make_tables.py --ukraine-full" >/dev/null\n'
     '  wrapcpu tables 8G 00:30:00 "$D" "TABLES_ROOT=$ROOT venv/bin/python $V/make_tables.py --ukraine-full" >/dev/null\n', ["tables"]),
    ("exported CRIME_TARGET", "pipeline/run.sh", "set -euo pipefail\n", "set -euo pipefail\nexport CRIME_TARGET=homicide\n", ["crime_ingest"]),
    ("global module load", "pipeline/tasks.sbatch", "cd $ROOT\n", "cd $ROOT\nmodule load anaconda\n", ["br_sim_hom"]),
    ("changed learning rate", "pipeline/tasks.sbatch", "L) LR=3e-4", "L) LR=3e-3", ["pretrain_L"]),
    ("changed memory request", "pipeline/run.sh", "--mem=64G --time=06:00:00", "--mem=32G --time=06:00:00", []),
    ("#SBATCH directive in tasks.sbatch", "pipeline/tasks.sbatch", "#!/bin/bash\n", "#!/bin/bash\n#SBATCH --constraint=__selftest__\n", ["br_sim_hom"]),
    ("merged script without its selector", "pipeline/tasks.sbatch", "  export XD_CAP\n  md5sum $X/xd_boot.py", "  unset XD_CAP\n  md5sum $X/xd_boot.py", ["xd_boot_M"]),
    ("merged figures script without the root argument", "pipeline/run.sh", "Rscript $V/figures.R $FO $ROOT", "Rscript $V/figures.R $FO", ["figures"]),
    ("merged Sudan script replaced by its first step", "pipeline/run.sh", '"venv/bin/python $P/sudan_cells.py"', '"venv/bin/python $P/audit_cells_export.py"', ["sudan_cells_keep"]),
    ("merged SIM prep with the wrong cause", "pipeline/tasks.sbatch", "XD_CAUSE=homicide venv_xd/bin/python $X/prep_sim.py", "XD_CAUSE=traffic venv_xd/bin/python $X/prep_sim.py", ["br_sim_hom"]),
    ("export mode NONE", "pipeline/run.sh", "D_PRIO=$(submit data_prio --account=open", "D_PRIO=$(submit data_prio --export=NONE --account=open", ["data_prio"]),
]

ok = False
try:
    build_tree()
    r = full_check()
    ok = report(r, "main")
    if not A.skip_selftest:
        for name, f, a, b, only in MUTATIONS:
            path = os.path.join(R, f); orig = open(path).read()
            assert orig.count(a) == 1, f"self-test '{name}': anchor not found exactly once in {f}"
            open(path, "w").write(orig.replace(a, b))
            rr = full_check(only_bodies=set(only) or {"__none__"})
            open(path, "w").write(orig)
            caught = bool(rr["plan"] or rr["pre"] or rr["bodies"])
            print(f"[self-test] {name}: {'caught' if caught else 'NOT CAUGHT'} "
                  f"(plan {len(rr['plan'])}, bodies {len(rr['bodies'])})")
            ok = ok and caught
finally:
    shutil.rmtree(W, ignore_errors=True)
print("PARITY_OK" if ok else "PARITY_FAILED")
sys.exit(0 if ok else 1)
