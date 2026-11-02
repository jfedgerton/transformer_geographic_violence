"""parity_phase1.py -- checks that pipeline/run.sh + pipeline/tasks.sbatch submit exactly what the v5 drivers
(replication_v5/, tag v5-as-run) + the 37 original launchers submitted.

SAFE ANYWHERE: copies the code (no data) into a fresh temporary directory, rewrites the hard-coded project path in the
copies to that directory, creates empty fake inputs there and runs everything against stub commands (python, Rscript,
md5sum, module, nvidia-smi). Nothing is submitted, nothing in the real tree is read beyond code or written.

Usage (from the project root):  python3 pipeline/tests/parity_phase1.py [project_root]
Checks:
 (1) plan: both drivers' submission functions in DRY mode -> same 77 job names, same effective sbatch options (launcher
     #SBATCH header overridden by the command line), same dependency graph by job name, identical --wrap commands.
 (2) bodies: every job x every array index, old launcher vs new tasks.sbatch branch, under stubs: the logged command
     lines (python/Rscript/md5sum with arguments) and the model-relevant environment at each call must be identical.
Exit 0 only if both checks show zero differences."""
import os, re, sys, glob, shutil, subprocess, tempfile, collections, difflib

REAL = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
for d in ("pipeline", "replication_v5", "scripts"):
    assert os.path.isdir(os.path.join(SRC, d)), f"{SRC} has no {d}/ (run from the project root)"
W = tempfile.mkdtemp(prefix="parity_p1_")
R = os.path.join(W, "root"); STUB = os.path.join(W, "stubbin"); TRACE = os.path.join(W, "trace.txt")
print("scratch tree:", R)

# ---------------- scratch tree: code only, project path rewritten ----------------
for d in ("pipeline", "replication_v5", "scripts", "figures_tables", "env"):
    s = os.path.join(SRC, d)
    if os.path.isdir(s):
        shutil.copytree(s, os.path.join(R, d), ignore=shutil.ignore_patterns("*.parquet", "*.csv", "*.gz", "*.pt", "*.npz", "__pycache__"))
for f in glob.glob(os.path.join(R, "**", "*"), recursive=True):
    if os.path.isfile(f) and f.endswith((".sh", ".sbatch")):
        t = open(f, errors="replace").read()
        if REAL in t: open(f, "w").write(t.replace(REAL, R))
for d in ("venv/bin", "venv_xd/bin", "data_model", "data_intermediate", "outputs/crossdomain", "logs", "data_raw/shapefiles/ne_50m_admin_0"):
    os.makedirs(os.path.join(R, d), exist_ok=True)
VARS = (r"^(REPR_|EVAL_|GRID_|BOOT_|SIZE=|XD_|SLICE_|WIN_|UKR_|OUTSFX|DIV=|SEED=|SMOKE|EPOCHS|PATIENCE|DOMAIN|REGION|CITY|"
        r"VOCAB_REMAP|RANDOM_ENCODER|PHASE_MODE|PERS_|PYTORCH|NE_ZIP|R_PROFILE|FS_ROOT|AP_ROOT|SWEEP_ROOT|TABLES_ROOT)")
PY = f"""#!/bin/bash
{{ echo "CMD $0 $*"; env | grep -E '{VARS}' | sort | sed 's/^/  ENV /'; }} >> "$TRACE"
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
os.makedirs(STUB)
stubs = {"python": PY,
         "Rscript": f"#!/bin/bash\n{{ echo \"CMD Rscript $*\"; env | grep -E '{VARS}' | sort | sed 's/^/  ENV /'; }} >> \"$TRACE\"; exit 0\n",
         "md5sum": '#!/bin/bash\necho "CMD md5sum $*" >> "$TRACE"\n',
         "module": "#!/bin/bash\nexit 0\n", "nvidia-smi": "#!/bin/bash\necho GPU\n",
         "sbatch": "#!/bin/bash\necho 'stub sbatch: must not be called' >&2; exit 1\n"}
for n, t in stubs.items():
    open(os.path.join(STUB, n), "w").write(t); os.chmod(os.path.join(STUB, n), 0o755)
for v in ("venv/bin/python", "venv_xd/bin/python"):
    shutil.copy(os.path.join(STUB, "python"), os.path.join(R, v))
open(os.path.join(R, "venv/bin/activate"), "w").close()
RGS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
with open(os.path.join(R, "logs/fullsrc_fake.out"), "w") as lg:
    for S in "SML":
        for rg in RGS:
            for sd in (123, 456, 789):
                t = f"{S}_contrastive_e_fullsrc_loro_{rg}_s{sd}"
                open(os.path.join(R, f"data_model/repr_model_{t}.pt"), "w").close(); lg.write(f"FULLSRC_DONE_{t}\n")
for rg in RGS: open(os.path.join(R, f"data_intermediate/seen_tokens_fullsrc_loro_{rg}.json"), "w").close()
open(os.path.join(R, "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"), "w").close()
os.chdir(R)

# ---------------- (1) plan ----------------
OLD = r'''set -euo pipefail; export DRY=1 V5_ROOT=%s PATH=%s:$PATH
source replication_v5/_common.sh
declare -A RUN=(); for s in 00 01 02 03 04 05 06 07 08 09; do RUN[$s]=1; done
on() { [ -n "${RUN[$1]:-}" ]; }
for f in 00_setup 01_build_panel 02_pretrain 03_frozen_eval 04_finetune_grids 05_bootstrap 06_alerts 07_appendix 08_brazil 09_figures_tables; do source replication_v5/$f.sh; done
for s in 01 02 03 04 05 06 07 08 09; do run_$s; done
echo "$LOG"''' % (R, STUB)
NEW = r'''set -euo pipefail; export DRY=1 PIPE_ROOT=%s PATH=%s:$PATH
source pipeline/run.sh
for s in 00 01 02 03 04 05 06 07 08 09; do RUN[$s]=1; done
for s in 01 02 03 04 05 06 07 08 09; do run_$s; done
echo "$LOG"''' % (R, STUB)
def plan(script):
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if out.returncode: print(out.stderr); sys.exit("driver failed")
    log = out.stdout.strip().splitlines()[-1]
    rows = [l.rstrip("\n") for l in open(log) if l.startswith("DRY-SUBMIT ")]
    for f in glob.glob(log + "*"): os.remove(f)
    return rows
ALIAS = {"-o": "--output", "-e": "--error", "-c": "--cpus-per-task", "-J": "--job-name"}
def header(path):
    o = {}
    for l in open(path):
        m = re.match(r"#SBATCH\s+(\S+)", l)
        if m:
            k, _, v = m.group(1).partition("="); o[ALIAS.get(k, k)] = v
    return o
def parse(row):
    wrap = None
    if " --wrap " in row: row, wrap = row.split(" --wrap ", 1)     # the DRY log prints $*, so the wrap is the rest of the line
    t = row.split(); name, did, args = t[1], t[2], t[5:]          # after 'sbatch --parsable'
    o, script = {}, None
    for a in args:
        if a.startswith("-"): k, _, v = a.partition("="); o[ALIAS.get(k, k)] = v
        else: script = a
    return name, did, o, script, wrap
def table(rows):
    T = {}
    for r in rows:
        name, did, o, script, wrap = parse(r)
        e = header(script) if script and not script.endswith("pipeline/tasks.sbatch") else {}
        e.update(o); e.pop("--parsable", None)
        exp = e.pop("--export", "")
        T[name] = dict(id=did, opts=e, exports=dict(kv.split("=", 1) for kv in exp.split(",")[1:]) if exp else {}, script=script, wrap=wrap)
    id2name = {v["id"]: k for k, v in T.items()}
    for v in T.values():
        d = v["opts"].pop("--dependency", "")
        v["deps"] = sorted(id2name[x] for x in d.replace("afterok:", "").split(":") if x)
    return T
O, N = table(plan(OLD)), table(plan(NEW))
print("submissions: old", len(O), "new", len(N))
ALLOWED = {"dengue_ingest": {"--job-name", "--output", "--error"}}   # old launcher: no job name, log slurm-%j.out in the root
fails = []
if set(O) != set(N): fails.append(("job set", sorted(set(O) ^ set(N))))
for n in sorted(set(O) & set(N)):
    a, b = O[n], N[n]
    for k in sorted(set(a["opts"]) | set(b["opts"])):
        if a["opts"].get(k) != b["opts"].get(k) and k not in ALLOWED.get(n, ()): fails.append((n, k, a["opts"].get(k), b["opts"].get(k)))
    if a["deps"] != b["deps"]: fails.append((n, "deps", a["deps"], b["deps"]))
    if a["wrap"] != b["wrap"]: fails.append((n, "wrap differs"))
    # exports are not compared directly (old launchers took their mode from XD_JOBSET etc., the new branches from TASK);
    # check (2) is what verifies that each job gets the same settings
print("plan differences:", len(fails))
for f in fails: print("  PLAN", f)

# ---------------- (2) bodies ----------------
def indices(spec):
    if not spec: return [None]
    out = []
    for part in spec.split(","):
        if "-" in part: lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else: out.append(int(part))
    return out
def clean():
    for g in ("outputs/alert_*", "outputs/cell_*", "outputs/ukraine_maps_*", "outputs/eval_region_*", "logs/ukraine_mapfs_export_*"):
        for f in glob.glob(g): shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
def run_body(script, exports, idx):
    if os.path.exists(TRACE): os.remove(TRACE)
    env = {"PATH": STUB + ":/usr/bin:/bin", "HOME": W, "SLURM_JOB_ID": "777", "TRACE": TRACE}
    if idx is not None: env["SLURM_ARRAY_TASK_ID"] = str(idx)
    env.update(exports)
    p = subprocess.run(["bash", script], env=env, capture_output=True, text=True, cwd=R)
    return p.returncode, (open(TRACE).read() if os.path.exists(TRACE) else "")
bfail, nb = [], 0
for n in sorted(set(O) & set(N)):
    a, b = O[n], N[n]
    if a["wrap"] is not None: continue
    for idx in indices(a["opts"].get("--array")):
        res = []
        for side in (a, b):
            clean()
            if n.startswith("rand_"):                 # eval_rand needs the pretrained eval csv
                for f in glob.glob("data_model/repr_model_*_contrastive_e_fullsrc_loro_*.pt"):
                    open(f"outputs/eval_region_{os.path.basename(f)[11:-3]}.csv", "w").close()
            if n.startswith("pretrain_"):             # pretraining refuses to overwrite an existing checkpoint
                for f in glob.glob("data_model/*.pt"): os.rename(f, f + ".hide")
            res.append(run_body(side["script"], side["exports"], idx))
            for f in glob.glob("data_model/*.pt.hide"): os.rename(f, f[:-5])
        nb += 1
        (ro, to), (rn, tn) = res
        if ro or rn or to != tn or not to: bfail.append((n, idx, ro, rn, to, tn))
clean()
print("bodies compared:", nb, "differences:", len(bfail))
if bfail: print("  failing jobs:", dict(collections.Counter(f[0] for f in bfail)))
for n, idx, ro, rn, to, tn in bfail[:6]:
    print(f"  BODY {n}[{idx}] rc old={ro} new={rn}")
    for l in difflib.unified_diff(to.splitlines(), tn.splitlines(), "old", "new", lineterm="", n=0): print("    " + l)
shutil.rmtree(W)
ok = not fails and not bfail
print("PARITY_OK" if ok else "PARITY_FAILED")
sys.exit(0 if ok else 1)
