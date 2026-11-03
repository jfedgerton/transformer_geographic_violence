"""parity_phase2.py -- output parity for Phase 2 script merges (CPU post-processing). For each case, the OLD scripts
(taken from git tag v5-as-run, commit b3ed0e0c28e3, so the test survives their removal) and the NEW merged script are run
on the SAME real input files and must write byte-identical outputs (and identical stdout). Each output is also compared
with the published file in outputs/ that the paper uses.

SAFE: every run gets its own scratch directory under $TMPDIR; inputs are symlinked into it (read-only use), outputs are
written only there. Nothing in outputs/ is written, moved or deleted. Needs the real inputs, so run it on Roar, e.g.
  sbatch --account=open --partition=open --mem=64G --time=02:00:00 -o logs/parity_p2_%j.out \
         --wrap "cd <root> && venv/bin/python pipeline/tests/parity_phase2.py xd_boot"
Exit 0 only if every compared file and stdout matches (PARITY2_OK)."""
import os, sys, glob, shutil, hashlib, subprocess, tempfile, filecmp

ROOT = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
V5_COMMIT = "b3ed0e0c28e34fd49dc431d93b45ea1e4a3e7743"
PY = f"{ROOT}/venv/bin/python"

# case -> list of runs. A run: (label, old script at the tag, old env, new script in the tree, new env,
#                                input globs relative to ROOT to symlink into the run dir, outputs to compare, published dir)
CASES = {
    "xd_boot": [
        ("M", "scripts/crossdomain/xd_m_boot.py", {}, "scripts/crossdomain/xd_boot.py", {"XD_CAP": "M"},
         ["outputs/crossdomain/preds_brazil_*.npz"], ["xd_m_boot.csv"], "outputs/crossdomain"),
        ("S", "scripts/crossdomain/xd_s_boot.py", {}, "scripts/crossdomain/xd_boot.py", {"XD_CAP": "S"},
         ["outputs/crossdomain/preds_brazil_*.npz"], ["xd_s_boot.csv"], "outputs/crossdomain"),
    ],
}
# env var each case uses for its input/output directory
DIRVAR = {"xd_boot": "XC_DIR"}

def sha(p): return hashlib.sha1(open(p, "rb").read()).hexdigest()

def main(case):
    assert case in CASES, f"unknown case {case}; known: {sorted(CASES)}"
    c = subprocess.run(["git", "-C", ROOT, "rev-parse", "v5-as-run^{commit}"], capture_output=True, text=True).stdout.strip()
    assert c == V5_COMMIT, f"tag v5-as-run resolves to '{c}', expected {V5_COMMIT}"
    W = tempfile.mkdtemp(prefix=f"parity_p2_{case}_", dir=os.environ.get("TMPDIR", "/tmp"))
    print("scratch:", W, flush=True)
    ok = True
    try:
        for label, old_s, old_env, new_s, new_env, globs, outs, pub in CASES[case]:
            old_copy = os.path.join(W, "old_scripts", old_s)
            os.makedirs(os.path.dirname(old_copy), exist_ok=True)
            open(old_copy, "wb").write(subprocess.run(["git", "-C", ROOT, "show", f"v5-as-run:{old_s}"],
                                                      capture_output=True, check=True).stdout)
            res = {}
            for side, script, env_extra in (("old", old_copy, old_env), ("new", os.path.join(ROOT, new_s), new_env)):
                d = os.path.join(W, f"{label}_{side}"); os.makedirs(d)
                n = 0
                for g in globs:
                    for f in sorted(glob.glob(os.path.join(ROOT, g))):
                        os.symlink(f, os.path.join(d, os.path.basename(f))); n += 1
                env = dict(os.environ); env.update(env_extra); env[DIRVAR[case]] = d
                p = subprocess.run([PY, script], cwd=ROOT, env=env, capture_output=True, text=True)
                res[side] = (p.returncode, p.stdout, p.stderr, d)
                print(f"[{label}/{side}] {os.path.relpath(script, W) if script.startswith(W) else new_s} inputs={n} rc={p.returncode}", flush=True)
                if p.returncode: print(p.stderr[-3000:]); ok = False
            (ro, so, eo, do), (rn, sn, en, dn) = res["old"], res["new"]
            if so != sn:
                ok = False; print(f"[{label}] STDOUT DIFFERS");
                import difflib; print("".join(list(difflib.unified_diff(so.splitlines(1), sn.splitlines(1), "old", "new"))[:40]))
            else:
                print(f"[{label}] stdout identical ({len(so.splitlines())} lines)")
            for o in outs:
                fo, fn, fp = os.path.join(do, o), os.path.join(dn, o), os.path.join(ROOT, pub, o)
                if not (os.path.isfile(fo) and os.path.isfile(fn)):
                    ok = False; print(f"[{label}] {o}: missing (old {os.path.isfile(fo)}, new {os.path.isfile(fn)})"); continue
                same = filecmp.cmp(fo, fn, shallow=False)
                pubsame = os.path.isfile(fp) and filecmp.cmp(fo, fp, shallow=False)
                print(f"[{label}] {o}: old==new {same} (sha1 {sha(fo)[:12]} / {sha(fn)[:12]}); old==published {pubsame}"
                      + (f" (published {sha(fp)[:12]})" if os.path.isfile(fp) else " (no published file)"))
                ok = ok and same
                if not pubsame: print(f"[{label}] NOTE {o}: rerun differs from the published file (reported, not a merge failure)")
    finally:
        shutil.rmtree(W, ignore_errors=True)
    print("PARITY2_OK" if ok else "PARITY2_FAILED", flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
