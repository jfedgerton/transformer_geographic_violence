# make_manifest.py -- pipeline helper (author tool). Writes pipeline/manifest.csv. Run on Roar from the project root:
#   venv/bin/python scripts/replication_v5/make_manifest.py . .
"""Compute the set of repo files the replication_v5 drivers need: drivers -> launchers -> scripts -> local imports/sources.
Run: python3 closure.py <conflict_forecast root> <pkg root>   (pkg root holds replication_v5/ and scripts/replication_v5/)"""
import os, re, sys, hashlib, csv
ROOT, PKG = sys.argv[1], sys.argv[2]
def path_of(rel):
    for base in (PKG, ROOT):
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            return p
    return None
PAT = re.compile(r'((?:\$P|\$X|\$V|\$T|pipeline|scripts|figures_tables)/[A-Za-z0-9_./${}-]+\.(?:py|R|sbatch|sh))')
SUB = {"$P": "scripts/phase4_scaling", "$X": "scripts/crossdomain", "$V": "scripts/replication_v5"}
todo = ["pipeline/run.sh", "pipeline/tasks.sbatch"]
seen, missing, why = {}, [], {}
while todo:
    rel = todo.pop()
    if rel in seen:
        continue
    p = path_of(rel)
    if p is None:
        missing.append(rel); continue
    seen[rel] = p
    txt = open(p, errors="replace").read()
    if rel.endswith(".py"):
        txt_code = txt
    else:
        txt_code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    for m in ([] if rel.endswith(".py") else PAT.findall(txt_code)):   # .py files: follow imports only (path strings in docstrings are provenance notes)
        r = m
        for k, v in SUB.items():
            r = r.replace(k, v)
        if "$" in r or "{" in r:
            # launcher pattern like $P/launch_fullsrc_$S.sbatch -> expand S/M/L and xd s/m
            for a, b in [("$S", "S"), ("$S", "M"), ("$S", "L"), ("$s", "m"), ("$s", "s"), ("${S}", "M")]:
                rr = r.replace(a, b)
                if "$" not in rr and "{" not in rr and path_of(rr):
                    todo.append(rr); why.setdefault(rr, rel)
            continue
        if path_of(r):
            todo.append(r); why.setdefault(r, rel)
        else:
            missing.append(r + "  (from " + rel + ")")
    d = os.path.dirname(rel)
    if rel.endswith(".py"):   # local imports: sibling modules
        for mod in re.findall(r'^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)', txt, re.M):
            for cand in (os.path.join(d, mod + ".py"),):
                if path_of(cand):
                    todo.append(cand); why.setdefault(cand, rel)
    if rel.endswith(".R"):
        for s in re.findall(r'source\(\s*(?:file\.path\([^)]*?"([^"]+\.R)"|"([^"]+\.R)")', txt):
            nm = s[0] or s[1]
            for cand in (nm, os.path.join(d, os.path.basename(nm))):
                if path_of(cand):
                    todo.append(cand); why.setdefault(cand, rel)
LOOP = ["scripts/%s.py" % x for x in "01_ingest_acled 01b_ingest_ucdp 01c_merge_sources 02_build_grid 03_aggregate_location_month 04_build_sequences".split()]  # rebuild_pipeline.sbatch loop
for rel in LOOP:
    todo.append(rel); why.setdefault(rel, "scripts/rebuild_pipeline.sbatch (loop)")
while todo:
    rel = todo.pop()
    if rel in seen: continue
    p = path_of(rel)
    if p is None: missing.append(rel); continue
    seen[rel] = p; txt = open(p, errors="replace").read(); d = os.path.dirname(rel)
    for mod in re.findall(r'^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)', txt, re.M):
        cand = os.path.join(d, mod + ".py")
        if path_of(cand): todo.append(cand); why.setdefault(cand, rel)
for extra in ["scripts/phase4_scaling/Rprofile_cairo", "env/requirements_frozen_2026-10-19.txt", "env/versions_2026-10-19.txt",
              "scripts/00_master.py", "scripts/crossdomain/fetch_sim_dores.sh",
              "scripts/replication_v5/check_tables.py", "scripts/replication_v5/make_manifest.py", "pipeline/README.md",
              "pipeline/tests/parity_phase1.py", "pipeline/tests/v5_baseline_blobs.txt"]:
    if path_of(extra):
        seen[extra] = path_of(extra); why.setdefault(extra, "extra (config, provenance, tools)")
    else:
        missing.append(extra)
w = csv.writer(open(os.path.join(PKG, "pipeline", "manifest.csv"), "w", newline=""))
w.writerow(["path", "md5", "bytes", "role", "first_referenced_by"])
for rel in sorted(seen):
    b = open(seen[rel], "rb").read()
    role = ("driver" if rel.startswith("pipeline/") else "new_v5" if rel.startswith("scripts/replication_v5/")
            else "config" if rel.startswith("env/") or rel.endswith("Rprofile_cairo")
            else "launcher" if rel.endswith((".sbatch", ".sh")) else "script")
    w.writerow([rel, hashlib.md5(b).hexdigest(), len(b), role, why.get(rel, "")])
print(len(seen), "files;", len(set(missing)), "missing:")
for m in sorted(set(missing)):
    print("  MISSING", m)
