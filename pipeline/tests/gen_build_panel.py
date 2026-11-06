# gen_build_panel.py -- builds scripts/build_panel.py (Phase 2h) from the eleven stage-01 scripts, then checks with the
# Python AST that every step block is the original script unchanged apart from the documented line. Usage:
#   python3 gen_build_panel.py <project_root>       (writes <root>/scripts/build_panel.py; refuses to overwrite)
import ast, os, sys, textwrap

ROOT = sys.argv[1]
S = os.path.join(ROOT, "scripts")
OUT = os.path.join(S, "build_panel.py")
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"

# step name -> old script, in pipeline order
STEPS = [("acled", "01_ingest_acled.py"), ("ucdp", "01b_ingest_ucdp.py"), ("merge", "01c_merge_sources.py"),
         ("grid", "02_build_grid.py"), ("aggregate", "03_aggregate_location_month.py"), ("sequences", "04_build_sequences.py"),
         ("baseline", "05_build_baseline_features.py"), ("super", "build_sequences_super.py"),
         ("prio", "build_priogrid_static.py"), ("pgm", "build_pgm_fatalities_target.py")]
IMPORT_OLD = 'master = import_module("00_master")\n'
IMPORT_NEW = 'master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)\n'

master = open(os.path.join(S, "00_master.py")).read()
cut = master.index('# ==========================================================================\n# MAIN: run full pipeline')
master_top = master[:cut].rstrip("\n") + "\n"
assert master_top.count('if __name__ == "__main__":') == 0


def split_docstring(src):
    """(docstring text or None, rest of the source after the docstring statement)"""
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        end = tree.body[0].end_lineno
        lines = src.splitlines(keepends=True)
        return tree.body[0].value.value, "".join(lines[end:])
    return None, src


out = ['''"""build_panel.py (consolidated 2026-11-05) -- stage 01: the PRIO-GRID cell-month panel, in one file.
Replaces eleven scripts (tag v5-as-run): 00_master.py (its configuration, verbatim below; its old full-pipeline
__main__ block is dropped) and the ten steps
'''] + [f"  {name:<10} was scripts/{fn}\n" for name, fn in STEPS] + ['''Run one step per call, from the project root:  python scripts/build_panel.py <step>
Each call is a fresh Python process that runs exactly one step, as the old one-script-per-step calls did. Every step block
is the old script unchanged, indented under its `if`, with two edits only: its docstring is kept as a comment, and its
config import reads this file (see IMPORT lines). pipeline/tasks.sbatch (rebuild, featbuild) and pipeline/run.sh
(data_prio, data_pgm) call the steps in the old order with the old environment (PHASE_MODE, REPR_SEQ_SUFFIX).
Parity: pipeline/tests/parity_phase2.py panel_build."""
# ======================= 00_master.py (configuration; verbatim apart from the dropped __main__ block) =======================
''', master_top, '''
# ======================= step selector =======================
STEPS = [''' + ", ".join(f'"{n}"' for n, _ in STEPS) + ''']
if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
    sys.exit("usage: python scripts/build_panel.py <step>, step one of: " + " ".join(STEPS))
STEP = sys.argv[1]
''']
checks = []
for name, fn in STEPS:
    src = open(os.path.join(S, fn)).read()
    doc, body = split_docstring(src)
    if fn != "build_pgm_fatalities_target.py":
        assert body.count(IMPORT_OLD) == 1, fn
        body = body.replace(IMPORT_OLD, IMPORT_NEW)
    else:
        assert IMPORT_OLD not in body
    if not body.endswith("\n"): body += "\n"
    head = f"\n# ======================= step {name}: was scripts/{fn} =======================\n"
    if doc is not None:
        head += "".join("# " + l if l.strip() else "#\n" for l in doc.strip("\n").splitlines(True)) + "\n"
    blk = textwrap.indent(body, "    ", lambda l: True)
    blk = "".join(l if l.strip() else "\n" for l in blk.splitlines(True))   # no trailing spaces on blank lines
    out.append(head + f'if STEP == "{name}":\n' + blk)
    checks.append((name, fn, src))
text = "".join(out)

# ---- AST check (strengthened after the GPT review, 2026-11-06) ----
# check(text) passes only if the generated module is exactly (a) the 00_master statements, (b) the three selector
# statements, (c) ten `if STEP == "<step>":` blocks in pipeline order with no else, and nothing else; and each block equals
# the old script statement by statement, where the one changed statement must be exactly the approved alias.
OLD_IMPORT = ast.dump(ast.parse(IMPORT_OLD).body[0])
NEW_ALIAS = ast.dump(ast.parse("master = sys.modules[__name__]\n").body[0])
SELECTOR = ast.parse("STEPS = [" + ", ".join(f'"{n}"' for n, _ in STEPS) + "]\n"
                     "if len(sys.argv) != 2 or sys.argv[1] not in STEPS:\n"
                     "    sys.exit(\"usage: python scripts/build_panel.py <step>, step one of: \" + \" \".join(STEPS))\n"
                     "STEP = sys.argv[1]\n").body
MASTER = ast.parse(master_top).body


def check(text, verbose=False):
    body = ast.parse(text).body
    assert body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        and isinstance(body[0].value.value, str), "module docstring missing"
    body = body[1:]   # the module docstring has no runtime effect beyond __doc__, which no step reads
    nm, ns = len(MASTER), len(SELECTOR)
    assert len(body) == nm + ns + len(STEPS), f"module has {len(body)} top-level statements, expected {nm + ns + len(STEPS)}"
    assert [ast.dump(x) for x in body[:nm]] == [ast.dump(x) for x in MASTER], "00_master part differs"
    assert [ast.dump(x) for x in body[nm:nm + ns]] == [ast.dump(x) for x in SELECTOR], "selector differs"
    for k, (name, fn, src) in enumerate(checks):
        blk = body[nm + ns + k]
        want_test = ast.dump(ast.parse(f'STEP == "{name}"', mode="eval").body)
        assert isinstance(blk, ast.If) and ast.dump(blk.test) == want_test and not blk.orelse, f"block {k} is not if STEP == '{name}'"
        old = ast.parse(src).body
        if old and isinstance(old[0], ast.Expr) and isinstance(old[0].value, ast.Constant) and isinstance(old[0].value.value, str):
            old = old[1:]
        od = [ast.dump(x) for x in old]; nd = [ast.dump(x) for x in blk.body]
        assert len(od) == len(nd), f"step {name}: {len(nd)} statements, old script has {len(od)}"
        j = [i for i, x in enumerate(od) if x == OLD_IMPORT]
        if fn == "build_pgm_fatalities_target.py":
            assert j == [] and od == nd, f"step {name}: AST differs from {fn}"
        else:
            assert len(j) == 1, f"{fn}: old import found {len(j)} times"
            assert nd[j[0]] == NEW_ALIAS, f"step {name}: replacement at statement {j[0]} is not exactly master = sys.modules[__name__]"
            assert od[:j[0]] + od[j[0] + 1:] == nd[:j[0]] + nd[j[0] + 1:], f"step {name}: AST differs from {fn}"
        if verbose: print(f"AST_IDENTICAL {name} ({fn}, {len(od)} statements; " + ("no changed statement)" if fn == "build_pgm_fatalities_target.py" else "changed statement is the approved alias)"))
    if verbose: print(f"AST_IDENTICAL config (00_master.py minus __main__, {nm} statements); selector exact; 10 guards in order; nothing else")


check(text, verbose=True)
# negative controls: each mutation must make check() fail
MUT = [("wrong alias", lambda t: t.replace("master = sys.modules[__name__]", 'master = sys.modules["__name__"]', 1)),
       ("code outside the guards", lambda t: t + "print('extra')\n"),
       ("two steps swapped in order", lambda t: t.replace('if STEP == "acled":', 'if STEP == "TMP":', 1).replace('if STEP == "ucdp":', 'if STEP == "acled":', 1).replace('if STEP == "TMP":', 'if STEP == "ucdp":', 1)),
       ("selector loosened", lambda t: t.replace("if len(sys.argv) != 2 or", "if", 1)),
       ("else branch added to a guard", lambda t: t.replace('if STEP == "pgm":\n', 'if STEP == "pgm":\n    pass\nelse:\n    pass\nif STEP == "pgm":\n', 1))]
for label, f in MUT:
    m = f(text); assert m != text, f"mutation '{label}' did not change the text"
    try:
        check(m); raise SystemExit(f"NEGATIVE CONTROL NOT CAUGHT: {label}")
    except AssertionError as e:
        print(f"NEGATIVE_CONTROL_CAUGHT {label}: {str(e)[:90]}")
open(OUT, "w").write(text)
print("WROTE", OUT)
