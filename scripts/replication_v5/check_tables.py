#!/usr/bin/env python3
"""check_tables.py -- compare every number in each generated table with the same table in the manuscript.

For each OUT_DIR/tab_<label>.tex, find \\label{tab:<label>} in main.tex / appendix.tex (with
\\input{tables/...} files inlined and comments stripped), take the first tabular after the label,
and compare the numbers in order (sign and printed digits must agree exactly). The table note
(everything after \\end{tabular} up to \\end{table}) is compared the same way and reported separately.

Env: TABLES_OUT (default ./out), OVERLEAF (default /home/claude/rb/overleaf).
Prints MATCH / DIFF per table; for DIFF, every differing cell with its row.
"""
import difflib
import os
import re
import sys

OUT_DIR = os.environ.get("TABLES_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
OVERLEAF = os.environ.get("OVERLEAF", "/home/claude/rb/overleaf")
EXPECTED_DIFF = {"civilian_remap": "swapped to M full-source (author decision)",
                 "onset_subsets": "swapped to M full-source (author decision)"}


def strip_comments(txt):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in txt.split("\n"))


def read_tex(path):
    with open(path) as f:
        txt = strip_comments(f.read())

    def inline(m):
        name = m.group(1)
        p = os.path.join(OVERLEAF, name if name.endswith(".tex") else name + ".tex")
        if not os.path.isfile(p):
            return m.group(0)
        with open(p) as g:
            return strip_comments(g.read())
    return re.sub(r"\\input\{(tables/[^}]+)\}", inline, txt)


NUM = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")


def tokens(txt):
    """[(number string, line)] in order; signs from $-$ / $+$ are attached to the number."""
    out = []
    for ln in txt.split("\n"):
        s = ln.replace("$-$", "-").replace("$+$", "+").replace("$", "")
        s = re.sub(r"\\\\\[\d+pt\]", r"\\\\", s)                 # row spacing \\[2pt]
        s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", s)      # column spans
        s = re.sub(r"\\(begin|end)\{tabular\}\{[^}]*\}", "", s)
        s = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}", "", s)
        for k, m in enumerate(NUM.finditer(s)):
            out.append((m.group(0), "number %d in row: %s" % (k + 1, ln.strip())))
    return out


def split_table(txt, start):
    """tabular body and note, beginning at position start."""
    b = txt.index(r"\begin{tabular}", start)
    e = txt.index(r"\end{tabular}", b)
    tab = txt[b:e]
    end_tab = txt.find(r"\end{table}", e)
    note = txt[e + len(r"\end{tabular}"): end_tab if end_tab >= 0 else len(txt)]
    return tab, note


def compare(a, b):
    """list of human-readable differences between token lists a (manuscript) and b (generated)."""
    diffs = []
    if len(a) == len(b):   # same layout: compare cell by cell
        for (ta, la), (tb, _) in zip(a, b):
            if ta != tb:
                diffs.append("    manuscript %-10s generated %-10s | %s" % (ta, tb, la[:120]))
        return diffs
    sm = difflib.SequenceMatcher(a=[t for t, _ in a], b=[t for t, _ in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                diffs.append("    manuscript %-10s generated %-10s | %s" % (a[i1 + k][0], b[j1 + k][0], a[i1 + k][1][:120]))
        else:
            diffs.append("    %s manuscript %s -> generated %s | rows: %s || %s" % (
                op, [t for t, _ in a[i1:i2]], [t for t, _ in b[j1:j2]],
                a[i1][1][:60] if i2 > i1 else "-", b[j1][1][:60] if j2 > j1 else "-"))
    return diffs


ms = read_tex(os.path.join(OVERLEAF, "main.tex")) + "\n" + read_tex(os.path.join(OVERLEAF, "appendix.tex"))
files = sorted(f for f in os.listdir(OUT_DIR) if f.startswith("tab_") and f.endswith(".tex"))
assert files, "no generated tables in " + OUT_DIR
n_match = n_diff = 0
for f in files:
    label = f[len("tab_"):-len(".tex")]
    with open(os.path.join(OUT_DIR, f)) as g:
        gen = strip_comments(g.read())
    pos = ms.find(r"\label{tab:" + label + "}")
    if pos < 0:
        print("%-22s NOT IN MANUSCRIPT" % label)
        continue
    m_tab, m_note = split_table(ms, pos)
    g_tab, g_note = split_table(gen, 0)
    mt, gt = tokens(m_tab), tokens(g_tab)
    d_tab = compare(mt, gt)
    d_note = compare(tokens(m_note), tokens(g_note))
    tag = "MATCH" if not d_tab else "DIFF"
    n_match += tag == "MATCH"; n_diff += tag == "DIFF"
    extra = ("  [expected: %s]" % EXPECTED_DIFF[label]) if label in EXPECTED_DIFF else ""
    print("%-22s %-5s table numbers %d vs %d, %d differ; note %s%s" % (
        label, tag, len(mt), len(gt), len(d_tab), "same numbers" if not d_note else "DIFF", extra))
    for d in d_tab:
        print(d)
    for d in d_note:
        print("    [note]" + d[3:])
print("summary: %d MATCH, %d DIFF" % (n_match, n_diff))
sys.exit(0)
