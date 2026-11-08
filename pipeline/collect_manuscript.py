"""collect_manuscript.py -- copy the compiled manuscript (Overleaf export) into manuscript/ and verify every
figure and table against the pipeline output that produced it. Writes manuscript/MAP.csv.

Usage (repo root):  python pipeline/collect_manuscript.py <overleaf_export.zip> [--check]
  --check   only verify manuscript/ against the pipeline outputs (no copy)
Fails loudly if a figure differs from its pipeline file (PDF creation dates ignored) or a table's numbers differ.
TikZ figures and the design-choices table are static LaTeX (no data); they are copied, not checked.
"""
import sys, os, re, csv, hashlib, zipfile, shutil, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MS = os.path.join(ROOT, "manuscript")
V5 = "figures_tables/v5_20261108"
T5 = "figures_tables/tables_v5_20261108b"

# manuscript file -> (pipeline file, producing script)
FIG = {
 "figures/fig_loro_gains.pdf": (V5 + "/fig3_loro_gains/fig_loro_gains.pdf", "scripts/replication_v5/figures.R fig3"),
 "figures/fig_cell_alerts_maps_nested_M_fullsrc.png": (V5 + "/fig4_cell_maps/fig_cell_alerts_maps_nested_M_fullsrc.png", "scripts/replication_v5/figures.R fig4"),
 "figures/fig_finetune_gains.pdf": (V5 + "/fig5_finetune_gains/fig_finetune_gains.pdf", "scripts/replication_v5/figures.R fig5"),
 "figures/fig_placebo_gradient_ab_M.pdf": (V5 + "/fig6_brazil/fig_placebo_gradient_ab_M.pdf", "scripts/replication_v5/figures.R fig6"),
 "figures/fig_random_encoder.png": (V5 + "/fig9_random_encoder/fig_random_encoder.png", "scripts/replication_v5/figures.R fig9"),
 "figures/fig_boot_shares_M.png": (V5 + "/fig10_share_curve/fig_boot_shares.png", "scripts/replication_v5/figures.R fig10"),
 "figures/fig_ukraine_f0025_full_1_invasion.pdf": ("outputs/ukraine_maps_M_f0025/fig_ukraine_f0.0025_M_fullsrc_full_1_invasion.pdf", "figures.R --one ukr_map (stage 08)"),
 "figures/fig_ukraine_f0025_full_2_counteroffensives.pdf": ("outputs/ukraine_maps_M_f0025/fig_ukraine_f0.0025_M_fullsrc_full_2_counteroffensives.pdf", "figures.R --one ukr_map (stage 08)"),
 "figures/fig_ukraine_f0025_full_3_southern_offensive.pdf": ("outputs/ukraine_maps_M_f0025/fig_ukraine_f0.0025_M_fullsrc_full_3_southern_offensive.pdf", "figures.R --one ukr_map (stage 08)"),
 "figures/fig_ukraine_f0025_full_4_2024_2025.pdf": ("outputs/ukraine_maps_M_f0025/fig_ukraine_f0.0025_M_fullsrc_full_4_2024_2025.pdf", "figures.R --one ukr_map (stage 08)"),
 "figures/fig_ukraine_alert_gain_full.pdf": ("outputs/ukraine_alert_gain_M_full/fig_ukraine_alert_gain_M_fullsrc_full.pdf", "figures.R --one ukr_gain (stage 08)"),
 "figures/fig_brazil_deaths_map.pdf": ("figures_tables/fig_brazil_deaths_map.pdf", "scripts/crossdomain/fig_brazil_deaths_map.R (br_desc)"),
 "figures/fig_placebo_gradient.pdf": (V5 + "/fig16_brazil_random/fig_placebo_gradient_M.pdf", "scripts/replication_v5/figures.R fig16"),
 "figures/fig_boot_share_regions_M.png": (V5 + "/fig17_share_regions/fig_boot_share_regions.png", "scripts/replication_v5/figures.R fig17"),
 "figures/fig_boot_persistence_M.png": (V5 + "/fig18_persistence_gap/fig_boot_persistence.png", "scripts/replication_v5/figures.R fig18"),
 "figures/fig_cell_alerts_countries_wide_M_fullsrc.png": ("outputs/cell_alerts_fig_M_fullsrc/fig_cell_alerts_incidence_M_fullsrc_print_opt4b_countries_wide.png", "figures.R --one fig19 (stage 08)"),
}
# \input tables in the manuscript -> pipeline table
TIN = {
 "tables/table3_nested_embbase.tex": T5 + "/tab_nested_embbase.tex",
 "tables/tab_random_control_temporal.tex": T5 + "/tab_random_temporal.tex",
 "tables/tab_random_control_pooled.tex": T5 + "/tab_random_pooled.tex",
 "tables/table2_pretrained_1pct.tex": T5 + "/tab_pretrained_1pct.tex",
 "tables/tab_boot_main.tex": T5 + "/tab_boot_main.tex",
 "tables/tab_boot_shares.tex": T5 + "/tab_boot_shares.tex",
 "tables/tab_brazil_summary.tex": "figures_tables/tab_brazil_summary.tex",
}
# tables typed inside main.tex -> pipeline table (label tab:X <- tab_X.tex)
INLINE = ["ukraine_ft", "domains", "brazil", "civilian_remap", "onset_subsets", "sizes_frozen", "sizes_finetune_full",
          "sizes_finetune", "sizes_brazil", "share_months", "finetune_grid_full", "finetune_grid"]
STATIC = ["figures/fig_architecture.tex", "figures/fig_loro_design.tex", "figures/fig_pipeline.tex",
          "tables/table_design_choices.tex", "bibliography.bib", "chicago-ff.sty", "chicago-ff.bst", "quiver.sty"]

def pdfstrip(b):
    b = re.sub(rb"/(CreationDate|ModDate)\s*\([^)]*\)", b"", b)
    return re.sub(rb"/ID\s*\[[^\]]*\]", b"", b)
def sha(b): return hashlib.sha256(b).hexdigest()[:16]
def nums(s):
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", s, re.S)
    s = m.group(0) if m else s
    s = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))
    s = s.replace("$-$", "-").replace("{,}", "").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return re.findall(r"[-+]?\.?\d+(?:\.\d+)?", s)

def main():
    check_only = "--check" in sys.argv
    src = MS
    if not check_only:
        z = [a for a in sys.argv[1:] if a.endswith(".zip")]
        assert len(z) == 1, "give the Overleaf export zip"
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(z[0]).extractall(tmp)
        src = tmp
    rows, bad = [], []
    main_tex = open(os.path.join(src, "main.tex")).read()
    live = "\n".join(l for l in main_tex.split("\n") if not l.lstrip().startswith("%"))
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", live))
    assert used == set(FIG), "manuscript figures and FIG map differ: %s" % sorted(used ^ set(FIG))
    used_in = set(t + ("" if t.endswith(".tex") else ".tex") for t in re.findall(r"\\input\{(tables/[^}]*)\}", live))
    assert used_in == set(TIN) | {"tables/table_design_choices.tex"}, sorted(used_in ^ (set(TIN) | {"tables/table_design_choices.tex"}))
    for ms, (pp, scr) in FIG.items():
        a, b = open(os.path.join(src, ms), "rb").read(), open(os.path.join(ROOT, pp), "rb").read()
        ok = a == b or (ms.endswith(".pdf") and pdfstrip(a) == pdfstrip(b))
        rows.append([ms, pp, scr, sha(b), "identical" if a == b else ("identical_except_pdf_dates" if ok else "DIFFERENT")])
        if not ok: bad.append(ms)
    for ms, pp in TIN.items():
        ok = nums(open(os.path.join(src, ms)).read()) == nums(open(os.path.join(ROOT, pp)).read())
        rows.append([ms, pp, "scripts/replication_v5/make_tables.py" if "tables_v5" in pp else "scripts/crossdomain/brazil_descriptives.py (br_desc)", "", "numbers_identical" if ok else "DIFFERENT"])
        if not ok: bad.append(ms)
    for lab in INLINE:
        m = re.search(r"\\label\{tab:%s\}" % lab, live); assert m, lab
        st = live.rfind("\\begin{table", 0, m.start()); en = live.find("\\end{table", m.end())
        pp = T5 + "/tab_%s.tex" % lab
        ok = nums(live[st:en]) == nums(open(os.path.join(ROOT, pp)).read())
        rows.append(["main.tex tab:" + lab, pp, "scripts/replication_v5/make_tables.py", "", "numbers_identical" if ok else "DIFFERENT"])
        if not ok: bad.append("tab:" + lab)
    for f in STATIC:
        assert os.path.exists(os.path.join(src, f)), f
        rows.append([f, "", "static LaTeX (no data)", "", "copied"])
    if bad:
        raise SystemExit("MISMATCH, manuscript not collected: " + ", ".join(bad))
    if not check_only:
        if os.path.exists(MS): shutil.rmtree(MS)
        for f in ["main.tex"] + STATIC + list(FIG) + list(TIN):
            os.makedirs(os.path.dirname(os.path.join(MS, f)) or MS, exist_ok=True)
            shutil.copy2(os.path.join(src, f), os.path.join(MS, f))
    with open(os.path.join(MS, "MAP.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["manuscript_file", "pipeline_file", "produced_by", "sha256_16", "check"]); w.writerows(rows)
    print("figures %d, input tables %d, inline tables %d, static %d; all checks passed" % (len(FIG), len(TIN), len(INLINE), len(STATIC)))
    print("COLLECT_OK")

if __name__ == "__main__":
    main()
