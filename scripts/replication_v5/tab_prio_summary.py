"""tab_prio_summary.py -- appendix table: summary statistics of the 34 PRIO-GRID static covariates (raw, unstandardized).

Cells: every grid_id in data_intermediate/baseline_features_super.parquet (the analysis panel).
Values: data_raw/priogrid/static/<name>.json, read exactly as build_panel.py step prio reads them (last value per gid wins).
Output: figures_tables/tab_prio_summary.tex (label tab:prio_summary) and figures_tables/tab_prio_summary.csv.
Usage (project root): venv/bin/python scripts/replication_v5/tab_prio_summary.py
"""
import os, glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq

RAW = "data_raw/priogrid/static"
PANEL = "data_intermediate/baseline_features_super.parquet"
OUT_TEX = "figures_tables/tab_prio_summary.tex"
OUT_CSV = "figures_tables/tab_prio_summary.csv"

# ---- 1. analysis cells (read grid_id only, one row group at a time) ----
pf = pq.ParquetFile(PANEL)
cells = set()
for i in range(pf.num_row_groups):
    cells.update(pf.read_row_group(i, columns=["grid_id"]).column(0).to_pylist())
cells = np.array(sorted(int(c) for c in cells))
assert len(cells) > 20000, len(cells)
print(f"[prio] analysis cells: {len(cells):,}")

# ---- 2. raw covariates, same reading rule as build_panel.py step prio ----
files = sorted(glob.glob(os.path.join(RAW, "*.json")))
assert len(files) == 34, f"expected 34 static files, found {len(files)}"
raw = {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    vals = {int(c["gid"]): float(c["value"]) for c in json.load(open(f))["cells"] if c.get("value") is not None}
    raw[name] = pd.Series(vals, dtype="float64").reindex(cells)

# ---- 3. labels and groups (hard-coded; every file must be listed exactly once) ----
GROUPS = [
    ("Land cover (percent of cell)", [("agri_gc", "Agricultural"), ("aquaveg_gc", "Aquatic vegetation"), ("barren_gc", "Barren"),
                                    ("forest_gc", "Forest"), ("herb_gc", "Herbaceous"), ("shrub_gc", "Shrubland"),
                                    ("urban_gc", "Urban"), ("water_gc", "Water")]),
    ("Terrain", [("mountains_mean", "Mountainous share"), ("landarea", "Land area")]),
    ("Child malnutrition", [("cmr_min", "Minimum"), ("cmr_mean", "Mean"), ("cmr_max", "Maximum"), ("cmr_sd", "Standard deviation")]),
    ("Infant mortality", [("imr_min", "Minimum"), ("imr_mean", "Mean"), ("imr_max", "Maximum"), ("imr_sd", "Standard deviation")]),
    ("Travel time to nearest major city", [("ttime_min", "Minimum"), ("ttime_mean", "Mean"), ("ttime_max", "Maximum"), ("ttime_sd", "Standard deviation")]),
    ("Natural resources", [("diamprim_s", "Primary diamonds"), ("diamsec_s", "Secondary diamonds"), ("gem_s", "Gems"),
                           ("goldplacer_s", "Placer gold"), ("goldsurface_s", "Surface gold"), ("goldvein_s", "Vein gold"),
                           ("petroleum_s", "Petroleum")]),
    ("Agriculture and climate", [("harvarea", "Harvested area"), ("maincrop", "Main crop (categorical code)"),
                                 ("growstart", "Growing season start (month)"), ("growend", "Growing season end (month)"),
                                 ("rainseas", "Rainy season start (month)")]),
]
listed = [v for _, vs in GROUPS for v, _ in vs]
assert len(listed) == 34 and len(set(listed)) == 34 and set(listed) == set(raw), set(raw) ^ set(listed)

def fmt(x):
    if pd.isna(x): return "--"
    a = abs(x)
    if a >= 1000: return f"{x:,.0f}"
    if a >= 10: return f"{x:.1f}"
    return f"{x:.3f}"

rows, lines = [], []
for g, vs in GROUPS:
    if g == "Natural resources":   # constant after build_panel.py step prio (source lists only deposit cells); carries no information
        for v, _ in vs:
            assert raw[v].dropna().nunique() <= 1, v
        continue
    lines.append(r"\multicolumn{7}{l}{\textit{" + g + r"}} \\")
    for v, lab in vs:
        s = raw[v]; obs = s.dropna()
        miss = 100 * s.isna().mean()
        if v.endswith("_s"):   # deposit indicators: the source lists only cells with a deposit (value 1); absent cells are missing
            assert set(obs.unique()) <= {1.0}, v
            stats = dict(n=len(obs), miss=np.nan, mean=len(obs) / len(cells), sd=np.nan, min=np.nan, max=np.nan, ncat=np.nan)
        elif v == "maincrop":
            stats = dict(n=len(obs), miss=miss, mean=np.nan, sd=np.nan, min=np.nan, max=np.nan, ncat=obs.nunique())
        else:
            stats = dict(n=len(obs), miss=miss, mean=obs.mean(), sd=obs.std(), min=obs.min(), max=obs.max(), ncat=np.nan)
        rows.append(dict(group=g, variable=v, label=lab, **stats))
        extra = f" ({stats['ncat']:.0f} categories)" if v == "maincrop" else ""
        lines.append(f"\\quad {lab}{extra} & \\texttt{{{v.replace('_', chr(92) + '_')}}} & {'--' if pd.isna(stats['miss']) else format(stats['miss'], '.1f')} & {fmt(stats['mean'])} & {fmt(stats['sd'])} & {fmt(stats['min'])} & {fmt(stats['max'])} \\\\")
df = pd.DataFrame(rows)
assert len(df) == 27
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
tex = r"""\begin{table}[!htbp]
\centering
\caption{PRIO-GRID Covariates, Summary Statistics}
\label{tab:prio_summary}
\small
\begin{tabular}{llrrrrr}
\hline
Covariate & PRIO-GRID name & Missing (\%) & Mean & SD & Min & Max \\
\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}
\begin{minipage}{0.95\textwidth}\footnotesize\vspace{0.5ex}
Notes. Raw values before standardization, over the """ + f"{len(cells):,}" + r""" grid cells in the analysis panel (each cell counted once). Missing values are set to the covariate mean (zero after standardization) before the covariates enter the models. Main crop is a categorical code; the number of distinct codes is shown instead of moments. The source also provides seven natural-resource indicators (primary and secondary diamonds, gems, three types of gold, and petroleum) that list only cells with a deposit. After standardization and mean imputation they take a single value in every cell, so they carry no information in the models and are omitted from this table and from the covariate count. Variable definitions and units follow the PRIO-GRID codebook \citep{tollefsen2012priogrid}.
\end{minipage}
\end{table}
"""
open(OUT_TEX, "w").write(tex)
print(df[["variable", "n", "miss", "mean", "sd", "min", "max"]].to_string())
print(f"[done] {OUT_TEX} {OUT_CSV}")
