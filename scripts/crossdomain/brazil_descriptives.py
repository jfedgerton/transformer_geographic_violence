"""brazil_descriptives.py (2026-11-06) -- descriptive statistics for the four Brazil SIM outcomes (Appendix app:vocab).

Reads the four Brazil cell-month panels built by brazil_panel.py (stage 08) and writes
  outputs/crossdomain/brazil_cell_totals.csv   one row per (outcome, grid_id): lon, lat, total deaths
                                               (outcome = deaths_next on every origin row 2001-01 .. 2024-11,
                                               i.e. deaths from February 2001 to December 2024)
  outputs/crossdomain/brazil_summary.csv       one row per (outcome, period) with the summary statistics
  figures_tables/tab_brazil_summary.tex        LaTeX table (label tab:brazil_summary)
The map (figures_tables/fig_brazil_deaths_map.pdf/.png) is drawn from brazil_cell_totals.csv by
fig_brazil_deaths_map.R, run in the same pipeline task.

Periods follow xd_grid_brazil_v6.py: training origins 2001-01 .. 2020-12, test origins 2022-01 .. 2024-11.
The outcome is deaths_next (deaths in month t+1), the count the forecasters predict.
CPU only. Reads three columns per panel. Fails loudly on any unexpected input.
"""
import os
import numpy as np
import pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OUT_CSV_DIR = f"{R}/outputs/crossdomain"
OUT_FT_DIR = f"{R}/figures_tables"
PANELS = [("Homicide", f"{R}/data_intermediate/xd_brazil_panel.parquet"),
          ("Traffic", f"{R}/data_intermediate/xd_brazil_traffic_panel.parquet"),
          ("Suicide", f"{R}/data_intermediate/xd_brazil_suicide_panel.parquet"),
          ("Drowning", f"{R}/data_intermediate/xd_brazil_drowning_panel.parquet")]
PERIODS = [("Training", "2001-01", "2020-12"),
           ("Test", "2022-01", "2024-11")]
ALL_FIRST, ALL_LAST = "2001-01", "2024-11"
N_CELLS_EXPECTED = 1478

os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_FT_DIR, exist_ok=True)

# ---- load the four panels (three columns each) ----
frames = {}
for name, path in PANELS:
    assert os.path.exists(path), f"missing panel {path}"
    d = pd.read_parquet(path, columns=["grid_id", "year_month", "deaths_next"])
    d["year_month"] = d["year_month"].astype(str)
    assert d["grid_id"].nunique() == N_CELLS_EXPECTED, f"{name}: {d['grid_id'].nunique()} cells, expected {N_CELLS_EXPECTED}"
    assert not d.duplicated(["grid_id", "year_month"]).any(), f"{name}: duplicate cell-months"
    print(f"[load] {name}: rows={len(d):,} cells={d['grid_id'].nunique()} months={d['year_month'].min()}..{d['year_month'].max()}", flush=True)
    frames[name] = d

# the four panels share the same cells and months (same construction)
ref = frames["Homicide"][["grid_id", "year_month"]].sort_values(["grid_id", "year_month"]).reset_index(drop=True)
for name in frames:
    other = frames[name][["grid_id", "year_month"]].sort_values(["grid_id", "year_month"]).reset_index(drop=True)
    assert ref.equals(other), f"{name}: cell-month grid differs from homicide"

# ---- cell totals for the map ----
rows = []
for name, d in frames.items():
    w = d[(d["year_month"] >= ALL_FIRST) & (d["year_month"] <= ALL_LAST)]
    assert w["deaths_next"].notna().all(), f"{name}: missing deaths_next inside {ALL_FIRST}..{ALL_LAST}"
    assert (w["deaths_next"] >= 0).all(), f"{name}: negative deaths"
    tot = w.groupby("grid_id", as_index=False)["deaths_next"].sum().rename(columns={"deaths_next": "deaths"})
    tot["outcome"] = name
    rows.append(tot)
cells = pd.concat(rows, ignore_index=True)
# PRIO-GRID ids: gid = row * 720 + col + 1, row 0 at 90S, col 0 at 180W, 0.5-degree cells; centroid coordinates
g0 = cells["grid_id"].astype(np.int64) - 1
cells["lon"] = -180.0 + 0.5 * (g0 % 720) + 0.25
cells["lat"] = -90.0 + 0.5 * (g0 // 720) + 0.25
assert cells["lon"].between(-75, -28).all() and cells["lat"].between(-35, 6).all(), "cell centroids fall outside Brazil's bounding box"
cells = cells[["outcome", "grid_id", "lon", "lat", "deaths"]]
cells.to_csv(f"{OUT_CSV_DIR}/brazil_cell_totals.csv", index=False)
print(f"[write] brazil_cell_totals.csv rows={len(cells)}", flush=True)

# ---- summary statistics by outcome and period ----
srows = []
for name, d in frames.items():
    for per, a, b in PERIODS:
        w = d[(d["year_month"] >= a) & (d["year_month"] <= b)]["deaths_next"]
        assert w.notna().all(), f"{name} {per}: missing deaths_next"
        n_months = d[(d["year_month"] >= a) & (d["year_month"] <= b)]["year_month"].nunique()
        srows.append(dict(outcome=name, period=per, first_origin=a, last_origin=b, months=n_months,
                          cell_months=len(w), total_deaths=int(w.sum()), mean=w.mean(), sd=w.std(ddof=1),
                          pct_any=100.0 * (w > 0).mean(), max=int(w.max())))
summ = pd.DataFrame(srows)
assert (summ["cell_months"] == summ["months"] * N_CELLS_EXPECTED).all(), "cell-month counts do not equal cells x months"
summ.to_csv(f"{OUT_CSV_DIR}/brazil_summary.csv", index=False)
print(summ.to_string(index=False), flush=True)

# ---- LaTeX table ----
def f_int(x):
    return f"{int(x):,}"

lines = []
lines.append("% Generated by scripts/crossdomain/brazil_descriptives.py from outputs/crossdomain/brazil_summary.csv. Do not edit by hand.")
lines.append("\\begin{table}[!htbp]")
lines.append("\\centering")
lines.append("\\caption{Summary Statistics for the Four Brazilian Mortality Outcomes}")
lines.append("\\label{tab:brazil_summary}")
lines.append("\\small")
lines.append("\\begin{tabular}{llrrrrr}")
lines.append("\\hline")
lines.append("Outcome & Period & Total deaths & Mean & SD & Any death (\\%) & Max \\\\")
lines.append("\\hline")
for name, _ in PANELS:
    for per, a, b in PERIODS:
        r = summ[(summ["outcome"] == name) & (summ["period"] == per)].iloc[0]
        first = name if per == "Training" else ""
        lines.append(f"{first} & {per} & {f_int(r['total_deaths'])} & {r['mean']:.2f} & {r['sd']:.2f} & {r['pct_any']:.1f} & {f_int(r['max'])} \\\\")
    lines.append("\\hline")
n_train = int(summ[summ["period"] == "Training"]["cell_months"].iloc[0])
n_test = int(summ[summ["period"] == "Test"]["cell_months"].iloc[0])
lines.append("\\end{tabular}")
lines.append("\\caption*{\\footnotesize \\textit{Note:} Statistics are for the outcome the forecasters predict, deaths in month $t+1$, "
             f"over {N_CELLS_EXPECTED:,} PRIO-GRID cells. The training period covers forecast origins from January 2001 to December 2020 "
             f"({n_train:,} cell-months) and the test period covers origins from January 2022 to November 2024 ({n_test:,} cell-months). "
             "Mean, SD, and Max are deaths per cell-month. Any death is the share of cell-months with at least one death. "
             "Source: SIM death records \\citep{sim_datasus}.}")
lines.append("\\end{table}")
with open(f"{OUT_FT_DIR}/tab_brazil_summary.tex", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("[write] figures_tables/tab_brazil_summary.tex", flush=True)
print("BRDESC_TABLE_DONE", flush=True)
