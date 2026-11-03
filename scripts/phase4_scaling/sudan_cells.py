"""sudan_cells.py -- the Sudan cell definition in one script (Phase 2 consolidation, 2026-11-03).
Replaces three scripts (tag v5-as-run) that ran in sequence from the project root:
  step 1 audit_cells_export.py  -> outputs/audit_cells_in.csv
  step 2 audit_cell_country.R   -> outputs/audit_cell_country.csv   (R + sf: Natural Earth 50 m point-in-polygon)
  step 3 sudan_cells_fix.py     -> outputs/sudan_cells_keep.csv, outputs/sudan_exclude_cells.csv
Each step's code is the original (steps 1 and 3 indented into functions; step 2's R code verbatim in R_STEP, run with
Rscript). Run from the project root; all paths are relative, as before. The Ukraine country scripts read
outputs/sudan_cells_keep.csv. Parity: pipeline/tests/parity_phase2.py sudan (old three vs this one, same inputs)."""
import os, subprocess, sys, tempfile


def step1_audit_cells_export():
    # audit_cells_export.py -- 2026-10-27, read-only. Exports one row per PRIO-GRID cell with its
    # baseline_features country label and 0.5-degree centroid, for audit_cell_country.R.
    import pandas as pd, numpy as np
    cm = pd.read_parquet('data_intermediate/baseline_features.parquet', columns=['grid_id','country'])
    c = cm.drop_duplicates('grid_id').reset_index(drop=True)
    assert c.grid_id.is_unique, 'more than one country label per cell'
    r = (c.grid_id.astype(np.int64) - 1) // 720; k = (c.grid_id.astype(np.int64) - 1) % 720
    c['lat'] = -90 + 0.5 * r + 0.25; c['lon'] = -180 + 0.5 * k + 0.25
    assert c.lat.between(-90, 90).all() and c.lon.between(-180, 180).all()
    c.to_csv('outputs/audit_cells_in.csv', index=False)
    print('cells', len(c), 'countries', c.country.nunique())


R_STEP = r'''# audit_cell_country.R -- 2026-10-27, read-only. Reads outputs/audit_cells_in.csv (audit_cells_export.py)
# and compares each cell's baseline_features country label with Natural Earth 50m borders by centroid.
# Writes outputs/audit_cell_country.csv and prints a mismatch summary. No input file is modified.
suppressPackageStartupMessages({library(sf); library(dplyr); library(readr)})
sf_use_s2(FALSE)
Z <- 'data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip'
stopifnot(file.exists(Z), file.exists('outputs/audit_cells_in.csv'))
ne <- st_read(paste0('/vsizip/', Z), quiet = TRUE)
stopifnot('ADMIN' %in% names(ne))
ne <- ne['ADMIN']; names(ne)[1] <- 'ne_country'
cells <- read_csv('outputs/audit_cells_in.csv', show_col_types = FALSE)
stopifnot(!any(duplicated(cells$grid_id)))
pts <- st_as_sf(cells, coords = c('lon','lat'), crs = 4326, remove = FALSE)
hit <- st_join(pts, ne, join = st_intersects, left = TRUE)
out <- st_drop_geometry(hit) %>% mutate(match = !is.na(ne_country) & ne_country == country)
stopifnot(nrow(out) >= nrow(cells))
write_csv(out, 'outputs/audit_cell_country.csv')
cat('cells', nrow(cells), ' match', sum(out$match), ' mismatch', sum(!out$match & !is.na(out$ne_country)), ' no NE polygon (coast/ocean)', sum(is.na(out$ne_country)), '\n\n')
cat('top disagreements (label -> Natural Earth):\n')
print(as.data.frame(head(out %>% filter(!is.na(ne_country), !match) %>% count(country, ne_country, sort = TRUE), 40)))
cat('\nmismatch by labelled country:\n')
print(as.data.frame(head(out %>% filter(!is.na(ne_country)) %>% group_by(country) %>% summarise(cells = n(), mismatch = sum(!match)) %>% filter(mismatch > 0) %>% arrange(desc(mismatch)), 40)))
'''


def step2_audit_cell_country():
    with tempfile.NamedTemporaryFile("w", suffix="_audit_cell_country.R", delete=False) as f:
        f.write(R_STEP)
    try:
        sys.stdout.flush()
        subprocess.run(["Rscript", f.name], check=True)
    finally:
        os.remove(f.name)


def step3_sudan_cells_fix():
    # sudan_cells_fix.py -- 2026-10-27. The country label in baseline_features.parquet is pre-2011 Sudan: of the
    # 473 cells labelled Sudan, 128 fall in South Sudan and 23 in neighbours (Ethiopia, Chad, Eritrea, DRC, Uganda, CAR)
    # by centroid under Natural Earth 50m. The Sudan case study is the SAF-RSF war (April 2023-), fought inside
    # post-2011 Sudan, so Sudan is defined here as the cells whose centroid lies in Natural Earth's Sudan polygon.
    # Writes outputs/sudan_cells_keep.csv (the 322 kept) and outputs/sudan_exclude_cells.csv (the 151 dropped).
    # Reads only outputs/audit_cell_country.csv; baseline_features.parquet is not modified.
    import pandas as pd
    a = pd.read_csv('outputs/audit_cell_country.csv')
    s = a[a.country == 'Sudan'].copy()
    assert len(s) == 473, f'expected 473 Sudan-labelled cells, got {len(s)}'
    keep = s[s.ne_country == 'Sudan']
    drop = s[s.ne_country != 'Sudan']
    assert len(keep) + len(drop) == len(s)
    assert keep.lat.min() > 8.5, f'kept a cell south of South Sudan border: {keep.lat.min()}'
    assert (drop.ne_country == 'South Sudan').sum() == 128
    keep[['grid_id','lat','lon']].sort_values('grid_id').to_csv('outputs/sudan_cells_keep.csv', index=False)
    drop[['grid_id','lat','lon','ne_country']].sort_values('grid_id').to_csv('outputs/sudan_exclude_cells.csv', index=False)
    print('kept', len(keep), 'lat', keep.lat.min(), 'to', keep.lat.max(), 'lon', keep.lon.min(), 'to', keep.lon.max())
    print('dropped', len(drop)); print(drop.ne_country.fillna('(no polygon)').value_counts().to_string())


if __name__ == "__main__":
    step1_audit_cells_export()
    step2_audit_cell_country()
    step3_sudan_cells_fix()
