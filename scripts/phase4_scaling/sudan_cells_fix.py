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
