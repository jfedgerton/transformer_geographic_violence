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
