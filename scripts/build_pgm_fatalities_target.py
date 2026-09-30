import pandas as pd, numpy as np
R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
u=pd.read_parquet(f"{R}/data_intermediate/ucdp_cleaned.parquet",columns=["priogrid_gid","year_month","type_of_violence","fatalities"])
u=u[u.type_of_violence==1]
agg=u.groupby(["priogrid_gid","year_month"],as_index=False)["fatalities"].sum().rename(columns={"priogrid_gid":"grid_id","fatalities":"sb_fatalities"})
print("UCDP sb total fatalities:",int(agg.sb_fatalities.sum()),"nonzero cell-months:",len(agg))
p=pd.read_parquet(f"{R}/data_intermediate/grid_month_panel.parquet",columns=["grid_id","year_month","grid_lat_center","grid_lon_center"])
p=p[(p.grid_lat_center>=-36)&(p.grid_lat_center<=43)&(p.grid_lon_center>=-19)&(p.grid_lon_center<=63)].copy()
print("AfricaME cell-months:",len(p),"cells:",p.grid_id.nunique())
m=p.merge(agg,on=["grid_id","year_month"],how="left")
m["sb_fatalities"]=m["sb_fatalities"].fillna(0).astype(int)
print("JOIN nonzero:",int((m.sb_fatalities>0).sum()),"fatalities landed:",int(m.sb_fatalities.sum()))
m=m.sort_values(["grid_id","year_month"]).reset_index(drop=True)
m["sb_fat_next"]=m.groupby("grid_id")["sb_fatalities"].shift(-1)
m=m.dropna(subset=["sb_fat_next"]); m["sb_fat_next"]=m["sb_fat_next"].astype(int)
m.to_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet",index=False)
print("SAVED rows",len(m),"zero_rate %.4f"%(m.sb_fat_next==0).mean())
print("describe",{k:round(float(v),3) for k,v in m.sb_fat_next.describe().items()})
