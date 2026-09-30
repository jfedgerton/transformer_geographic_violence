import pandas as pd, glob, numpy as np, math, pgeocode
R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
print("=== GTD ===",flush=True)
g=pd.read_excel(f"{R}/data_raw/gtd/globalterrorismdb_0522dist.xlsx",
    usecols=lambda c: c in ['iyear','imonth','country_txt','latitude','longitude','nkill'])
print("rows",len(g),"| year",int(g.iyear.min()),"-",int(g.iyear.max()),
    "| latlon nonnull %.3f"%g.latitude.notna().mean())
print("US rows",int((g.country_txt=='United States').sum()),"top countries:")
print(g.country_txt.value_counts().head(8).to_string())
print("=== MVP MASS SHOOTER ===",flush=True)
f=[x for x in glob.glob(f"{R}/data_raw/violence_project/*.xlsx") if 'MASS SHOOTER' in x][0]
m=pd.read_excel(f,sheet_name=0)
print("rows",len(m))
zc=[c for c in m.columns if 'zip' in str(c).lower()]
dc=[c for c in m.columns if any(k in str(c).lower() for k in ['date','year','day'])]
print("zip cols:",zc,"| date cols:",dc[:6])
print("COLS:",[str(c) for c in m.columns][:45])
nomi=pgeocode.Nominatim('us')
zips=m[zc[0]].dropna().astype(str).str.extract(r'(\d{5})')[0].dropna().tolist()
print("shootings with a 5-digit zip:",len(zips),"of",len(m))
q=nomi.query_postal_code(zips)
print("zip->latlon resolved: %d/%d"%(q.latitude.notna().sum(),len(zips)))
print(q[['postal_code','latitude','longitude','place_name','state_code']].head(8).to_string())
pan=pd.read_parquet(f"{R}/data_intermediate/grid_month_panel.parquet",columns=['grid_id','grid_lat_center','grid_lon_center']).drop_duplicates('grid_id')
us=pan[(pan.grid_lat_center>=24)&(pan.grid_lat_center<=50)&(pan.grid_lon_center>=-125)&(pan.grid_lon_center<=-66)]
emb=pd.read_parquet(f"{R}/data_model/embeddings_all_seed123_d64.parquet",columns=['grid_id']).drop_duplicates()
usemb=us.merge(emb,on='grid_id',how='inner')
print(f"US cells in panel: {len(us)} | US cells WITH an embedding: {len(usemb)}")
uskeys=set(zip(np.floor(us.grid_lat_center/0.5).astype(int),np.floor(us.grid_lon_center/0.5).astype(int)))
tot=hit=0
for la,lo in zip(q.latitude,q.longitude):
    if pd.notna(la):
        tot+=1
        if (int(math.floor(la/0.5)),int(math.floor(lo/0.5))) in uskeys: hit+=1
print(f"MVP shooting cells that exist in our US panel: {hit}/{tot}")
