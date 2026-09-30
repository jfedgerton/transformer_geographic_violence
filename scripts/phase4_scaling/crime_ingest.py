"""Crime (CODE) ingest v2 -> city-cell-by-month panel (ALL cell-months incl quiet, from first appearance).
~150m grid (PredPol standard) main; env grid: CRIME_SPATIAL, CRIME_TARGET, CRIME_PANEL.
Feature-a (event-only, no PRIO). Out: data_intermediate/crime_{TAG}.parquet + crime_vocab_{TAG}.json
"""
import os, json, bisect, numpy as np, pandas as pd
R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SP=os.environ.get("CRIME_SPATIAL","grid150m")   # grid150m | grid1km | blockgroup
TG=os.environ.get("CRIME_TARGET","violent")      # violent | any
PN=os.environ.get("CRIME_PANEL","wide")          # wide=2016-2018 | deep=2010-2018
SMOKE=os.environ.get("SMOKE","0")=="1"
TMIN=int(os.environ.get("TARGET_MIN","1"))
TAG=f"{SP}_{TG}_{PN}"; GRID=0.0015; MAXLEN=200
FIELDS=["offense_type","offense_group","offense_against"]
print(f"[cfg] TAG={TAG} SMOKE={SMOKE} GRID={GRID}",flush=True)

cols=["city_name","offense_type","offense_group","offense_against","date_single","longitude","latitude","census_block"]
df=pd.read_parquet(f"{R}/data_raw/crimecount/code_crime_core_all.parquet",columns=cols)
dt=pd.to_datetime(df["date_single"],errors="coerce")
keep=dt.notna() & df["latitude"].notna() & df["longitude"].notna()
df=df[keep].copy(); dt=dt[keep]
df["yr"]=dt.dt.year.values; df["t"]=(dt.dt.year*12+dt.dt.month).astype(int).values
yrs=list(range(2016,2019)) if PN=="wide" else list(range(2010,2019))
df=df[df["yr"].isin(yrs)].copy()
if SMOKE: df=df[df["city_name"].astype(str).isin(["Chicago","Los Angeles"])].copy()
wmin=min(yrs)*12+1; wmax=max(yrs)*12+12
print(f"[data] events={len(df)} months={df.t.nunique()} cities={df.city_name.nunique()}",flush=True)

if SP.startswith("grid"):
    G=0.01 if SP=="grid1km" else GRID
    df["clat"]=(df["latitude"]/G).round()*G; df["clon"]=(df["longitude"]/G).round()*G
    df["cell"]=df["city_name"].astype(str)+"|"+df["clat"].round(4).astype(str)+"|"+df["clon"].round(4).astype(str)
else:
    cb=df["census_block"].astype(str); df=df[cb.str.len()>=12].copy()
    df["cell"]=df["city_name"].astype(str)+"|"+cb.str[:12]; df["clat"]=df["latitude"]; df["clon"]=df["longitude"]
df["tgt"]=(df["offense_type"].astype(str)=="murder and nonnegligent manslaughter").astype(int) if TG=="homicide" else ((df["offense_against"].astype(str)=="persons").astype(int) if TG=="violent" else 1)

vocab={}
for f in FIELDS:
    cats=sorted(df[f].astype(str).fillna("NA").unique()); vocab[f]={c:i+1 for i,c in enumerate(cats)}
    df[f+"_id"]=df[f].astype(str).fillna("NA").map(vocab[f]).astype(int)
json.dump(vocab, open(f"{R}/data_intermediate/crime_vocab_{TAG}.json","w"))

df=df.sort_values(["cell","t"]).reset_index(drop=True)
mon=df.groupby(["cell","t"]).agg(ntgt=("tgt","sum")).reset_index()
meta=df.groupby("cell").agg(city=("city_name","first"),clat=("clat","mean"),clon=("clon","mean")).reset_index()
cellmon={}
for cell,sub in mon.groupby("cell"): cellmon[cell]=dict(zip(sub.t.astype(int),sub.ntgt.astype(int)))
evtok={}; evtt={}
for cell,sub in df.groupby("cell"):
    tk=list(zip(sub.offense_type_id.astype(int),sub.offense_group_id.astype(int),sub.offense_against_id.astype(int),sub.t.astype(int)))
    evtok[cell]=tk; evtt[cell]=[e[3] for e in tk]
def neigh(cell):
    if not SP.startswith("grid"): return []
    G=0.01 if SP=="grid1km" else GRID
    city,la,lo=cell.split("|"); la=float(la); lo=float(lo); out=[]
    for dla in(-G,0,G):
        for dlo in(-G,0,G):
            if dla==0 and dlo==0: continue
            out.append(f"{city}|{round(la+dla,4)}|{round(lo+dlo,4)}")
    return out

rows=[]
for cell in meta.cell.values:
    cm=cellmon[cell]; tk=evtok[cell]; tt=evtt[cell]; nb=neigh(cell); first=min(cm.keys())
    for t in range(max(first,wmin), wmax):
        hi=bisect.bisect_right(tt,t); seq=tk[max(0,hi-MAXLEN):hi]
        seq=[[a,b,c,min(t-d,199)] for (a,b,c,d) in seq]
        y=1 if cm.get(t+1,0)>=TMIN else 0
        lag1=1 if cm.get(t,0)>0 else 0
        lag3=1 if sum(cm.get(t-k,0) for k in range(0,3))>0 else 0
        lag6=1 if sum(cm.get(t-k,0) for k in range(0,6))>0 else 0
        roll3=sum(cm.get(t-k,0) for k in range(0,3))
        pastm=[m for m in cm if m<=t]; msl=(t-max(pastm)) if pastm else 99
        slag=sum(cellmon.get(x,{}).get(t,0) for x in nb) if nb else 0
        rows.append((cell,t,y,lag1,lag3,lag6,roll3,msl,slag,json.dumps(seq)))
out=pd.DataFrame(rows,columns=["cell","t","crime_next_month","lag1","lag3","lag6","roll3","months_since_last","spatial_lag","event_sequence"])
out=out.merge(meta,on="cell",how="left"); out["year_month"]=out["t"]
op=f"{R}/data_intermediate/crime_{TAG}.parquet"; out.to_parquet(op)
print(f"[out] rows={len(out)} cells={out.cell.nunique()} pos_rate={out.crime_next_month.mean():.4f} vocab_type={len(vocab['offense_type'])} -> {op}",flush=True)
print("[done]",flush=True)
