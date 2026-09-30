import pandas as pd, numpy as np
f="data_raw/crimecount/code_crime_core_all.parquet"
df=pd.read_parquet(f, columns=["city_name","date_single","offense_against"])
dt=pd.to_datetime(df["date_single"], errors="coerce")
df["year"]=dt.dt.year; df["month"]=dt.dt.month
print("rows",len(df)," cities",df.city_name.nunique()," yr",int(dt.dt.year.min()),"-",int(dt.dt.year.max())," NaT",int(dt.isna().sum()),flush=True)
print("=== offense_against ===\n"+df["offense_against"].value_counts(dropna=False).to_string(),flush=True)
c=df.city_name.astype(str)
g=pd.DataFrame({"city":c,"year":df["year"],"month":df["month"]}).dropna(subset=["year"])
g["year"]=g["year"].astype(int)
piv=g.groupby(["city","year"])["month"].nunique().reset_index().pivot(index="city",columns="year",values="month").fillna(0).astype(int)
print("=== months present per city-year (12=full) ===\n"+piv.to_string(),flush=True)
