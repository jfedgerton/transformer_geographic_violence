import numpy as np, pandas as pd
R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
seeds=["123","456","789","101112","131415"]
df=pd.concat([pd.read_csv(f"{R}/outputs/eval_full_L_contrastive_e_s{s}.csv") for s in seeds],ignore_index=True)
df["m2"]=np.where(df["mode"].str.startswith("A_pooled"),"A","B")
g=df.groupby(["unit","m2","target","model"],as_index=False)["score"].mean()
p=g.pivot_table(index=["unit","m2","target"],columns="model",values="score").reset_index()
B=p[p.m2=="B"]
rng=np.random.default_rng(123)
def ci(v):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    if len(v)<2: return (np.nan,np.nan,np.nan)
    b=[rng.choice(v,len(v),replace=True).mean() for _ in range(2000)]
    return v.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
print("5-seed L.e Mode-B transfer (mean over held-out units, country-block bootstrap 95pct CI)")
for t in ["incidence","onset","noncontig_onset","rare_event_f1","severity_crps"]:
    s=B[B.target==t]
    if len(s)==0: continue
    hi=t!="severity_crps"
    deb=(s.emb_base-s.base) if hi else (s.base-s.emb_base)
    dep=(s.emb-s.persistence) if hi else (s.persistence-s.emb)
    m1,l1,u1=ci(deb.values); m2,l2,u2=ci(dep.values)
    print(f"{t:16s} emb_base={s.emb_base.mean():.4f} base={s.base.mean():.4f} d_eb={m1:+.4f}[{l1:+.3f},{u1:+.3f}] | emb={s.emb.mean():.4f} pers={s.persistence.mean():.4f} d_ep={m2:+.4f}[{l2:+.3f},{u2:+.3f}] n={s.unit.nunique()}")
