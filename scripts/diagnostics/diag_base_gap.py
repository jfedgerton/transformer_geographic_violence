import pandas as pd, numpy as np
O='outputs/'; rows=[]
for r in ['africa','mena','europe','americas','south_asia','asia_pacific']:
  for s in [123,456,789]:
    t=f'L_contrastive_e_loro_{r}_s{s}'
    a=pd.read_csv(O+f'eval_region_{t}.csv'); b=pd.read_csv(O+f'eval_region_{t}_rand.csv')
    k=['target','mode','model']
    m=a.merge(b,on=k,suffixes=('_p','_r'))
    m=m[m.model.isin(['base','persistence'])]
    m['gap']=(m.score_p-m.score_r).abs()
    for tg,g in m.groupby('target'): rows.append((tg,r,s,g.gap.max(),len(g)))
d=pd.DataFrame(rows,columns=['target','region','seed','maxgap','n'])
print(d.groupby('target').agg(maxgap=('maxgap','max'),runs=('maxgap','size'),runs_nonzero=('maxgap',lambda x:(x>1e-6).sum())).to_string())
