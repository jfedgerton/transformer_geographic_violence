"""crime_finetune.py — cross-domain transfer test (conflict L encoder -> US crime).
Warm-start transformer BODY from repr_model_L_contrastive_a_s123.pt (feat-a, no PRIO);
RE-INIT crime field embeddings + field_proj + a new forecasting head.
ARMS (env ARM): warm_ft | scratch_ft | warm_frozen | scratch_frozen | baseline
  warm_ft      : body warm-started, fine-tune everything
  scratch_ft   : body random, train everything
  warm_frozen  : body warm-started + FROZEN, train only embeds+proj+head (pure repr transfer)
  scratch_frozen: body random + FROZEN, train only embeds+proj+head (control floor)
  baseline     : persistence(lag1) + featurized xgb (no neural net)
LOW-DATA (env FRAC): fraction of TRAIN-city rows used (val/test untouched).
Eval = held-out CITIES. Metrics = PR-AUC (AP) + best-F1. seed 123.
"""
import os, json, time, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
CTAG=os.environ.get("CRIME_TAG","grid150m_violent_wide")
PANEL=f"{R}/data_intermediate/crime_{CTAG}.parquet"
VOCAB=f"{R}/data_intermediate/crime_vocab_{CTAG}.json"
CKPT=f"{R}/data_model/repr_model_L_contrastive_a_s123.pt"
OUT=os.environ.get("CRIME_OUT",f"{R}/outputs/crime_ft_results.csv")
D, NL, NH, DROP = 256, 6, 8, 0.1          # L capacity
MAXLEN=200
HELDOUT=["Boston","St Louis","Seattle","Austin","Los Angeles"]
E=os.environ.get
ARM=E("ARM","warm_ft"); FRAC=float(E("FRAC","1.0")); SEED=int(E("SEED","123"))
EPOCHS=int(E("EPOCHS","30")); PATIENCE=int(E("PATIENCE","6")); BATCH=int(E("BATCH","256"))
LR=float(E("LR","3e-4")); SMOKE=E("SMOKE","0")=="1"
assert ARM in ("warm_ft","scratch_ft","warm_frozen","scratch_frozen","baseline","warm_lnopen")
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cfg] ARM={ARM} FRAC={FRAC} SEED={SEED} SMOKE={SMOKE} dev={DEV}",flush=True)

voc=json.load(open(VOCAB))
VS=[len(voc["offense_type"])+1, len(voc["offense_group"])+1, len(voc["offense_against"])+1, 201]  # +pad; month shifted +1
cols=["city","t","crime_next_month","lag1","lag3","lag6","roll3","months_since_last","spatial_lag","event_sequence"]
df=pd.read_parquet(PANEL,columns=cols)
df=df[df["city"].astype(str)!="nan"].reset_index(drop=True)
is_test=df["city"].isin(HELDOUT).values
wmax=int(df["t"].max())                       # last month with target
VALCUT=wmax-6                                 # last 6 months (of train cities) = val
test_df=df[is_test].reset_index(drop=True)
pool=df[~is_test].reset_index(drop=True)
train_df=pool[pool["t"]<VALCUT].reset_index(drop=True)
val_df=pool[pool["t"]>=VALCUT].reset_index(drop=True)
if FRAC<1.0 and len(train_df)>0:
    train_df=train_df.sample(frac=FRAC,random_state=SEED).reset_index(drop=True)
if SMOKE:
    train_df=train_df.sample(n=min(40000,len(train_df)),random_state=SEED).reset_index(drop=True)
    val_df=val_df.sample(n=min(10000,len(val_df)),random_state=SEED).reset_index(drop=True)
    test_df=test_df.sample(n=min(10000,len(test_df)),random_state=SEED).reset_index(drop=True)
    EPOCHS=2
print(f"[data] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
      f"pos: tr={train_df.crime_next_month.mean():.3f} te={test_df.crime_next_month.mean():.3f}",flush=True)

FEAT=["lag1","lag3","lag6","roll3","months_since_last","spatial_lag"]
def ap_f1(y,s):
    ap=average_precision_score(y,s)
    p,r,th=precision_recall_curve(y,s); f1=2*p*r/(p+r+1e-9); return ap,float(np.nanmax(f1))

def write_row(d):
    hdr=not os.path.exists(OUT)
    pd.DataFrame([d]).to_csv(OUT,mode="a",header=hdr,index=False)
    print(f"[RESULT] {d}",flush=True)

if ARM=="baseline":
    yte=test_df["crime_next_month"].values.astype(int)
    ap_p,f1_p=ap_f1(yte,test_df["lag1"].values.astype(float))
    write_row(dict(arm="persistence_lag1",frac=1.0,seed=SEED,test_ap=round(ap_p,4),test_bestf1=round(f1_p,4),n_test=len(yte),pos_test=round(yte.mean(),4)))
    import xgboost as xgb
    tr=pd.concat([train_df,val_df]) if not SMOKE else train_df
    clf=xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.8,
                          colsample_bytree=0.8,eval_metric="aucpr",n_jobs=4,random_state=SEED)
    clf.fit(tr[FEAT].values, tr["crime_next_month"].values.astype(int))
    s=clf.predict_proba(test_df[FEAT].values)[:,1]
    ap_x,f1_x=ap_f1(yte,s)
    write_row(dict(arm="xgb_featurized",frac=1.0,seed=SEED,test_ap=round(ap_x,4),test_bestf1=round(f1_x,4),n_test=len(yte),pos_test=round(yte.mean(),4)))
    print("[done] baseline",flush=True); raise SystemExit

class DS(Dataset):
    def __init__(self,d):
        self.seq=[]; self.y=[]
        for s,lb in zip(d["event_sequence"].values,d["crime_next_month"].values):
            seq=json.loads(s) if isinstance(s,str) else []
            if len(seq)>MAXLEN: seq=seq[-MAXLEN:]
            self.seq.append(seq); self.y.append(float(lb) if lb==lb else 0.0)
    def __len__(self): return len(self.seq)
    def __getitem__(self,i):
        t=torch.zeros(MAXLEN,4,dtype=torch.long); m=torch.zeros(MAXLEN,dtype=torch.bool)
        for j,ev in enumerate(self.seq[i]):
            t[j,0]=ev[0]; t[j,1]=ev[1]; t[j,2]=ev[2]; t[j,3]=min(int(ev[3]),199)+1  # month shift +1
            m[j]=True
        return t,m,torch.tensor(self.y[i],dtype=torch.float32)
def coll(b): return (torch.stack([x[0] for x in b]),torch.stack([x[1] for x in b]),torch.stack([x[2] for x in b]))

class CrimeEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(v,D,padding_idx=0) for v in VS])
        self.field_proj=nn.Linear(4*D,D); self.pd=nn.Dropout(DROP)
        self.cls_token=nn.Parameter(torch.zeros(1,1,D)); nn.init.normal_(self.cls_token,std=0.02)
        layer=nn.TransformerEncoderLayer(D,NH,D*4,DROP,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,NL,enable_nested_tensor=False)
        self.ln=nn.LayerNorm(D); self.head=nn.Linear(D,1)
    def cls_vec(self,seq,mask):
        e=self.pd(self.field_proj(torch.cat([self.embs[i](seq[:,:,i]) for i in range(4)],-1)))
        B=seq.size(0); cls=self.cls_token.expand(B,-1,-1)
        e=torch.cat([cls,e],1)
        mwc=torch.cat([torch.ones(B,1,dtype=torch.bool,device=mask.device),mask],1)
        return self.ln(self.transformer(e,src_key_padding_mask=~mwc)[:,0,:])
    def forward(self,seq,mask): return self.head(self.cls_vec(seq,mask)).squeeze(-1)

model=CrimeEnc().to(DEV)
if ARM.startswith("warm"):
    sd=torch.load(CKPT,map_location="cpu")
    keep={k:v for k,v in sd.items() if k.startswith("transformer.") or k.startswith("ln.") or k=="cls_token"}
    miss,unexp=model.load_state_dict(keep,strict=False)
    warmed=[k for k in keep if k not in unexp]
    print(f"[warm] transferred {len(warmed)}/{len(keep)} tensors from conflict L; unexpected={len(unexp)}",flush=True)
frozen=ARM.endswith("frozen")
lnopen=ARM=="warm_lnopen"
if frozen:
    for n,p in model.named_parameters():
        if n.startswith("transformer.") or n.startswith("ln.") or n=="cls_token": p.requires_grad=False
    print("[frozen] body frozen; training embeds+proj+head only",flush=True)
if lnopen:
    for n,p in model.named_parameters():
        if n.startswith("transformer.") or n=="cls_token": p.requires_grad=False
    print("[lnopen] transformer body + cls_token frozen; ln UNFROZEN; training ln+embeds+proj+head",flush=True)
ntr=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[model] trainable params={ntr:,}",flush=True)

tr_ds,va_ds,te_ds=DS(train_df),DS(val_df),DS(test_df)
g=torch.Generator(); g.manual_seed(SEED)
tl=DataLoader(tr_ds,batch_size=BATCH,shuffle=True,generator=g,collate_fn=coll,num_workers=4,pin_memory=True)
vl=DataLoader(va_ds,batch_size=512,shuffle=False,collate_fn=coll,num_workers=4)
el=DataLoader(te_ds,batch_size=512,shuffle=False,collate_fn=coll,num_workers=4)
pos=max(1,int(train_df["crime_next_month"].sum())); neg=len(train_df)-pos
pw=torch.tensor([neg/pos],dtype=torch.float32,device=DEV)
lossfn=nn.BCEWithLogitsLoss(pos_weight=pw)
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR)
steps=EPOCHS*max(1,len(tl)); WARM=min(500,steps//10)
def lrl(s):
    if s<WARM: return (s+1)/max(1,WARM)
    pr=(s-WARM)/max(1,steps-WARM); return 0.5*(1+math.cos(math.pi*min(1.0,pr)))
sch=torch.optim.lr_scheduler.LambdaLR(opt,lrl)

def evalset(loader):
    model.eval(); ys,ss=[],[]
    with torch.no_grad():
        for seq,mask,y in loader:
            s=torch.sigmoid(model(seq.to(DEV),mask.to(DEV))).cpu().numpy()
            ss.append(s); ys.append(y.numpy())
    y=np.concatenate(ys); s=np.concatenate(ss); return ap_f1(y.astype(int),s)

best=-1; wait=0; bestpath=f"{R}/data_model/crime_{ARM}_f{FRAC}_s{SEED}.pt"
for ep in range(1,EPOCHS+1):
    model.train(); t0=time.time(); tot=0; nb=0
    for seq,mask,y in tl:
        opt.zero_grad()
        loss=lossfn(model(seq.to(DEV),mask.to(DEV)),y.to(DEV))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); sch.step()
        tot+=loss.item(); nb+=1
    vap,vf1=evalset(vl)
    print(f"[ep {ep:2d}] loss={tot/max(nb,1):.4f} val_ap={vap:.4f} val_f1={vf1:.4f} {time.time()-t0:.0f}s",flush=True)
    if vap>best+1e-4:
        best=vap; wait=0; torch.save(model.state_dict(),bestpath)
    else:
        wait+=1
    if wait>=PATIENCE: print(f"[stop] ep {ep} best_val_ap={best:.4f}",flush=True); break
model.load_state_dict(torch.load(bestpath,map_location=DEV))
tap,tf1=evalset(el)
write_row(dict(arm=ARM,frac=FRAC,seed=SEED,test_ap=round(tap,4),test_bestf1=round(tf1,4),
               n_test=len(te_ds),pos_test=round(float(test_df.crime_next_month.mean()),4),val_ap=round(best,4)))
print("[done]",flush=True)
