"""
crps_v4_dimcheck.py

PART A  v4 STACK: does (embedding hurdle  +  raw nonparametric neighbors-12), linearly pooled,
        beat raw neighbors-12 (CRPS 0.8713)?  Weight sweep + cell-block bootstrap CI on the best.
PART B  DIM-CHECK: africa-64d-seed456 vs global-64d-seed456 on LatAm/ME incidence.
        Same seed, same 64 dims, only TRAINING REGION differs -> isolates region from the
        64-vs-128 embed-dim confound that muddied the africa_emb-vs-global_emb comparison.

pgm Africa+ME, s=1, test 2018-2023 (Part A) ; LatAm+ME test 2022+ (Part B).
"""
import json, time
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEEDS = [123, 456, 789, 1024, 2025]
S = 250
CHUNK = 100_000
META = {"grid_id", "year_month", "split", "country", "grid_lat_center", "grid_lon_center"}
np.random.seed(123)
rng = np.random.default_rng(123)

def crps_ens(draws, y):
    n, Sx = draws.shape
    t1 = np.abs(draws - y[:, None]).mean(1)
    xs = np.sort(draws, 1); k = np.arange(1, Sx + 1)
    EdXX = (2.0 / Sx**2) * ((2 * k - Sx - 1) * xs).sum(1)
    return t1 - 0.5 * EdXX

# =====================================================================================
# PART A — v4 STACK
# =====================================================================================
try:
    print("="*70, flush=True); print("PART A — v4 STACK", flush=True); print("="*70, flush=True)
    tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")
    tgt = tgt.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
    months = sorted(tgt.year_month.unique()); mi = {m: i for i, m in enumerate(months)}
    cells = sorted(tgt.grid_id.unique()); ci = {c: i for i, c in enumerate(cells)}
    nC, nM = len(cells), len(months)
    M = np.zeros((nC + 1, nM))
    M[tgt.grid_id.map(ci).to_numpy(), tgt.year_month.map(mi).to_numpy()] = tgt.sb_fatalities.to_numpy()
    tgt["crow"] = tgt.grid_id.map(ci).to_numpy(); tgt["mcol"] = tgt.year_month.map(mi).to_numpy()
    ll = tgt.drop_duplicates("grid_id").set_index("grid_id")[["grid_lat_center", "grid_lon_center"]]
    key2row = {(round(la * 2), round(lo * 2)): ci[c] for c, (la, lo) in ll.iterrows()}
    neigh = np.full((nC, 9), nC, dtype=np.int64)
    for c, (la, lo) in ll.iterrows():
        r = ci[c]; k = 0
        for dla in (-1, 0, 1):
            for dlo in (-1, 0, 1):
                neigh[r, k] = key2row.get((round(la*2)+dla, round(lo*2)+dlo), nC); k += 1
    base = tgt[tgt.mcol >= 11].reset_index(drop=True)
    tr_m = (base.year_month <= "2017-12").to_numpy()
    te_m = ((base.year_month >= "2018-01") & (base.year_month <= "2023-12")).to_numpy()
    ylab = base["sb_fat_next"].to_numpy(float); ytr = ylab[tr_m]; yte = ylab[te_m]
    nte = len(yte)
    print(f"test n={nte}  pos={(yte>0).sum()}", flush=True)

    # embedding hurdle, 5-seed (128-dim)
    keys = base[["grid_id", "year_month"]]
    gate = np.zeros(nte); mupos = np.zeros(nte)
    for s in SEEDS:
        e = pd.read_parquet(f"{R}/data_model/embeddings_all_seed{s}.parquet")
        ec = [c for c in e.columns if c not in META]
        X = np.nan_to_num(keys.merge(e[["grid_id","year_month"]+ec], on=["grid_id","year_month"], how="left")[ec].to_numpy())
        Xtr, Xte = X[tr_m], X[te_m]
        g = xgb.train({"objective":"binary:logistic","max_depth":6,"eta":0.1,"subsample":0.8,"colsample_bytree":0.8,"seed":s},
                      xgb.DMatrix(Xtr, label=(ytr>0).astype(int)), num_boost_round=300)
        gate += g.predict(xgb.DMatrix(Xte))
        pos = ytr > 0
        mm = xgb.train({"objective":"count:poisson","max_depth":6,"eta":0.1,"subsample":0.8,"seed":s},
                       xgb.DMatrix(Xtr[pos], label=ytr[pos]), num_boost_round=300)
        mupos += np.clip(mm.predict(xgb.DMatrix(Xte)), 1.0, None)
        print(f"  hurdle seed {s} done", flush=True)
    gate /= len(SEEDS); mupos /= len(SEEDS)
    yp = ytr[ytr > 0]; rpos = max(yp.mean()**2 / max(yp.var()-yp.mean(), 1e-6), 1e-3)

    # neighbours-12 empirical draws for test rows
    crow = base["crow"].to_numpy()[te_m]; mcol = base["mcol"].to_numpy()[te_m]
    wte = mcol[:, None] - np.arange(11, -1, -1)[None, :]
    Nb = M[neigh[crow][:, :, None], wte[:, None, :]].reshape(nte, -1)   # (nte,108)

    WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]   # w = weight on EMBEDDING hurdle
    crps_sum = {w: 0.0 for w in WEIGHTS}
    for i in range(0, nte, CHUNK):
        sl = slice(i, i+CHUNK); n = len(yte[sl])
        g = gate[sl]; mu = mupos[sl]
        fire = rng.random((n, S)) < g[:, None]
        p = rpos / (rpos + mu[:, None])
        cnt = np.maximum(stats.nbinom.rvs(rpos, p, size=(n, S), random_state=123), 1)
        H = (fire * cnt).astype(float)                       # hurdle draws (n,S)
        Nbc = Nb[sl]; yc = yte[sl]
        for w in WEIGHTS:
            mask = rng.random((n, S)) < w
            idxN = rng.integers(0, Nbc.shape[1], (n, S))
            mix = np.where(mask, H, Nbc[np.arange(n)[:, None], idxN])
            crps_sum[w] += crps_ens(mix, yc).sum()
    crps_mean = {w: crps_sum[w]/nte for w in WEIGHTS}
    best_w = min(WEIGHTS, key=lambda w: crps_mean[w])
    print("STACK weight sweep (w = weight on embedding hurdle; w=0 ~ raw neighbors-12):", flush=True)
    for w in WEIGHTS:
        print(f"  w={w:.1f}  CRPS {crps_mean[w]:.4f}", flush=True)
    print(f"BEST w={best_w}  CRPS {crps_mean[best_w]:.4f}   (raw neighbors-12 ref = 0.8713)", flush=True)

    # per-cell CRPS for best_w mix and for raw Nb, then cell-block bootstrap CI on the difference
    cr_mix = np.empty(nte); cr_nbr = np.empty(nte)
    for i in range(0, nte, CHUNK):
        sl = slice(i, i+CHUNK); n = len(yte[sl])
        g = gate[sl]; mu = mupos[sl]
        fire = rng.random((n, S)) < g[:, None]
        p = rpos / (rpos + mu[:, None])
        cnt = np.maximum(stats.nbinom.rvs(rpos, p, size=(n, S), random_state=7), 1)
        H = (fire * cnt).astype(float); Nbc = Nb[sl]; yc = yte[sl]
        mask = rng.random((n, S)) < best_w
        idxN = rng.integers(0, Nbc.shape[1], (n, S))
        mix = np.where(mask, H, Nbc[np.arange(n)[:, None], idxN])
        cr_mix[sl] = crps_ens(mix, yc); cr_nbr[sl] = crps_ens(Nbc, yc)
    diff = cr_nbr - cr_mix                                   # >0 => stack better than raw neighbors-12
    uniq, inv = np.unique(crow, return_inverse=True)
    csum = np.zeros(len(uniq)); ccnt = np.zeros(len(uniq))
    np.add.at(csum, inv, diff); np.add.at(ccnt, inv, 1.0)
    boot = np.array([(lambda ix: csum[ix].sum()/ccnt[ix].sum())(rng.integers(0, len(uniq), len(uniq))) for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = "YES" if lo > 0 else ("worse" if hi < 0 else "n.s.")
    print(f"STACK vs raw neighbors-12:  CRPS {crps_mean[best_w]:.4f} vs {cr_nbr.mean():.4f}", flush=True)
    print(f"  Delta (nbr - stack) = {diff.mean():.5f}  95% CI [{lo:.5f},{hi:.5f}]  beats nbr-12? {sig}", flush=True)
    pd.DataFrame({"w": WEIGHTS, "crps": [crps_mean[w] for w in WEIGHTS]}).to_csv(f"{R}/outputs/crps_v4_stack_sweep.csv", index=False)
    pd.DataFrame({"best_w":[best_w], "stack_crps":[crps_mean[best_w]], "nbr12_crps":[cr_nbr.mean()],
                  "delta":[diff.mean()], "ci_lo":[lo], "ci_hi":[hi], "beats_nbr12":[sig]}).to_csv(
        f"{R}/outputs/crps_v4_stack_result.csv", index=False)
    print("saved outputs/crps_v4_stack_*.csv", flush=True)
except Exception as ex:
    import traceback; print("PART A FAILED:", ex); traceback.print_exc()

# =====================================================================================
# PART B — DIM-CHECK (matched 64-dim, same seed 456, region differs)
# =====================================================================================
try:
    print("="*70, flush=True); print("PART B — DIM-CHECK", flush=True); print("="*70, flush=True)
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset"]
    NF_FIELDS = 7; MAX_SEQ_LEN = 200; BATCH = 512
    vocabs = json.load(open(f"{R}/data_intermediate/vocab.json"))
    vsz = {k: max(v.values())+1 for k, v in vocabs.items()}; vsm = {k: vsz[k]+1 for k in vsz}

    class Enc(nn.Module):
        def __init__(self, vsm, d, nh=4, nl=2, dr=0.1):
            super().__init__()
            self.emb_event_type=nn.Embedding(vsm["event_type"],d,padding_idx=0)
            self.emb_sub_event_type=nn.Embedding(vsm["sub_event_type"],d,padding_idx=0)
            self.emb_actor1=nn.Embedding(vsm["actor1"],d,padding_idx=0)
            self.emb_actor2=nn.Embedding(vsm["actor2"],d,padding_idx=0)
            self.emb_actor1_type=nn.Embedding(vsm["actor1_type"],d,padding_idx=0)
            self.emb_actor2_type=nn.Embedding(vsm["actor2_type"],d,padding_idx=0)
            self.emb_month_offset=nn.Embedding(vsm["month_offset"],d,padding_idx=0)
            self.field_proj=nn.Linear(NF_FIELDS*d,d); self.proj_dropout=nn.Dropout(dr)
            self.cls_token=nn.Parameter(torch.zeros(1,1,d)); nn.init.normal_(self.cls_token,std=0.02)
            lyr=nn.TransformerEncoderLayer(d_model=d,nhead=nh,dim_feedforward=d*4,dropout=dr,batch_first=True)
            self.transformer=nn.TransformerEncoder(lyr,num_layers=nl,enable_nested_tensor=False)
            self.ln=nn.LayerNorm(d)
            self.head_event_type=nn.Linear(d,vsm["event_type"]-1)
            self.head_sub_event_type=nn.Linear(d,vsm["sub_event_type"]-1)
            self.head_actor1=nn.Linear(d,vsm["actor1"]-1); self.head_actor2=nn.Linear(d,vsm["actor2"]-1)
            self.head_actor1_type=nn.Linear(d,vsm["actor1_type"]-1); self.head_actor2_type=nn.Linear(d,vsm["actor2_type"]-1)
            self.head_month_offset=nn.Linear(d,vsm["month_offset"]-1)
        def encode_sequence(self, seq, mask):
            B=seq.size(0)
            e=torch.cat([self.emb_event_type(seq[:,:,0]),self.emb_sub_event_type(seq[:,:,1]),
                self.emb_actor1(seq[:,:,2]),self.emb_actor2(seq[:,:,3]),
                self.emb_actor1_type(seq[:,:,4]),self.emb_actor2_type(seq[:,:,5]),
                self.emb_month_offset(seq[:,:,6])],dim=-1)
            e=self.proj_dropout(self.field_proj(e))
            cls=self.cls_token.expand(B,-1,-1); e=torch.cat([cls,e],dim=1)
            cm=torch.ones(B,1,dtype=torch.bool,device=mask.device); mwc=torch.cat([cm,mask],dim=1)
            out=self.transformer(e,src_key_padding_mask=~mwc); z=self.ln(out[:,0,:]); z[mask.sum(1)==0]=0.0
            return z
    class DS(Dataset):
        def __init__(self, ss): self.s=[json.loads(x) if isinstance(x,str) else [] for x in ss]
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            seq=self.s[i][-MAX_SEQ_LEN:] if len(self.s[i])>MAX_SEQ_LEN else self.s[i]
            t=torch.zeros(MAX_SEQ_LEN,NF_FIELDS,dtype=torch.long); m=torch.zeros(MAX_SEQ_LEN,dtype=torch.bool)
            for j,ev in enumerate(seq):
                for f in range(NF_FIELDS): t[j,f]=ev[f]
                m[j]=True
            return t,m
    def coll(b): return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])
    def extract(ss, ckpt, d):
        mdl=Enc(vsm,d).to(DEVICE); mdl.load_state_dict(torch.load(ckpt,map_location=DEVICE,weights_only=True)); mdl.eval()
        ld=DataLoader(DS(ss),batch_size=BATCH,shuffle=False,num_workers=0,collate_fn=coll); Z=[]
        with torch.no_grad():
            for seq,mask in ld: Z.append(mdl.encode_sequence(seq.to(DEVICE),mask.to(DEVICE)).cpu().numpy())
        return np.concatenate(Z,0)
    def fe(Xtr,ytr,Xte,yte):
        m=xgb.train({"objective":"binary:logistic","max_depth":6,"eta":0.1,"subsample":0.8,"colsample_bytree":0.8,"seed":456},
                    xgb.DMatrix(Xtr,label=ytr),num_boost_round=300)
        p=m.predict(xgb.DMatrix(Xte)); return roc_auc_score(yte,p), average_precision_score(yte,p)

    bf = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet",
                         columns=["year_month","grid_lat_center","grid_lon_center","event_sequence","violence_next_month"])
    rows=[]
    for name,(a0,a1,o0,o1) in {"LatAm":(-56,33,-118,-34),"ME":(12,42,34,63)}.items():
        d=bf[(bf.grid_lat_center>=a0)&(bf.grid_lat_center<=a1)&(bf.grid_lon_center>=o0)&(bf.grid_lon_center<=o1)].copy()
        d=d[d.violence_next_month.notna()].sort_values("year_month").reset_index(drop=True)
        trm=(d.year_month<="2020-12").to_numpy(); tem=(d.year_month>="2022-01").to_numpy()
        y=d.violence_next_month.to_numpy().astype(int); ytr,yte=y[trm],y[tem]
        ss=d.event_sequence.values
        za=extract(ss, f"{R}/data_model/repr_model_phase2a_seed456.pt", 64)   # AFRICA 64d
        zg=extract(ss, f"{R}/data_model/repr_model_seed456_d64.pt", 64)       # GLOBAL 64d (matched)
        for nm,Z in [("africa_64_s456",za),("global_64_s456",zg)]:
            roc,pr=fe(Z[trm],ytr,Z[tem],yte); rows.append({"region":name,"model":nm,"roc":roc,"pr":pr})
            print(f"  {name} {nm:16s} ROC {roc:.4f}  PR {pr:.4f}", flush=True)
    out=pd.DataFrame(rows); out.to_csv(f"{R}/outputs/dimcheck_africa_vs_global_64d.csv", index=False)
    print(out.to_string(index=False)); print("saved outputs/dimcheck_africa_vs_global_64d.csv", flush=True)
except Exception as ex:
    import traceback; print("PART B FAILED:", ex); traceback.print_exc()
