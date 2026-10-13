"""count_alerts.py -- exact true-positive / false-alarm counts from saved Mode B forecasts.

Reads outputs/preds_region_L_contrastive_e_loro_{region}_s123.npz (from transfer_eval_region_preds.py).
Step 0: re-computes PR-AUC from the saved forecasts and asserts it matches eval_region_*.csv (Table 1 source).
Then, per region, two alert rules over the full test period (Jan 2022 - Apr 2025 forecast months):
  A. same alerts as persistence: persistence flags every cell with violence this month (k = its alert count);
     base and base+embedding each flag their top-k cells.  -> extra true positives at equal alert count
  B. same detection as base at 50% recall: k = alerts base needs to catch half of violent cell-months;
     report true positives for each model at that k, and alerts each needs to reach 50% recall.
Ties in model scores are broken by a fixed random order (seed 123).
Output: outputs/alert_counts_s123.csv
"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
rng = np.random.RandomState(123)
rows = []

def topk_tp(y, s, k):
    jitter = rng.rand(len(s)) * 1e-12            # deterministic tie-break
    idx = np.argsort(-(s + jitter))[:k]
    return int(y[idx].sum())

def alerts_for_recall(y, s, r):
    jitter = rng.rand(len(s)) * 1e-12
    cum = np.cumsum(y[np.argsort(-(s + jitter))])
    return int(np.searchsorted(cum, np.ceil(r * y.sum())) + 1)

for rg in REGIONS:
    tag = f"L_contrastive_e_loro_{rg}_s123"
    z = np.load(f"{R}/outputs/preds_region_{tag}.npz", allow_pickle=True)
    y = z["y"].astype(int); P = int(y.sum()); n = len(y)
    pers, base, eb = z["persistence"].astype(float), z["base"], z["emb_base"]
    assert len(base) == len(eb) == n and np.isfinite(base).all() and np.isfinite(eb).all()

    # Step 0: saved forecasts must reproduce the Table 1 numbers
    ref = pd.read_csv(f"{R}/outputs/eval_region_{tag}.csv")
    ref = ref[(ref["mode"] == "B_temporal") & (ref["target"] == "incidence")].set_index("model")
    assert int(ref.loc["base", "n"]) == n, f"{rg}: n {n} != Table 1 n {ref.loc['base','n']}"
    ref = ref["score"]
    for name, s in [("persistence", pers), ("base", base), ("emb_base", eb)]:
        ap = average_precision_score(y, s)
        print(f"  {rg} {name}: AP rerun {ap:.5f} vs Table 1 {ref[name]:.5f} (diff {ap-ref[name]:+.5f})", flush=True)
        assert abs(ap - ref[name]) < 2e-3, f"{rg} {name}: rerun AP differs from Table 1 by more than 0.002"

    # A. same number of alerts as persistence
    kA = int((pers > 0).sum()); tpA_pers = int(y[pers > 0].sum())
    tpA_base, tpA_eb = topk_tp(y, base, kA), topk_tp(y, eb, kA)
    # B. alerts base needs to reach 50% recall
    kB = alerts_for_recall(y, base, 0.5)
    tpB_base, tpB_eb, tpB_pers = topk_tp(y, base, kB), topk_tp(y, eb, kB), topk_tp(y, pers, kB)
    kB_eb = alerts_for_recall(y, eb, 0.5)
    # P. persistence's own operating point: same recall as persistence -> alerts each model needs
    rP = tpA_pers / P
    kP_eb, kP_b = alerts_for_recall(y, eb, rP), alerts_for_recall(y, base, rP)

    rows.append(dict(region=rg, n=n, violent=P,
        A_alerts=kA, A_tp_persistence=tpA_pers, A_tp_base=tpA_base, A_tp_embbase=tpA_eb,
        A_extra_tp_base_vs_pers=tpA_base - tpA_pers, A_extra_tp_emb_vs_base=tpA_eb - tpA_base,
        B_alerts=kB, B_tp_base=tpB_base, B_tp_embbase=tpB_eb, B_tp_persistence=tpB_pers,
        B_extra_tp_emb_vs_base=tpB_eb - tpB_base,
        B_alerts_needed_embbase=kB_eb, B_fewer_false_alarms_emb=kB - kB_eb,
        A_extra_tp_embbase_vs_pers=tpA_eb - tpA_pers,
        P_recall_persistence=round(rP, 4), P_alerts_persistence=kA,
        P_fewer_false_alarms_embbase_vs_pers=kA - kP_eb, P_fewer_false_alarms_base_vs_pers=kA - kP_b))
    print(f"[ok] {rg}: AP reproduced; n={n} violent={P}", flush=True)

out = pd.DataFrame(rows)
tot = out.drop(columns="region").sum(numeric_only=True); tot["region"] = "TOTAL"
out = pd.concat([out, tot.to_frame().T], ignore_index=True)
out.to_csv(f"{R}/outputs/alert_counts_s123.csv", index=False)
pd.set_option("display.width", 250)
print(out[["region", "violent", "A_alerts", "A_tp_persistence", "A_tp_base", "A_tp_embbase",
           "A_extra_tp_embbase_vs_pers", "A_extra_tp_emb_vs_base", "P_fewer_false_alarms_embbase_vs_pers", "B_alerts", "B_extra_tp_emb_vs_base", "B_fewer_false_alarms_emb"]].to_string(index=False))
