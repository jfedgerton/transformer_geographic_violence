#!/usr/bin/env python
"""alert_counts_finetune_fullsrc.py -- equal-alert-budget counts for the FINE-TUNE arm.

Written 2026-10-30 (Claude), approved by Jared as a pipeline stage. Belongs with
05_finetune_eval / stage 6a in the v5 driver.

WHY: main.tex Section 4.3 (sec:finetune) needs the count of additional violent
cell-months identified at 0.25, 0.5 and 0.75 percent for the fine-tuned encoder
versus the same network trained from scratch. No existing output holds that
comparison. alert_counts_share_M_fullsrc.csv is the NESTED arm at each share
(base vs base+embedding), per alert_counts_fullsrc.py part C.

Logic mirrors alert_counts_fullsrc.py part B exactly. Budget k = number of violent
test cell-months. Flag the top-k by np.argsort(-score, kind="stable") so ties break
identically. extra_hits = TP(pretrained) - TP(scratch).

Part 1 is a VALIDATION GATE. It recomputes the nested counts from
outputs/preds_region_M_contrastive_e_fullsrc_loro_{rg}_s{seed}.npz and requires an
exact match against outputs/alert_counts_nested_M_fullsrc.csv. If that fails the
top-k routine is wrong and the script exits without writing anything.

Part 2 computes the fine-tune counts and writes
outputs/alert_counts_finetune_M_fullsrc.csv. Refuses to overwrite.

Run: venv/bin/python scripts/phase4_scaling/alert_counts_finetune_fullsrc.py
CPU only. Requires the project venv; system python3.6 cannot read these npz files.
"""
import csv
import os
import numpy as np

ROOT = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SIZE = "M"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
SHARES = ["0.0025", "0.005", "0.0075"]
DRAWS = [0, 1, 2, 3, 4]
OUT = os.path.join(ROOT, "outputs", "alert_counts_finetune_%s_fullsrc_full.csv" % SIZE)
NESTED_REF = os.path.join(ROOT, "outputs", "alert_counts_nested_%s_fullsrc.csv" % SIZE)


def topk_tp(score, y, k):
    assert score.shape == y.shape, "score/label length mismatch"
    assert np.isfinite(score).all(), "non-finite score"
    flag = np.zeros(score.shape[0], dtype=bool)
    flag[np.argsort(-score, kind="stable")[:k]] = True
    assert int(flag.sum()) == k, "flagged != k"
    tp = int(y[flag].sum())
    assert 0 <= tp <= k, "tp out of range"
    return tp


# ---- part 1: validation gate, reproduce the nested counts ----
def validate_against_nested():
    ref = {}
    with open(NESTED_REF) as fh:
        for r in csv.DictReader(fh):
            ref[(r["region"], int(r["seed"]))] = (
                int(r["events"]), int(r["tp_base"]), int(r["tp_embbase"]), int(r["extra_hits"])
            )
    assert len(ref) == len(REGIONS) * len(SEEDS), "nested reference has %d rows" % len(ref)
    bad = []
    for rg in REGIONS:
        for sd in SEEDS:
            f = os.path.join(ROOT, "outputs",
                             "preds_region_%s_contrastive_e_fullsrc_loro_%s_s%d.npz" % (SIZE, rg, sd))
            assert os.path.exists(f), "missing %s" % f
            d = np.load(f, allow_pickle=True)
            for key in ("y", "base", "emb_base"):
                assert key in d, "%s missing key %s" % (f, key)
            y = d["y"].astype(np.int64)
            k = int(y.sum())
            assert k > 0, "k == 0 for %s s%d" % (rg, sd)
            tp_b = topk_tp(d["base"].astype(np.float64), y, k)
            tp_e = topk_tp(d["emb_base"].astype(np.float64), y, k)
            got = (k, tp_b, tp_e, tp_e - tp_b)
            want = ref[(rg, sd)]
            if got != want:
                bad.append((rg, sd, got, want))
    if bad:
        print("[VALIDATION FAILED] top-k routine does not reproduce the nested counts")
        for rg, sd, got, want in bad[:10]:
            print("   %-13s s%d got=%s want=%s" % (rg, sd, got, want))
        raise SystemExit(1)
    tot = sum(v[3] for v in ref.values()) / float(len(SEEDS))
    print("[validation] OK, reproduced all %d nested rows exactly" % len(ref))
    print("[validation] seed-averaged nested total = %.1f (expected 709.7)" % tot)


# ---- part 2: fine-tune counts ----
def finetune_counts():
    rows = []
    for rg in REGIONS:
        for sd in SEEDS:
            for sh in SHARES:
                f = os.path.join(ROOT, "outputs", "share_grid_fullsrc",
                                 "preds_%s_finetune_%s_s%d_f%s_full.npz" % (SIZE, rg, sd, sh))
                assert os.path.exists(f), "missing %s" % f
                d = np.load(f, allow_pickle=True)
                assert "__y" in d, "%s missing __y" % f
                y = d["__y"].astype(np.int64)
                k = int(y.sum())
                assert k > 0, "k == 0 for %s s%d f%s" % (rg, sd, sh)
                pre_keys = [key for key in d.keys() if key.startswith("ft_pretrained|%s|" % sh)]
                scr_keys = [key for key in d.keys() if key.startswith("ft_scratch|%s|" % sh)]
                assert len(pre_keys) == len(DRAWS), "%s has %d pretrained draws" % (f, len(pre_keys))
                assert len(scr_keys) == len(DRAWS), "%s has %d scratch draws" % (f, len(scr_keys))
                for dr in DRAWS:
                    kp = "ft_pretrained|%s|%d" % (sh, dr)
                    ks = "ft_scratch|%s|%d" % (sh, dr)
                    assert kp in d and ks in d, "%s missing draw %d" % (f, dr)
                    tp_s = topk_tp(d[ks].astype(np.float64), y, k)
                    tp_p = topk_tp(d[kp].astype(np.float64), y, k)
                    rows.append(dict(region=rg, seed=sd, share=sh, draw=dr, events=k, k=k,
                                     tp_scratch=tp_s, tp_pretrained=tp_p,
                                     extra_hits=tp_p - tp_s))
    assert len(rows) == len(REGIONS) * len(SEEDS) * len(SHARES) * len(DRAWS), "row count %d" % len(rows)
    return rows


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite %s" % OUT
    validate_against_nested()
    rows = finetune_counts()
    cols = ["region", "seed", "share", "draw", "events", "k", "tp_scratch", "tp_pretrained", "extra_hits"]
    tmp = OUT + ".part"
    with open(tmp, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.rename(tmp, OUT)
    print("[done]", OUT, len(rows), "rows")
    print("")
    print("share    " + "".join("%14s" % rg for rg in REGIONS) + "%14s" % "TOTAL")
    for sh in SHARES:
        per = []
        for rg in REGIONS:
            v = [r["extra_hits"] for r in rows if r["region"] == rg and r["share"] == sh]
            per.append(sum(v) / float(len(v)))
        print("%-9s" % sh + "".join("%14.1f" % x for x in per) + "%14.1f" % sum(per))
