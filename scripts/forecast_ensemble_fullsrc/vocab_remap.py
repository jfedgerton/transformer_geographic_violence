"""vocab_remap.py -- main-specification token remap for the conflict LORO encoders (decided 2026-10-10).

Why: vocab_super.json / vocab.json keep the top-500 actor1 / actor2 by count over ALL regions and years. A token that
occurs only in the held-out region, or only after 2020, therefore has an embedding row that pretraining never updated.
Rule (identical for every arm, every row, every seed): any token the encoder never saw in pretraining is mapped to
<UNK> (id 1) before it reaches the encoder. <PAD> (0) is untouched. The month_offset field is positional and is not
remapped. The same rule applies to the small categorical fields; if their <UNK> was never trained this is logged.

"Seen in pretraining" = appears in the last MAX_SEQ_LEN events of some row the encoder was trained on, exactly as
train_repr_obj.py selects them (baseline_features.parquet, split == "train", country not in the held-out region, no
country_{HOLDOUT_COUNTRIES} flag). These sets are built once by build_seen_tokens.py into
data_intermediate/seen_tokens_loro_{region}.json; they depend on the region only, not on the seed.

Env VOCAB_REMAP=0 turns the remap off (original vocabulary; appendix sensitivity check). Callers add "_vorig" to
their output names in that case so the two specifications never overwrite each other.
"""
import json, os, re
import numpy as np

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
PAD, UNK = 0, 1
NO_REMAP = {"month_offset"}


def remap_enabled():
    v = os.environ.get("VOCAB_REMAP", "1")
    assert v in ("0", "1"), f"VOCAB_REMAP must be 0 or 1, got {v}"
    return v == "1"


def loro_region(tag):
    """'L_contrastive_e_loro_south_asia_s123' -> 'south_asia'; None for non-LORO encoders."""
    m = re.search(r"_loro_(.+)_s\d+$", tag)
    return m.group(1) if m else None


def remap_tables(tag, fields, vsm, enabled):
    """One lookup list per field (index = raw token id, value = id fed to the encoder)."""
    luts = [list(range(vsm[f])) for f in fields]
    region = loro_region(tag)
    if not enabled:
        print(f"[vocab] remap OFF (original vocabulary) for {tag}", flush=True)
        return luts
    assert region is not None, f"VOCAB_REMAP=1 but {tag} is not a LORO encoder; set VOCAB_REMAP=0 explicitly"
    path = f"{R}/data_intermediate/seen_tokens_loro_{region}.json"
    assert os.path.exists(path), f"missing {path}; run build_seen_tokens.py first"
    seen = json.load(open(path))
    assert seen["region"] == region, f"{path} is for {seen['region']}, not {region}"
    for i, f in enumerate(fields):
        if f in NO_REMAP: continue
        assert f in seen["tokens"], f"{path} has no field {f}"
        ok = np.zeros(vsm[f], dtype=bool)
        ids = np.asarray(seen["tokens"][f], dtype=np.int64)
        assert ids.size and ids.min() >= 0 and ids.max() < vsm[f], f"{f}: seen ids out of range"
        ok[ids] = True
        ok[PAD] = True; ok[vsm[f] - 1] = True        # padding and the trainer's mask slot never occur as data tokens
        n_unseen = int((~ok).sum())
        if n_unseen and not ok[UNK]:
            # the rule is applied anyway (one rule for every field); <UNK> itself has no trained row here, which
            # build_seen_tokens.py reports (outputs/vocab_unseen_counts.csv says whether such ids occur in test rows)
            assert f not in ("actor1", "actor2"), f"{f}: <UNK> never seen in pretraining"
            print(f"[vocab][WARN] {region} {f}: {n_unseen} ids unseen and <UNK> itself untrained", flush=True)
        luts[i] = [j if ok[j] else UNK for j in range(vsm[f])]
        print(f"[vocab] {region} {f}: {n_unseen} of {vsm[f]} ids unseen in pretraining -> <UNK>", flush=True)
    return luts
