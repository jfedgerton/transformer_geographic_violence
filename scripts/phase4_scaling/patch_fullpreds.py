"""patch_fullpreds.py -- adds the main-specification vocab remap to transfer_eval_region_fullpreds.py (seed evals that
feed Tables 2/3, alert counts and the transfer tables). Exact-text edits only; fails if any anchor is missing or
appears more than once, and refuses to run on anything but the reviewed version of the file.
Edits:
  1. after the vsm line: import vocab_remap, build per-field lookup lists (LUT) from the encoder TAG,
     add "_vorig" to OUT when VOCAB_REMAP=0
  2. SeqDS: t[j, f] = ev[SEL[f]]  ->  t[j, f] = LUT[f][ev[SEL[f]]]
  3. preds file name gets "_vorig" when VOCAB_REMAP=0
Usage (on Roar, from conflict_forecast/):  python3 scripts/phase4_scaling/patch_fullpreds.py
"""
import hashlib, os, shutil

F = "scripts/phase4_scaling/transfer_eval_region_fullpreds.py"
BAK = "scripts/phase4_scaling/archive_v2_2026-10-10/transfer_eval_region_fullpreds.py"
EXPECT_MD5_PREFIX = "b9b46cebfb03"
src = open(F).read()
assert hashlib.md5(src.encode()).hexdigest().startswith(EXPECT_MD5_PREFIX), "fullpreds changed since review; not patching"

EDITS = [
    ('vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}  # +1 mask slot (matches trainer)\n',
     'vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}  # +1 mask slot (matches trainer)\n'
     '# main specification (2026-10-10): tokens never seen in this encoder\'s pretraining -> <UNK>; VOCAB_REMAP=0 = original\n'
     'import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
     'from vocab_remap import remap_enabled, remap_tables\n'
     'VOCAB_REMAP = remap_enabled()\n'
     'LUT = remap_tables(TAG, FIELDS, vsm, VOCAB_REMAP)\n'
     'if not VOCAB_REMAP: OUT = OUT.replace(".csv", "_vorig.csv")\n'
     'print(f"[vocab] VOCAB_REMAP={int(VOCAB_REMAP)} OUT={OUT}", flush=True)\n'),
    ('                t[j, f] = ev[SEL[f]]\n',
     '                t[j, f] = LUT[f][ev[SEL[f]]]\n'),
    ('np.savez_compressed(f"{R}/outputs/preds_region_{TAG}.npz",',
     'np.savez_compressed(f"{R}/outputs/preds_region_{TAG}" + ("" if VOCAB_REMAP else "_vorig") + ".npz",'),
]
for old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"anchor found {n} times: {old[:70]!r}"
    src = src.replace(old, new)
assert src.index("LUT = remap_tables") < src.index("LUT[f][ev[SEL[f]]]"), "LUT used before it is defined"
os.makedirs(os.path.dirname(BAK), exist_ok=True)
assert not os.path.exists(BAK), f"backup already exists: {BAK}"
shutil.copy2(F, BAK)
open(F, "w").write(src)
print(f"[done] patched {F}; original saved to {BAK}")
