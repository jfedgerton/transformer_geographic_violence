"""patch_fullpreds_random.py -- adds the random-encoder control to transfer_eval_region_fullpreds.py (approved in
principle 2026-10-11, built 2026-10-13). Exact-text edits only; fails if any anchor is missing or appears more than once,
and refuses to run on anything but the deployed (vocab-remap) version of the file.

Control: RANDOM_ENCODER=1 builds the same Encoder architecture from the encoder's seed with PyTorch default init (the
state training starts from) and does NOT load the pretrained checkpoint. Everything else is identical: same held-out
region, same rows, same vocab remap (main spec only), same XGBoost heads and seeds. Outputs get a "_rand" suffix.
Transfer attributable to pretraining = emb_base(pretrained) - emb_base(random), per region x seed x mode.

Edits:
  1. after the "_vorig" OUT line: read RANDOM_ENCODER (no silent default other than 0), require VOCAB_REMAP=1, add
     "_rand" to OUT
  2. checkpoint load: skipped under RANDOM_ENCODER; weights replaced by a fresh init built inside torch.random.fork_rng
     (encoder still constructed once via the common path, so later RNG use is identical across arms)
  3. preds npz name gets "_rand"
Usage (on Roar, from conflict_forecast/):  python3 scripts/phase4_scaling/patch_fullpreds_random.py
"""
import hashlib, os, shutil

F = "scripts/phase4_scaling/transfer_eval_region_fullpreds.py"
BAK = "scripts/phase4_scaling/archive_v4_2026-10-13/transfer_eval_region_fullpreds.py"
EXPECT_MD5_PREFIX = "5c02bdb3e1fb"
src = open(F).read()
assert hashlib.md5(src.encode()).hexdigest().startswith(EXPECT_MD5_PREFIX), "fullpreds changed since review; not patching"
assert "RANDOM_ENCODER" not in src, "already patched"

ENC_LINE = 'enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)\n'
EDITS = [
    ('if not VOCAB_REMAP: OUT = OUT.replace(".csv", "_vorig.csv")\n',
     'if not VOCAB_REMAP: OUT = OUT.replace(".csv", "_vorig.csv")\n'
     '# random-encoder control (2026-10-13): RANDOM_ENCODER=1 -> same architecture, default init from SEED, no checkpoint\n'
     'RANDOM_ENCODER = os.environ.get("RANDOM_ENCODER", "0")\n'
     'assert RANDOM_ENCODER in ("0", "1"), f"RANDOM_ENCODER must be 0 or 1, got {RANDOM_ENCODER}"\n'
     'RANDOM_ENCODER = RANDOM_ENCODER == "1"\n'
     'if RANDOM_ENCODER:\n'
     '    assert VOCAB_REMAP, "random-encoder control runs under the main specification only (VOCAB_REMAP=1)"\n'
     '    OUT = OUT.replace(".csv", "_rand.csv")\n'
     'print(f"[random] RANDOM_ENCODER={int(RANDOM_ENCODER)} OUT={OUT}", flush=True)\n'),
    (ENC_LINE +
     'sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)\n'
     'missing, unexpected = enc.load_state_dict(sd, strict=False)\n'
     'enc.eval()\n',
     # encoder is still built once through the common path in BOTH arms, so the global RNG state after this point is
     # identical across arms (Codex review 4, finding 2); the random arm re-initializes inside fork_rng
     ENC_LINE +
     'if RANDOM_ENCODER:\n'
     '    _devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []\n'
     '    with torch.random.fork_rng(devices=_devs):   # restores CPU and CUDA RNG state on exit\n'
     '        torch.manual_seed(SEED)                  # same random init for a given seed\n'
     '        _rand = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)\n'
     '    enc.load_state_dict(_rand.state_dict(), strict=True); del _rand\n'
     '    missing, unexpected = [], []\n'
     '    print(f"[model] RANDOM encoder, no checkpoint loaded; seed={SEED} "\n'
     '          f"params={sum(p.numel() for p in enc.parameters())}", flush=True)\n'
     'else:\n'
     '    sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)\n'
     '    missing, unexpected = enc.load_state_dict(sd, strict=False)\n'
     'enc.eval()\n'),
    ('np.savez_compressed(f"{R}/outputs/preds_region_{TAG}" + ("" if VOCAB_REMAP else "_vorig") + ".npz",',
     'np.savez_compressed(f"{R}/outputs/preds_region_{TAG}" + ("" if VOCAB_REMAP else "_vorig") + ("_rand" if RANDOM_ENCODER else "") + ".npz",'),
]
for old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"anchor found {n} times: {old[:70]!r}"
    src = src.replace(old, new)
assert src.index("RANDOM_ENCODER = os.environ") < src.index("if RANDOM_ENCODER:\n    _devs"), "flag used before defined"
compile(src, F, "exec")   # syntax check before writing
os.makedirs(os.path.dirname(BAK), exist_ok=True)
assert not os.path.exists(BAK), f"backup already exists: {BAK}"
shutil.copy2(F, BAK)
open(F, "w").write(src)
print(f"[done] patched {F}; original saved to {BAK}; new md5 {hashlib.md5(src.encode()).hexdigest()[:12]}")
