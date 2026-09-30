"""03_package_ucdp_release.py -- assemble the open HF release directory for the UCDP mini encoder.
Env: UCDP_TAG (default ucdp_S), UCDP_DIM/LAYERS/HEADS matching the trained tag.
Builds release/ucdp_mini_{TAG}/ with: pytorch_model.pt, config.json, ucdp_vocab.json,
modeling_ucdp.py (standalone encoder class + loader), README.md (model card).
The actual HF push is done by Jared (credentialed step).
"""
import os, json, shutil

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
TAG = os.environ.get("UCDP_TAG", "ucdp_S")
DIM = int(os.environ.get("UCDP_DIM", "64"))
NL = int(os.environ.get("UCDP_LAYERS", "2"))
NH = int(os.environ.get("UCDP_HEADS", "4"))
FIELDS = ["type_of_violence", "side_a", "side_b", "deaths_bin", "month_offset"]
SRC = f"{R}/data_model/ucdp_repr_{TAG}.pt"
OUT = f"{R}/release/ucdp_mini_{TAG}"
assert os.path.exists(SRC), f"missing {SRC} -- train first"
os.makedirs(OUT, exist_ok=True)
shutil.copyfile(SRC, f"{OUT}/pytorch_model.pt")
shutil.copyfile(f"{R}/data_intermediate/ucdp_vocab.json", f"{OUT}/ucdp_vocab.json")
vocab = json.load(open(f"{R}/data_intermediate/ucdp_vocab.json"))
vocab_sizes = {f: max(vocab[f].values()) + 1 for f in FIELDS}
config = {"model_type": "ucdp-event-encoder", "fields": FIELDS,
          "vocab_sizes_no_mask": vocab_sizes, "embed_dim": DIM, "n_layers": NL,
          "n_heads": NH, "max_seq_len": 200, "objective": "masked-event-modeling",
          "unit": "PRIO-GRID 0.5 deg cell-month", "training_data": "UCDP GED 25.1 (CC BY 4.0)",
          "seed": 123}
json.dump(config, open(f"{OUT}/config.json", "w"), indent=2)

MODELING = '''"""Standalone loader for the UCDP mini event encoder. Requires torch>=2.0."""
import json, torch, torch.nn as nn

class UcdpEncoder(nn.Module):
    def __init__(self, vsm, fields, d, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.fields = list(fields); self.embed_dim = d
        self.emb = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d)
        self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({f: nn.Linear(d, vsm[f] - 1) for f in fields})

    def encode_sequence(self, seq, mask):
        B = seq.size(0)
        e = torch.cat([self.emb[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        e = torch.cat([self.cls_token.expand(B, -1, -1), e], dim=1)
        cm = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        out = self.transformer(e, src_key_padding_mask=~torch.cat([cm, mask], dim=1))
        z = self.ln(out[:, 0, :])
        pad = (mask.sum(dim=1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

def load_encoder(release_dir="."):
    cfg = json.load(open(f"{release_dir}/config.json"))
    vsm = {f: s + 1 for f, s in cfg["vocab_sizes_no_mask"].items()}
    m = UcdpEncoder(vsm, cfg["fields"], cfg["embed_dim"], cfg["n_heads"], cfg["n_layers"])
    m.load_state_dict(torch.load(f"{release_dir}/pytorch_model.pt", map_location="cpu",
                                 weights_only=True))
    m.eval()
    return m, cfg
'''
open(f"{OUT}/modeling_ucdp.py", "w").write(MODELING)

README = f"""# UCDP Mini Spatial Event Encoder ({TAG})

A small transformer encoder that learns representations of location-months from
UCDP Georeferenced Event Dataset (GED) event histories. Each PRIO-GRID 0.5 degree
cell-month is represented by its trailing 12-month sequence of organized-violence
events (type of violence, dyad sides, fatality bin, month offset), encoded with
masked-event modeling. The CLS embedding is a general-purpose representation for
conflict forecasting and transfer learning: fine-tune it for incidence forecasting,
rebel/state violence, escalation, or related event-based tasks.

- Architecture: field embeddings -> concat-project -> [CLS] + TransformerEncoder
  ({NL} layers, {NH} heads, d={DIM}), masked-event-modeling objective.
- Unit: PRIO-GRID 0.5 deg cell-month; max 200 events per sequence; seed 123.
- Training data: UCDP GED 25.1 (1989-2024), the open (CC BY 4.0) georeferenced
  event dataset of organized violence.

## Usage
```python
from modeling_ucdp import load_encoder
model, cfg = load_encoder(".")
# seq: LongTensor [B, 200, 5] (ids per ucdp_vocab.json; 0 = pad)
# mask: BoolTensor [B, 200] (True = real event)
z = model.encode_sequence(seq, mask)   # [B, {DIM}] cell-month embeddings
```

## License and attribution
Model weights and code: CC BY 4.0. Training data is the UCDP GED (CC BY 4.0).
If you use this model, cite UCDP:

- Davies, S., Pettersson, T., Sollenberg, M., & Oberg, M. (2025). Organized violence
  1989-2024, and the challenges of identifying civilian victims. Journal of Peace
  Research 62(4).
- Sundberg, R. & Melander, E. (2013). Introducing the UCDP Georeferenced Event
  Dataset. Journal of Peace Research 50(4).

## Limitations
- UCDP GED covers organized violence with at least one fatality; protests and
  non-lethal events are not included. Coverage reflects media/source reporting.
- The encoder is a research artifact for representation learning and forecasting
  research. It is not suitable for operational targeting or safety-critical use.
- Embeddings reflect historical reporting patterns and inherit their biases.
"""
open(f"{OUT}/README.md", "w").write(README)
print(f"[done] release dir -> {OUT}")
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")
