import json, torch, torch.nn as nn, pandas as pd, numpy as np

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
CKPT = f"{R}/data_model/repr_model_L_contrastive_a_s123.pt"
PANEL = f"{R}/data_intermediate/dengue_br_muni_v1.parquet"
VOCAB = f"{R}/data_intermediate/dengue_vocab_br_muni_v1.json"
D, NL, NH, DROP, MAXLEN = 256, 6, 8, 0.1, 200

voc = json.load(open(VOCAB))
VS = [len(voc["severity"]) + 1, 201]

class DengueEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, D, padding_idx=0) for v in VS])
        self.field_proj = nn.Linear(2 * D, D); self.pd = nn.Dropout(DROP)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D)); nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(D, NH, D * 4, DROP, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
    def cls_vec(self, seq, mask):
        e = self.pd(self.field_proj(torch.cat([self.embs[i](seq[:, :, i]) for i in range(2)], -1)))
        B = seq.size(0); cls = self.cls_token.expand(B, -1, -1)
        e = torch.cat([cls, e], 1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool), mask], 1)
        return self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])

sd = torch.load(CKPT, map_location="cpu")
print("=== checkpoint ln stats vs default init ===")
lw, lb = sd["ln.weight"], sd["ln.bias"]
print(f"ln.weight  (pretrained): mean={lw.mean():.4f} std={lw.std():.4f} min={lw.min():.4f} max={lw.max():.4f}")
print(f"ln.bias    (pretrained): mean={lb.mean():.4f} std={lb.std():.4f} min={lb.min():.4f} max={lb.max():.4f}")
print(f"ln.weight default init would be all-ones, ln.bias all-zeros")
print()

torch.manual_seed(123)
m_scratch = DengueEnc().eval()
torch.manual_seed(123)
m_warm = DengueEnc().eval()
keep = {k: v for k, v in sd.items() if k.startswith("transformer.") or k.startswith("ln.") or k == "cls_token"}
miss, unexp = m_warm.load_state_dict(keep, strict=False)
print(f"[warm] transferred {len(keep)-len(unexp)}/{len(keep)} tensors; unexpected={len(unexp)}")
print()

df = pd.read_parquet(PANEL, columns=["event_sequence"]).head(64)
seqs = [json.loads(s) if isinstance(s, str) else [] for s in df["event_sequence"].values]
seq_t = torch.zeros(64, MAXLEN, 2, dtype=torch.long); mask_t = torch.zeros(64, MAXLEN, dtype=torch.bool)
for i, sq in enumerate(seqs):
    for j, ev in enumerate(sq[:MAXLEN]):
        seq_t[i, j, 0] = ev[0]; seq_t[i, j, 1] = min(int(ev[1]), 199) + 1
        mask_t[i, j] = True

with torch.no_grad():
    cls_scratch = m_scratch.cls_vec(seq_t, mask_t)
    cls_warm = m_warm.cls_vec(seq_t, mask_t)

print("=== CLS vector produced by the body, fed into the (identically fresh) head/embeds ===")
print(f"scratch body: mean={cls_scratch.mean():.4f} std={cls_scratch.std():.4f} norm/row={cls_scratch.norm(dim=1).mean():.4f}")
print(f"warm    body: mean={cls_warm.mean():.4f} std={cls_warm.std():.4f} norm/row={cls_warm.norm(dim=1).mean():.4f}")
