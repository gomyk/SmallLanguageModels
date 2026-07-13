"""Create me5s_compressed_v3 with byte_fallback enabled.

1. Copy me5s_compressed -> me5s_compressed_v3
2. Modify tokenizer.json: set byte_fallback=true, add 256 byte tokens
3. Resize model embeddings by +256 (random init from mean)
"""
import json
import os
import shutil
import torch
from safetensors.torch import load_file, save_file

SRC = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed"
DST = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v3"

if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
print(f"Copied -> {DST}")

tok_path = os.path.join(DST, "tokenizer.json")
with open(tok_path, "r", encoding="utf-8") as f:
    tok = json.load(f)

assert tok["model"]["type"] == "Unigram"
old_vocab = tok["model"]["vocab"]
old_vocab_size = len(old_vocab)
print(f"Old vocab size: {old_vocab_size}")
print(f"Old byte_fallback: {tok['model'].get('byte_fallback')}")

existing_tokens = {entry[0] for entry in old_vocab}
byte_tokens_to_add = []
for b in range(256):
    tok_str = f"<0x{b:02X}>"
    if tok_str not in existing_tokens:
        byte_tokens_to_add.append(tok_str)

print(f"Byte tokens to add: {len(byte_tokens_to_add)}")

for tok_str in byte_tokens_to_add:
    old_vocab.append([tok_str, 0.0])

tok["model"]["byte_fallback"] = True
new_vocab_size = len(old_vocab)
print(f"New vocab size: {new_vocab_size}")

with open(tok_path, "w", encoding="utf-8") as f:
    json.dump(tok, f, ensure_ascii=False)

cfg_path = os.path.join(DST, "config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
old_cfg_vocab = cfg["vocab_size"]
cfg["vocab_size"] = new_vocab_size
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Config vocab_size: {old_cfg_vocab} -> {new_vocab_size}")

sf_path = os.path.join(DST, "model.safetensors")
from safetensors import safe_open
state = {}
with safe_open(sf_path, framework="pt") as sf:
    for k in sf.keys():
        state[k] = sf.get_tensor(k).clone()

emb_key = None
for k in state:
    if "embeddings.word_embeddings.weight" in k or "embed_tokens.weight" in k:
        emb_key = k
        break
assert emb_key is not None, "Couldn't find embedding key"
print(f"Embedding key: {emb_key}")

old_emb = state[emb_key]
print(f"Old embedding shape: {tuple(old_emb.shape)}")
assert old_emb.shape[0] == old_vocab_size, f"Mismatch: emb {old_emb.shape[0]} vs vocab {old_vocab_size}"

mean = old_emb.mean(dim=0, keepdim=True)
std = old_emb.std(dim=0, keepdim=True)
n_new = new_vocab_size - old_vocab_size
new_rows = mean + torch.randn(n_new, old_emb.shape[1], dtype=old_emb.dtype) * std * 0.1
new_emb = torch.cat([old_emb, new_rows], dim=0)
state[emb_key] = new_emb
print(f"New embedding shape: {tuple(new_emb.shape)}")

save_file(state, sf_path)
print(f"Saved -> {sf_path}")

print("\nv3 ready at:", DST)
