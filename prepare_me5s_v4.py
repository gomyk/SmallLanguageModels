"""Create me5s_compressed_v4 with teacher-initialized anchor embeddings.

Only anchors that have a DIRECT single-token match in the teacher vocab
(either `c` or `▁c`) are added to the student vocab. For each added token,
the embedding row is copied from the teacher (prefers `▁c` form if both exist,
otherwise `c`; averages the two when both present).

Base: me5s_compressed_v3 (byte_fallback=True, 256 byte tokens).
"""
import json
import os
import shutil
import unicodedata

import torch
from safetensors.torch import save_file, safe_open
from transformers import AutoModel, AutoTokenizer

SRC = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v3"
DST = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v4"
TEACHER_ID = "intfloat/multilingual-e5-small"


def build_anchor_set():
    chars = []
    chars += [chr(c) for c in range(0x1100, 0x1200)]          # Hangul Jamo
    chars += [chr(c) for c in range(0x3131, 0x3164)]          # Compat Jamo
    chars += [chr(c) for c in range(0xAC00, 0xAC00 + 2350)]   # Hangul syllables
    chars += [chr(c) for c in range(0x3041, 0x3097)]          # Hiragana
    chars += [chr(c) for c in range(0x30A1, 0x30FB)]          # Katakana
    chars += [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]   # CJK
    chars += [chr(c) for c in range(0x0620, 0x064B)]          # Arabic
    chars += [chr(c) for c in range(0x0670, 0x06D4)]
    chars += [chr(c) for c in range(0x0905, 0x0940)]          # Devanagari
    chars += [chr(c) for c in range(0x0958, 0x097F)]
    chars += [chr(c) for c in range(0x0E01, 0x0E3B)]          # Thai
    chars += [chr(c) for c in range(0x0E40, 0x0E4F)]
    chars += [chr(c) for c in range(0x0410, 0x0450)]          # Cyrillic
    chars += [chr(c) for c in range(0x0452, 0x0460)]

    seen, result = set(), []
    for c in chars:
        if not c.isprintable():
            continue
        nfkc = unicodedata.normalize("NFKC", c)
        for form in (nfkc, c):
            if form not in seen:
                seen.add(form)
                result.append(form)
    return result


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Copied v3 -> v4 at {DST}")

    tok_path = os.path.join(DST, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = json.load(f)
    assert tok["model"]["type"] == "Unigram"
    old_vocab = tok["model"]["vocab"]
    old_vocab_size = len(old_vocab)
    print(f"v3 vocab size: {old_vocab_size}")

    chars_in_vocab, existing_tokens = set(), set()
    for entry in old_vocab:
        existing_tokens.add(entry[0])
        for c in entry[0]:
            chars_in_vocab.add(c)

    anchors = build_anchor_set()
    print(f"Anchor candidates: {len(anchors)}")

    missing = [c for c in anchors
               if c not in chars_in_vocab and c not in existing_tokens]
    print(f"Missing anchors (before teacher filter): {len(missing)}")

    print(f"Loading teacher: {TEACHER_ID}")
    t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)
    t_vocab = t_tok.get_vocab()
    t_model = AutoModel.from_pretrained(TEACHER_ID)
    t_emb = t_model.get_input_embeddings().weight.detach().cpu()
    print(f"Teacher vocab: {len(t_vocab)}, embedding: {tuple(t_emb.shape)}")

    UNDER = "\u2581"
    direct_addable = []  # (char_to_add, teacher_row_indices)
    for c in missing:
        ids = []
        if (UNDER + c) in t_vocab:
            ids.append(t_vocab[UNDER + c])
        if c in t_vocab:
            ids.append(t_vocab[c])
        if ids:
            direct_addable.append((c, ids))
    print(f"Directly mappable from teacher: {len(direct_addable)} "
          f"({len(direct_addable) / max(len(missing), 1) * 100:.1f}%)")

    byte_score = -20.0
    byte_count = 0
    for entry in old_vocab:
        t = entry[0]
        if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
            entry[1] = byte_score
            byte_count += 1
    print(f"Lowered {byte_count} byte-fallback scores to {byte_score}")

    single_char_scores = []
    for entry in old_vocab:
        t = entry[0]
        if t.startswith("<") and t.endswith(">"):
            continue
        stripped = t.lstrip(UNDER)
        if len(stripped) == 1:
            single_char_scores.append(entry[1])
    single_char_scores.sort()
    anchor_score = (single_char_scores[len(single_char_scores) // 2]
                    if single_char_scores else -11.0)
    print(f"Anchor score: {anchor_score:.2f} "
          f"(median of {len(single_char_scores)} existing single-char tokens)")

    for c, _ in direct_addable:
        old_vocab.append([c, anchor_score])
        existing_tokens.add(c)

    new_vocab_size = len(old_vocab)
    print(f"v4 vocab size: {new_vocab_size} "
          f"(+{new_vocab_size - old_vocab_size} from teacher-direct anchors)")

    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)

    cfg_path = os.path.join(DST, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["vocab_size"] = new_vocab_size
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Updated config.json vocab_size -> {new_vocab_size}")

    sf_path = os.path.join(DST, "model.safetensors")
    state = {}
    with safe_open(sf_path, framework="pt") as sf:
        for k in sf.keys():
            state[k] = sf.get_tensor(k).clone()

    emb_key = None
    for k in state:
        if "embeddings.word_embeddings.weight" in k or "embed_tokens.weight" in k:
            emb_key = k
            break
    assert emb_key is not None, "Embedding key not found"
    print(f"Embedding key: {emb_key}")

    old_emb = state[emb_key]
    assert old_emb.shape[0] == old_vocab_size
    assert old_emb.shape[1] == t_emb.shape[1], \
        f"Hidden dim mismatch: student {old_emb.shape[1]} vs teacher {t_emb.shape[1]}"
    print(f"Old student embedding: {tuple(old_emb.shape)}")

    new_rows = []
    for _, ids in direct_addable:
        row = t_emb[ids].mean(dim=0).to(old_emb.dtype)
        new_rows.append(row)
    new_rows = torch.stack(new_rows, dim=0)
    new_emb = torch.cat([old_emb, new_rows], dim=0)
    state[emb_key] = new_emb
    print(f"New embedding: {tuple(new_emb.shape)} "
          f"(+{new_rows.shape[0]} teacher-copied rows)")

    save_file(state, sf_path)
    print(f"Saved -> {sf_path}")

    size_mb = os.path.getsize(sf_path) / (1024 * 1024)
    print(f"\nv4 ready at: {DST}")
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
