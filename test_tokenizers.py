"""
BpeTokenizer.kt / UnigramTokenizer.kt 의 로직을 Python으로 재현하여
HuggingFace tokenizer와 동일한 결과를 내는지 검증한다.

Usage:
    python test_tokenizers.py
"""

import json
import os
import re
import sys
import glob

try:
    import regex
except ImportError:
    print("Installing regex...")
    os.system(f"{sys.executable} -m pip install regex -q")
    import regex

# ═══════════════════════════════════════════════════════════════
# BPE Tokenizer (BpeTokenizer.kt 재현)
# ═══════════════════════════════════════════════════════════════

class BpeTokenizerTest:
    """BpeTokenizer.kt의 로직을 그대로 Python으로 재현."""

    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            t = json.load(f)

        model = t["model"]

        # Vocab
        self.vocab = dict(model["vocab"])
        for at in t.get("added_tokens", []):
            self.vocab[at["content"]] = at["id"]

        # Merges — can be "a b" string or ["a", "b"] list
        self.merge_ranks = {}
        for i, merge_entry in enumerate(model["merges"]):
            if isinstance(merge_entry, str):
                parts = merge_entry.split(" ", 1)
            else:
                parts = merge_entry  # already a list
            if len(parts) == 2:
                self.merge_ranks[(parts[0], parts[1])] = i

        self.eos_id = self.vocab.get("<|end_of_text|>", 41775)
        self.pad_id = self.vocab.get("<|pad|>", 41777)

        # GPT-2 byte-to-unicode mapping
        bs = list(range(ord('!'), ord('~') + 1)) + \
             list(range(ord('¡'), ord('¬') + 1)) + \
             list(range(ord('®'), ord('ÿ') + 1))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_to_unicode = {b: chr(c) for b, c in zip(bs, cs)}

    # GPT-4 style regex — requires `regex` module for \p{L}, \p{N}
    PRE_TOKENIZE_RE = regex.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
        r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
        r"""|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
        r"""|\s*[\r\n]+"""
        r"""|\s+(?!\S)"""
        r"""|\s+"""
    )

    def pre_tokenize(self, text):
        return [m.group() for m in self.PRE_TOKENIZE_RE.finditer(text, concurrent=False)]

    def byte_level_encode(self, text):
        bs = text.encode("utf-8")
        return "".join(self.byte_to_unicode[b & 0xFF] for b in bs)

    def bpe(self, token):
        if len(token) <= 1:
            return [token]

        pieces = list(token)  # char 단위

        while len(pieces) > 1:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            merged = pieces[best_idx] + pieces[best_idx + 1]
            pieces[best_idx] = merged
            del pieces[best_idx + 1]

        return pieces

    def encode(self, text):
        words = self.pre_tokenize(text)
        token_ids = []

        for word in words:
            encoded = self.byte_level_encode(word)
            bpe_tokens = self.bpe(encoded)

            for token in bpe_tokens:
                tid = self.vocab.get(token)
                if tid is not None:
                    token_ids.append(tid)
                else:
                    for ch in token:
                        cid = self.vocab.get(ch)
                        if cid is not None:
                            token_ids.append(cid)

        # Post-process: EOS
        token_ids.append(self.eos_id)
        return token_ids


# ═══════════════════════════════════════════════════════════════
# Unigram Tokenizer (UnigramTokenizer.kt 재현)
# ═══════════════════════════════════════════════════════════════

class UnigramTokenizerTest:
    """UnigramTokenizer.kt의 로직을 그대로 Python으로 재현."""

    METASPACE = "\u2581"  # ▁

    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            t = json.load(f)

        model = t["model"]

        # Vocab: [[piece, score], ...]
        self.vocab = [(entry[0], entry[1]) for entry in model["vocab"]]
        self.piece_to_id = {}
        for i, (piece, _) in enumerate(self.vocab):
            self.piece_to_id[piece] = i

        # Added tokens
        for at in t.get("added_tokens", []):
            self.piece_to_id[at["content"]] = at["id"]

        self.unk_id = model.get("unk_id", 3)
        self.cls_id = self.piece_to_id.get("<s>", 0)
        self.sep_id = self.piece_to_id.get("</s>", 2)
        self.pad_id = self.piece_to_id.get("<pad>", 1)

    def normalize(self, text):
        # SentencePiece Precompiled normalizer:
        # control chars / whitespace variants → space
        text = re.sub(r"[\t\n\r\x0b\x0c\u00a0\u2000-\u200b\u2028\u2029\u3000\ufeff]", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text

    def pre_tokenize(self, text):
        replaced = text.replace(" ", self.METASPACE)
        if not replaced.startswith(self.METASPACE):
            replaced = self.METASPACE + replaced
        return replaced

    def tokenize(self, text):
        """Viterbi 알고리즘."""
        if not text:
            return []

        n = len(text)
        best_score = [float("-inf")] * (n + 1)
        best_prev = [-1] * (n + 1)
        best_score[0] = 0.0

        max_piece_len = 64

        for end in range(1, n + 1):
            start_min = max(0, end - max_piece_len)
            for start in range(start_min, end):
                if best_score[start] == float("-inf"):
                    continue

                piece = text[start:end]
                pid = self.piece_to_id.get(piece)

                if pid is not None:
                    score = best_score[start] + self.vocab[pid][1]
                    if score > best_score[end]:
                        best_score[end] = score
                        best_prev[end] = start

            # UNK fallback
            if best_score[end] == float("-inf") and end > 0:
                prev_end = end - 1
                if best_score[prev_end] > float("-inf"):
                    best_score[end] = best_score[prev_end] + (-100.0)
                    best_prev[end] = prev_end

        # Backtrack
        token_ids = []
        pos = n
        while pos > 0:
            prev = best_prev[pos]
            if prev < 0:
                token_ids.append(self.unk_id)
                pos -= 1
            else:
                piece = text[prev:pos]
                pid = self.piece_to_id.get(piece, self.unk_id)
                token_ids.append(pid)
                pos = prev

        token_ids.reverse()
        return token_ids

    def encode(self, text):
        normalized = self.normalize(text)
        if not normalized:
            return [self.cls_id, self.sep_id]
        pre_tokenized = self.pre_tokenize(normalized)
        token_ids = self.tokenize(pre_tokenized)

        # Post-process: [CLS] + tokens + [SEP]
        return [self.cls_id] + token_ids + [self.sep_id]


# ═══════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════

def find_tokenizer_json(model_id):
    """HuggingFace cache에서 tokenizer.json 경로를 찾는다."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # cache에서 찾기
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    safe_name = "models--" + model_id.replace("/", "--")
    pattern = os.path.join(cache_base, safe_name, "snapshots", "*", "tokenizer.json")
    matches = glob.glob(pattern)
    if matches:
        return matches[0], tok
    # local path
    local = os.path.join(model_id, "tokenizer.json")
    if os.path.exists(local):
        return local, tok
    raise FileNotFoundError(f"Cannot find tokenizer.json for {model_id}")


TEST_TEXTS = [
    "Hello world",
    "I love NLP",
    "The quick brown fox jumps over the lazy dog",
    "Banking77 is a classification task",
    "  multiple   spaces  ",
    "Hello\nworld",
    "it's a test don't you think?",
    "안녕하세요",
    "日本語テスト",
    "Это тест на русском",
    "1234567890",
    "",
    "a",
    " ",
    "Hello, World! How's it going? I'm fine.",
]


def test_bpe():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print("BPE TOKENIZER TEST (Jina v5)")
    print("=" * 70)

    model_id = "gomyk/jina-v5-h256-distilled-conv"
    json_path, hf_tok = find_tokenizer_json(model_id)
    my_tok = BpeTokenizerTest(json_path)

    passed = 0
    failed = 0

    for text in TEST_TEXTS:
        hf_ids = hf_tok.encode(text)
        my_ids = my_tok.encode(text)

        match = hf_ids == my_ids
        status = "PASS" if match else "FAIL"

        if match:
            passed += 1
            print(f"  [{status}] {repr(text)}")
            print(f"         IDs: {my_ids}")
        else:
            failed += 1
            print(f"  [{status}] {repr(text)}")
            print(f"         HF:  {hf_ids}")
            print(f"         Mine: {my_ids}")
            # 토큰 비교
            hf_tokens = hf_tok.convert_ids_to_tokens(hf_ids)
            print(f"         HF tokens:  {hf_tokens}")
        print()

    print(f"  Result: {passed} passed, {failed} failed / {passed + failed} total")
    return failed


def test_unigram():
    print()
    print("=" * 70)
    print("UNIGRAM TOKENIZER TEST (mE5s / XLMRoberta)")
    print("=" * 70)

    model_id = "gomyk/me5s-student-me5s_compressed_distilled"
    json_path, hf_tok = find_tokenizer_json(model_id)
    my_tok = UnigramTokenizerTest(json_path)

    passed = 0
    failed = 0

    for text in TEST_TEXTS:
        hf_ids = hf_tok.encode(text)
        my_ids = my_tok.encode(text)

        match = hf_ids == my_ids
        status = "PASS" if match else "FAIL"

        if match:
            passed += 1
            print(f"  [{status}] {repr(text)}")
            print(f"         IDs: {my_ids}")
        else:
            failed += 1
            print(f"  [{status}] {repr(text)}")
            print(f"         HF:  {hf_ids}")
            print(f"         Mine: {my_ids}")
            hf_tokens = hf_tok.convert_ids_to_tokens(hf_ids)
            print(f"         HF tokens:  {hf_tokens}")
        print()

    print(f"  Result: {passed} passed, {failed} failed / {passed + failed} total")
    return failed


if __name__ == "__main__":
    bpe_fails = test_bpe()
    unigram_fails = test_unigram()

    print()
    print("=" * 70)
    total_fails = bpe_fails + unigram_fails
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {total_fails}")
    sys.exit(total_fails)
