"""
BpeTokenizer.kt / UnigramTokenizer.kt 검증 테스트

코틀린 로직을 Python으로 재현하고 HuggingFace tokenizer와 1:1 비교한다.
모든 TC가 PASS여야 코틀린 구현이 올바르다고 판단할 수 있다.

카테고리:
  1. 16개국어 기본 문장
  2. 16개국어 긴 문장
  3. 특수문자 / 이모지 / 구두점
  4. 엣지 케이스 (빈 문자열, 공백, 제어문자)
  5. 혼합 언어
  6. 숫자 / URL / 코드
  7. 반복 / 유니코드 경계

Usage:
    python test_tokenizers.py
    python test_tokenizers.py --bpe-only
    python test_tokenizers.py --unigram-only
    python test_tokenizers.py -v          # verbose (토큰 출력)
"""

import json
import os
import re
import sys
import glob
import argparse
import traceback

try:
    import regex
except ImportError:
    os.system(f"{sys.executable} -m pip install regex -q")
    import regex


# ═══════════════════════════════════════════════════════════════
# BPE Tokenizer (BpeTokenizer.kt 재현)
# ═══════════════════════════════════════════════════════════════

class BpeTokenizerTest:
    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            t = json.load(f)
        model = t["model"]
        self.vocab = dict(model["vocab"])
        for at in t.get("added_tokens", []):
            self.vocab[at["content"]] = at["id"]
        self.merge_ranks = {}
        for i, entry in enumerate(model["merges"]):
            parts = entry.split(" ", 1) if isinstance(entry, str) else entry
            if len(parts) == 2:
                self.merge_ranks[(parts[0], parts[1])] = i
        self.eos_id = self.vocab.get("<|end_of_text|>", 41775)

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
        return "".join(self.byte_to_unicode[b & 0xFF] for b in text.encode("utf-8"))

    def bpe(self, token):
        if len(token) <= 1:
            return [token]
        pieces = list(token)
        while len(pieces) > 1:
            best_rank, best_idx = float("inf"), -1
            for i in range(len(pieces) - 1):
                r = self.merge_ranks.get((pieces[i], pieces[i + 1]))
                if r is not None and r < best_rank:
                    best_rank, best_idx = r, i
            if best_idx == -1:
                break
            pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
            del pieces[best_idx + 1]
        return pieces

    def encode(self, text):
        ids = []
        for word in self.pre_tokenize(text):
            for token in self.bpe(self.byte_level_encode(word)):
                tid = self.vocab.get(token)
                if tid is not None:
                    ids.append(tid)
                else:
                    for ch in token:
                        cid = self.vocab.get(ch)
                        if cid is not None:
                            ids.append(cid)
        ids.append(self.eos_id)
        return ids


# ═══════════════════════════════════════════════════════════════
# Unigram Tokenizer (UnigramTokenizer.kt 재현)
# ═══════════════════════════════════════════════════════════════

class UnigramTokenizerTest:
    METASPACE = "\u2581"

    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            t = json.load(f)
        model = t["model"]
        self.vocab = [(e[0], e[1]) for e in model["vocab"]]
        self.piece_to_id = {piece: i for i, (piece, _) in enumerate(self.vocab)}
        for at in t.get("added_tokens", []):
            self.piece_to_id[at["content"]] = at["id"]
        self.unk_id = model.get("unk_id", 3)
        self.cls_id = self.piece_to_id.get("<s>", 0)
        self.sep_id = self.piece_to_id.get("</s>", 2)

    def normalize(self, text):
        import unicodedata
        # SentencePiece Precompiled normalizer:
        # 1. Protect Hangul Compatibility Jamo (U+3131-U+318E) from NFKC
        #    by replacing with PUA placeholders, then restoring after NFKC.
        protected = {}
        pua_base = 0xF0000
        chars = list(text)
        for i, ch in enumerate(chars):
            cp = ord(ch)
            if 0x3131 <= cp <= 0x318E:
                pua = chr(pua_base + len(protected))
                protected[pua] = ch
                chars[i] = pua
        text = "".join(chars)
        # 2. Full NFKC (combining chars are properly composed)
        text = unicodedata.normalize("NFKC", text)
        # 3. Restore protected jamo
        for pua, orig in protected.items():
            text = text.replace(pua, orig)
        # 4. Control chars → space (NOT U+0000 — kept for UNK)
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f\t\n\r\u00a0\u200b-\u200f\u2028\u2029\u3000\ufeff]", " ", text)
        # 5. Collapse multiple spaces
        return re.sub(r" {2,}", " ", text)

    def pre_tokenize(self, text):
        replaced = text.replace(" ", self.METASPACE)
        return replaced if replaced.startswith(self.METASPACE) else self.METASPACE + replaced

    def tokenize(self, text):
        if not text:
            return []
        n = len(text)
        best_score = [float("-inf")] * (n + 1)
        best_prev = [-1] * (n + 1)
        best_score[0] = 0.0
        for end in range(1, n + 1):
            for start in range(max(0, end - 64), end):
                if best_score[start] == float("-inf"):
                    continue
                pid = self.piece_to_id.get(text[start:end])
                if pid is not None:
                    score = best_score[start] + self.vocab[pid][1]
                    if score > best_score[end]:
                        best_score[end] = score
                        best_prev[end] = start
            if best_score[end] == float("-inf") and end > 0:
                if best_score[end - 1] > float("-inf"):
                    best_score[end] = best_score[end - 1] + (-100.0)
                    best_prev[end] = end - 1
        ids = []
        pos = n
        while pos > 0:
            prev = best_prev[pos]
            if prev < 0:
                ids.append(self.unk_id)
                pos -= 1
            else:
                ids.append(self.piece_to_id.get(text[prev:pos], self.unk_id))
                pos = prev
        ids.reverse()
        # SentencePiece: 연속 UNK를 하나로 합침
        merged = []
        for tid in ids:
            if tid == self.unk_id and merged and merged[-1] == self.unk_id:
                continue  # skip consecutive UNK
            merged.append(tid)
        return merged

    def encode(self, text):
        normalized = self.normalize(text)
        if not normalized:
            return [self.cls_id, self.sep_id]
        return [self.cls_id] + self.tokenize(self.pre_tokenize(normalized)) + [self.sep_id]


# ═══════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════

# 1. 16개국어 기본 문장
LANG_BASIC = {
    "ko": "안녕하세요, 반갑습니다",
    "en": "Hello, nice to meet you",
    "ja": "こんにちは、はじめまして",
    "zh": "你好，很高兴认识你",
    "es": "Hola, mucho gusto en conocerte",
    "fr": "Bonjour, enchanté de vous rencontrer",
    "de": "Hallo, freut mich Sie kennenzulernen",
    "pt": "Olá, prazer em conhecê-lo",
    "it": "Ciao, piacere di conoscerti",
    "ru": "Здравствуйте, приятно познакомиться",
    "ar": "مرحبا، سعيد بلقائك",
    "hi": "नमस्ते, आपसे मिलकर खुशी हुई",
    "th": "สวัสดีครับ ยินดีที่ได้รู้จัก",
    "vi": "Xin chào, rất vui được gặp bạn",
    "id": "Halo, senang bertemu dengan Anda",
    "pl": "Cześć, miło mi cię poznać",
}

# 2. 16개국어 긴 문장 (복잡한 문법/어미)
LANG_LONG = {
    "ko": "오늘 날씨가 좋아서 공원에서 산책을 하면서 커피를 마셨습니다",
    "en": "The researchers discovered that the algorithm's performance significantly improved after fine-tuning",
    "ja": "機械学習モデルの圧縮技術は、エッジデバイスでの推論を高速化するために重要です",
    "zh": "深度学习模型的压缩技术对于在边缘设备上进行高效推理非常重要",
    "es": "Los investigadores descubrieron que el rendimiento del algoritmo mejoró significativamente después del ajuste fino",
    "fr": "Les chercheurs ont découvert que les performances de l'algorithme se sont considérablement améliorées après le réglage fin",
    "de": "Die Forscher entdeckten, dass die Leistung des Algorithmus nach der Feinabstimmung erheblich verbessert wurde",
    "pt": "Os pesquisadores descobriram que o desempenho do algoritmo melhorou significativamente após o ajuste fino",
    "it": "I ricercatori hanno scoperto che le prestazioni dell'algoritmo sono migliorate significativamente dopo il fine-tuning",
    "ru": "Исследователи обнаружили, что производительность алгоритма значительно улучшилась после тонкой настройки",
    "ar": "اكتشف الباحثون أن أداء الخوارزمية تحسن بشكل ملحوظ بعد الضبط الدقيق",
    "hi": "शोधकर्ताओं ने पाया कि फाइन-ट्यूनिंग के बाद एल्गोरिदम का प्रदर्शन काफी बेहतर हुआ",
    "th": "นักวิจัยค้นพบว่าประสิทธิภาพของอัลกอริทึมดีขึ้นอย่างมีนัยสำคัญหลังจากการปรับแต่งอย่างละเอียด",
    "vi": "Các nhà nghiên cứu phát hiện rằng hiệu suất của thuật toán được cải thiện đáng kể sau khi tinh chỉnh",
    "id": "Para peneliti menemukan bahwa kinerja algoritma meningkat secara signifikan setelah penyesuaian halus",
    "pl": "Naukowcy odkryli, że wydajność algorytmu znacznie się poprawiła po dostrojeniu",
}

# 3. 특수문자 / 구두점 / 이모지
SPECIAL_CHARS = {
    "punctuation_basic": "Hello! How are you? I'm fine, thanks.",
    "punctuation_heavy": "Wait... really?! No way!!! Sure; okay: got it.",
    "brackets": "(hello) [world] {test} <tag>",
    "math_symbols": "2 + 3 = 5, x² + y² = r², ∑(i=1→n)",
    "currency": "$100 €50 ¥1000 £75 ₩50000",
    "quotes": '"Hello" \'world\' «bonjour» „hallo" 「こんにちは」',
    "dashes_hyphens": "self-attention — fine-tuning – state-of-the-art",
    "slashes_pipes": "input/output | yes/no \\ backslash",
    "at_hash": "@user #hashtag email@test.com",
    "emoji_basic": "Hello 😀 World 🌍",
    "emoji_complex": "👨‍👩‍👧‍👦 family, 🇰🇷 flag, 🤖 robot",
    "unicode_arrows": "→ ← ↑ ↓ ⇒ ⇐ ➔",
    "unicode_math": "α β γ δ ε ∞ ∫ ∂ ∇ ≠ ≤ ≥",
    "fullwidth": "ＨＥＬＬＯ　ＷＯＲＬＤ",
    "cjk_punctuation": "「こんにちは」、「世界」。",
    "arabic_punctuation": "مرحبا، كيف حالك؟",
    "mixed_scripts_punct": "Hello（世界）こんにちは！",
}

# 4. 엣지 케이스
EDGE_CASES = {
    "empty": "",
    "single_char": "a",
    "single_space": " ",
    "multiple_spaces": "   ",
    "tab": "\t",
    "newline": "\n",
    "crlf": "\r\n",
    "mixed_whitespace": " \t\n\r ",
    "leading_spaces": "   hello",
    "trailing_spaces": "hello   ",
    "only_newlines": "\n\n\n",
    "null_char": "hello\x00world",
    "nbsp": "hello\u00a0world",
    "zero_width_space": "hello\u200bworld",
    "bom": "\ufeffhello",
    "very_long_word": "a" * 200,
    "very_long_number": "1" * 100,
    "repeated_punct": "!!!!!!!!!!!!",
    "single_emoji": "🤖",
    "only_special": "!@#$%^&*()",
}

# 5. 혼합 언어
MIXED_LANG = {
    "ko_en": "오늘 meeting이 있어서 presentation 준비를 했습니다",
    "ja_en": "今日のmeetingでpresentationの準備をしました",
    "zh_en": "今天有个meeting需要准备presentation",
    "ko_ja": "한국어と日本語を混ぜて書いてみます",
    "ar_en": "Hello مرحبا World عالم",
    "ru_en": "Привет Hello Мир World",
    "th_en": "สวัสดี Hello ครับ World",
    "hi_en": "नमस्ते Hello दुनिया World",
    "multi_3lang": "Hello 你好 こんにちは",
    "multi_4lang": "Hello 안녕 Bonjour Hallo",
    "code_mixed_ko": "Python으로 모델을 train하고 ONNX로 export했습니다",
}

# 6. 숫자 / URL / 코드
NUMBERS_CODE = {
    "integers": "0 1 42 100 999 1000 12345 999999",
    "floats": "3.14 0.001 -2.5 1e10 6.022e23",
    "negative": "-1 -100 -3.14",
    "phone": "+82-10-1234-5678",
    "date": "2024-01-15 15:30:00",
    "url": "https://huggingface.co/gomyk/model-name",
    "email": "user@example.com",
    "path": "/home/user/models/tokenizer.json",
    "code_python": "def encode(self, text: str) -> List[int]:",
    "code_json": '{"key": "value", "num": 42}',
    "ip_address": "192.168.1.1:8080",
    "hex": "0xFF 0x1A2B #FFFFFF",
    "version": "v1.2.3-beta.4",
}

# 7. 반복 / 유니코드 경계
STRESS = {
    "repeated_word": "hello " * 20,
    "alternating": "ab" * 50,
    "unicode_boundary": "a\u0300",  # a + combining grave accent = à
    "hangul_jamo": "ㄱㄴㄷㄹㅁㅂㅅㅇ",
    "hangul_compat": "ㅎㅏㄴㄱㅜㄱㅇㅓ",
    "surrogate_emoji": "𝕳𝖊𝖑𝖑𝖔",  # Mathematical Fraktur
    "rare_cjk": "𠀀𠀁𠀂",  # CJK Extension B
    "rtl_bidi": "Hello مرحبا World عالم mixed",
    "zalgo": "H̷̨̧e̸̺̓l̵̰̈́l̸̡̛o̸̧̍",
    "ligatures": "ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ",
    "accented_chars": "àáâãäåèéêëìíîïòóôõöùúûüýÿñ",
    "turkish_i": "İstanbul istanbul ISTANBUL",
}


# ═══════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════

def find_tokenizer_json(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    safe_name = "models--" + model_id.replace("/", "--")
    for p in glob.glob(os.path.join(cache_base, safe_name, "snapshots", "*", "tokenizer.json")):
        return p, tok
    local = os.path.join(model_id, "tokenizer.json")
    if os.path.exists(local):
        return local, tok
    raise FileNotFoundError(f"Cannot find tokenizer.json for {model_id}")


def run_category(name, cases, hf_tok, my_tok, verbose=False):
    """테스트 카테고리를 실행하고 결과를 반환한다."""
    passed, failed, errors = 0, 0, 0
    failures = []

    for label, text in cases.items():
        try:
            hf_ids = hf_tok.encode(text)
            my_ids = my_tok.encode(text)
            if hf_ids == my_ids:
                passed += 1
                if verbose:
                    print(f"    [PASS] {label}: {repr(text[:60])}")
                    print(f"           IDs({len(my_ids)}): {my_ids[:15]}{'...' if len(my_ids)>15 else ''}")
            else:
                failed += 1
                failures.append((label, text, hf_ids, my_ids))
                print(f"    [FAIL] {label}: {repr(text[:60])}")
                print(f"           HF ({len(hf_ids)}):  {hf_ids[:15]}{'...' if len(hf_ids)>15 else ''}")
                print(f"           Mine({len(my_ids)}): {my_ids[:15]}{'...' if len(my_ids)>15 else ''}")
                # 차이 위치 찾기
                for i in range(min(len(hf_ids), len(my_ids))):
                    if hf_ids[i] != my_ids[i]:
                        hf_tokens = hf_tok.convert_ids_to_tokens(hf_ids)
                        print(f"           First diff at pos {i}: "
                              f"HF={hf_ids[i]}({hf_tokens[i] if i < len(hf_tokens) else '?'}) "
                              f"vs Mine={my_ids[i]}")
                        break
        except Exception as e:
            errors += 1
            print(f"    [ERROR] {label}: {e}")
            traceback.print_exc()

    return passed, failed, errors, failures


def test_tokenizer(tok_name, model_id, tok_class, verbose=False):
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"\n{'='*70}")
    print(f" {tok_name} TOKENIZER TEST")
    print(f" Model: {model_id}")
    print(f"{'='*70}")

    json_path, hf_tok = find_tokenizer_json(model_id)
    my_tok = tok_class(json_path)

    categories = [
        ("16-Language Basic", LANG_BASIC),
        ("16-Language Long", LANG_LONG),
        ("Special Characters", SPECIAL_CHARS),
        ("Edge Cases", EDGE_CASES),
        ("Mixed Languages", MIXED_LANG),
        ("Numbers / URLs / Code", NUMBERS_CODE),
        ("Stress / Unicode Boundary", STRESS),
    ]

    total_pass, total_fail, total_err = 0, 0, 0
    all_failures = []

    for cat_name, cases in categories:
        print(f"\n  [{cat_name}] ({len(cases)} cases)")
        p, f, e, failures = run_category(cat_name, cases, hf_tok, my_tok, verbose)
        total_pass += p
        total_fail += f
        total_err += e
        all_failures.extend(failures)
        status = "PASS" if f == 0 and e == 0 else "FAIL"
        print(f"    → {status}: {p} passed, {f} failed, {e} errors / {p+f+e}")

    total = total_pass + total_fail + total_err
    print(f"\n{'─'*70}")
    print(f"  {tok_name} TOTAL: {total_pass} passed, {total_fail} failed, "
          f"{total_err} errors / {total} cases")
    if total_fail == 0 and total_err == 0:
        print(f"  ✓ ALL {total} TESTS PASSED")
    else:
        print(f"  ✗ {total_fail + total_err} FAILURES")

    return total_fail + total_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-only", action="store_true")
    parser.add_argument("--unigram-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    total_failures = 0

    if not args.unigram_only:
        total_failures += test_tokenizer(
            "BPE", "gomyk/jina-v5-h256-distilled-conv",
            BpeTokenizerTest, args.verbose)

    if not args.bpe_only:
        total_failures += test_tokenizer(
            "UNIGRAM", "gomyk/me5s-student-me5s_compressed_distilled",
            UnigramTokenizerTest, args.verbose)

    print(f"\n{'='*70}")
    if total_failures == 0:
        print(" ALL TESTS PASSED")
    else:
        print(f" TOTAL FAILURES: {total_failures}")
    print(f"{'='*70}")
    sys.exit(total_failures)


if __name__ == "__main__":
    main()
