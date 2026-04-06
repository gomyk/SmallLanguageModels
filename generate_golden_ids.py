"""
Android 유닛 테스트용 golden expected IDs를 생성한다.

HuggingFace tokenizer로 각 TC의 expected token IDs를 계산하고
JSON 파일로 저장 → Android test assets에 넣어서 사용.

Usage:
    python generate_golden_ids.py
    → golden_bpe.json, golden_unigram.json 생성
"""

import json
import sys
from test_tokenizers import (
    LANG_BASIC, LANG_LONG, SPECIAL_CHARS, EDGE_CASES,
    MIXED_LANG, NUMBERS_CODE, STRESS,
)

def generate(model_id, output_file):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    categories = {
        "lang_basic": LANG_BASIC,
        "lang_long": LANG_LONG,
        "special_chars": SPECIAL_CHARS,
        "edge_cases": EDGE_CASES,
        "mixed_lang": MIXED_LANG,
        "numbers_code": NUMBERS_CODE,
        "stress": STRESS,
    }

    golden = {}
    total = 0
    for cat_name, cases in categories.items():
        for label, text in cases.items():
            key = f"{cat_name}/{label}"
            ids = tok.encode(text)
            golden[key] = {"text": text, "ids": ids}
            total += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    print(f"  → {output_file}: {total} cases")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Generating golden IDs...")
    print("\n[BPE - Jina v5]")
    generate("gomyk/jina-v5-h256-distilled-conv", "golden_bpe.json")
    print("\n[Unigram - mE5]")
    generate("gomyk/me5s-student-me5s_compressed_distilled", "golden_unigram.json")
    print("\nDone! Copy golden_*.json to Android test assets.")


if __name__ == "__main__":
    main()
