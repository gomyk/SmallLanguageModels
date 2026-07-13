"""training.csv(개인 대화 데이터셋) → conversation_distill.txt 재생성.

conversation_distill.txt는 공개 데이터가 아니라 개인 대화 데이터셋 `data/training.csv`에서
파생된다. 이 스크립트는 그 추출 로직을 그대로 재현한다 (실제 파일과 앞부분 40/40 일치 검증됨).

training.csv 구조:
    chat:   "[{'Other': 'text'}, {'Self': 'text'}, ...]"  (파이썬 리터럴 문자열)
    labels: "[0, 0, 0, 1]"
    lang:   "de" 등

출력(conversation_distill.txt) — row 하나당:
    1) 각 turn의 텍스트를 한 줄씩
    2) 마지막에 전체 turn을 " [SEP] "로 이은 한 줄

사용:
    python build_conversation_corpus.py \
        --csv data/training.csv \
        --out data/distill_corpus/conversation_distill.txt

생성 후 동일성 검증:
    wc -l  data/distill_corpus/conversation_distill.txt   # 19520517
    md5sum data/distill_corpus/conversation_distill.txt   # 7c45f097b12b9f5c69af3109f06b28f3
"""
import argparse
import ast
import csv
import os
import sys

csv.field_size_limit(10 * 1024 * 1024)  # 긴 chat 필드 대비


def build(csv_path, out_path):
    if not os.path.exists(csv_path):
        sys.exit(f"[error] CSV not found: {csv_path}\n"
                 f"        training.csv는 개인 데이터셋이라 원 소유자로부터 직접 받아야 한다. "
                 f"docs/DATA_SOURCES.md 참고.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_rows, n_lines, n_bad = 0, 0, 0
    # utf-8-sig: training.csv 선두 BOM 처리. newline="": csv 규격 준수.
    with open(csv_path, encoding="utf-8-sig", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            n_rows += 1
            try:
                chat = ast.literal_eval(row["chat"])
                turns = [list(d.values())[0] for d in chat]
            except Exception:
                n_bad += 1
                continue
            if not turns:
                continue
            for t in turns:
                fout.write(t + "\n")
                n_lines += 1
            fout.write(" [SEP] ".join(turns) + "\n")
            n_lines += 1
            if n_rows % 200000 == 0:
                print(f"  {n_rows:,} rows -> {n_lines:,} lines")

    print(f"Done. rows={n_rows:,}, lines={n_lines:,}, skipped={n_bad:,}")
    print(f"Wrote: {out_path}")
    print("검증: wc -l / md5sum 로 아래와 대조 (docs/DATA_SOURCES.md)")
    print("  expected lines: 19,520,517   md5: 7c45f097b12b9f5c69af3109f06b28f3")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/training.csv")
    p.add_argument("--out", default="data/distill_corpus/conversation_distill.txt")
    args = p.parse_args()
    build(args.csv, args.out)
