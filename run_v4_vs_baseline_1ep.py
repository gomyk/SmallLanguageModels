"""1-epoch distillation for v4 (anchor) and v3 baseline (no anchor) comparison.

Both start from fresh compressed checkpoints (not the 20-epoch distilled one).
"""
import os
import sys
import argparse
import shutil
import torch

sys.path.insert(0, os.path.dirname(__file__))
from distill import distill_student, load_mteb_task_texts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smallmodel-lib"))
from smallmodel.teachers import TEACHERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["v4", "v3_baseline"], required=True,
                        help="Which model to distill")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    t = TEACHERS["me5s"]

    if args.target == "v4":
        src = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v4"
        out = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v4_1ep"
    else:
        src = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v3"
        out = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v3_baseline_1ep"

    # Fresh copy (don't resume from existing distilled checkpoint)
    if os.path.exists(out):
        print(f"Removing existing {out}")
        shutil.rmtree(out)
    print(f"Copying fresh base: {src} -> {out}")
    shutil.copytree(src, out)

    print(f"Teacher: {t['model_id']}")
    print(f"Student: {out}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    os.chdir(os.path.dirname(__file__))
    texts = load_mteb_task_texts(include_conversations=True)
    print(f"Total distillation corpus: {len(texts):,} texts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    distill_student(
        teacher_name=t["model_id"],
        student_path=out,
        texts=texts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        trust_remote_code=t["trust_remote_code"],
        patience=99,  # disable early stopping for fixed-epoch run
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
