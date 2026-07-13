"""Distill me5s_compressed_v4 on the full 19.88M corpus (20 epochs).

Uses SmallModel/distill.py (has checkpoint resume every 1000 steps).
Continues from v4_1ep checkpoint (copied to v4_distilled).
"""
import os
import torch
from distill import distill_student, load_mteb_task_texts

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smallmodel-lib"))
from smallmodel.teachers import TEACHERS

STUDENT = "C:/Users/Moon/finetuning-workshop/SmallModel/students/me5s/me5s_compressed_v4"
TEACHER_KEY = "me5s"
EPOCHS = 20
BATCH_SIZE = 512
LR = 2e-5
PATIENCE = 3

t = TEACHERS[TEACHER_KEY]
print(f"Teacher: {t['model_id']}")
print(f"Student: {STUDENT}")

os.chdir(os.path.dirname(__file__))
texts = load_mteb_task_texts(include_conversations=True)
print(f"Total distillation corpus: {len(texts):,} texts")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

distill_student(
    teacher_name=t["model_id"],
    student_path=STUDENT,
    texts=texts,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    device=device,
    trust_remote_code=t["trust_remote_code"],
    patience=PATIENCE,
)
print("Done")
