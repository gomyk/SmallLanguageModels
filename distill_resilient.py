"""
Segfault-resilient distillation runner.

Jina v5 teacher의 PEFT custom code가 Windows CUDA에서 간헐적 segfault를 일으키므로,
crash 시 자동 재시작하여 체크포인트에서 학습을 이어간다.

Usage:
    python distill_resilient.py --teacher jina_v5 --student jina_v5_h256_hf \
        --student-hf gomyk/jina-v5-h256-distilled --epochs 10 --batch-size 64
"""

import subprocess
import sys
import time


def main():
    max_restarts = 50
    restart_count = 0

    # 모든 CLI 인자를 distill.py에 그대로 전달
    args = sys.argv[1:]

    while restart_count < max_restarts:
        cmd = [sys.executable, "distill.py"] + args
        print(f"\n{'='*60}")
        print(f"  Run #{restart_count + 1} (max {max_restarts})")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*60}\n", flush=True)

        result = subprocess.run(cmd)

        rc = result.returncode
        # Windows crash codes:
        #   0xC0000005 = ACCESS_VIOLATION = 3221225477 / -1073741819
        #   0xC0000374 = HEAP_CORRUPTION = 3221226356 / -1073741020
        CRASH_CODES = {-11, 139, 3221225477, -1073741819, 3221226356, -1073741020}

        if rc == 0:
            print("\nDistillation completed successfully!")
            break
        elif rc in CRASH_CODES:
            restart_count += 1
            print(f"\n*** Crash detected (exit code {rc}). "
                  f"Restarting in 5s... ({restart_count}/{max_restarts}) ***",
                  flush=True)
            time.sleep(5)
        else:
            print(f"\nProcess exited with code {rc}. Not restarting.")
            sys.exit(rc)

    if restart_count >= max_restarts:
        print(f"\nMax restarts ({max_restarts}) reached. Giving up.")
        sys.exit(1)


if __name__ == "__main__":
    main()
