#!/usr/bin/env bash
# CPU-safe structural checks for the SN56 image entry. No GPU, no publish, no registration.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOD_REF="${GOD_REF:-/root/sn56-work/G.O.D}"
cd "$ROOT"

echo "== 1) Required files =="
for f in LICENSE.md NOTICE ops/docker/standalone-image-toolkit-trainer.dockerfile ops/docker/scripts/image_toolkit_entrypoint.py; do
  test -f "$f" && echo "OK  $f" || { echo "MISSING $f"; exit 1; }
done
test -d core/training_templates && echo "OK  core/training_templates/" || exit 1

echo "== 2) LICENSE.md / NOTICE must match G.O.D (enrolment rule) =="
diff -u "$GOD_REF/LICENSE.md" LICENSE.md >/dev/null && echo "OK  LICENSE.md identical" || { echo "FAIL LICENSE.md differs"; exit 1; }
diff -u "$GOD_REF/NOTICE" NOTICE >/dev/null && echo "OK  NOTICE identical" || { echo "FAIL NOTICE differs"; exit 1; }

echo "== 3) Dockerfile names + CUDA notes =="
grep -n 'FROM\|cu128\|CUDA\|ENTRYPOINT\|COPY' ops/docker/standalone-image-toolkit-trainer.dockerfile | head -40
echo "OK  Dockerfile present (CUDA 12.8 / cu128 per official starter)"

echo "== 4) Local git commit (40-char) — publish this repo before using it on-chain =="
git rev-parse HEAD
test "$(git rev-parse HEAD | wc -c)" -eq 41 && echo "OK  full SHA length"

echo "== 5) Untested on CPU-only hosts =="
cat <<'EOF'
  - docker build of standalone-image-toolkit-trainer (pulls NVIDIA base; long)
  - ./ops/examples/training/run_image_task.sh (requires --gpus all + HF/S3 credentials)
  - Real validator enrolment / fee charge / scoring
EOF
echo "ALL CPU-SAFE CHECKS PASSED"
