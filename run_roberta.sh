#!/usr/bin/env bash
# RoBERTa 超参数实验脚本
# 用法: bash run_roberta.sh
# 每个实验结果会保存到 Finetune_Params/ 下对应的子目录中。

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi

# echo "===== [0/5] 基线: roberta-base, lr=2e-5, batch=16, epochs=3, max-len=128 ====="
# "$PY" classifier_roberta.py \
#   --model roberta-base \
#   --lr 2e-5 \
#   --batch 16 \
#   --epochs 3 \
#   --max-len 128 \
#   --seed 42

echo "===== [1/5] 更多epochs: roberta-base, lr=2e-5, batch=16, epochs=5, max-len=128 ====="
"$PY" classifier_roberta.py \
  --model roberta-base \
  --lr 2e-5 \
  --batch 16 \
  --epochs 5 \
  --max-len 128 \
  --seed 42

echo "===== [2/5] 更高学习率: lr=3e-5, epochs=3 ====="
"$PY" classifier_roberta.py \
  --model roberta-base \
  --lr 3e-5 \
  --batch 16 \
  --epochs 3 \
  --max-len 128 \
  --seed 42

echo "===== [3/5] 短文本优化: max-len=64, batch=32, lr=2e-5 ====="
"$PY" classifier_roberta.py \
  --model roberta-base \
  --lr 2e-5 \
  --batch 32 \
  --epochs 3 \
  --max-len 64 \
  --seed 42

echo "===== [4/5] 更复杂模型: roberta-large, batch=8, lr=1e-5, epochs=3 ====="
"$PY" classifier_roberta.py \
  --model roberta-large \
  --lr 1e-5 \
  --batch 8 \
  --epochs 3 \
  --max-len 128 \
  --seed 42



echo "===== 所有实验完成 ====="
