#!/usr/bin/env bash
# 用法: ./run_classifier.sh <1-6> [传给对应 Python 脚本的额外参数...]
#   1 — BERT (classifier_bert.py)
#   2 — GloVe + CNN (classifier_glove_cnn.py)
#   3 — GloVe + MLP (classifier_glove_mlp.py)
#   4 — RoBERTa (classifier_roberta.py)
#   5 — TF-IDF + Logistic Regression (classifier_tfidf_logistic_regression.py)
#   6 — TF-IDF + SVM (classifier_tfidf_svm.py)
#
# 各脚本支持的额外参数（与直接 python classifier_*.py 相同）：
#   1,4  [--csv PATH] [--model STR] [--epochs N] [--batch N] [--lr F] [--max-len N] [--seed N]
#        (1 默认 model=bert-base-uncased；4 默认 roberta-base)
#   2    [--csv PATH] [--glove PATH] [--filters N] [--epochs N] [--batch N] [--lr F] [--max-len N] [--seed N]
#   3    [--csv PATH] [--glove PATH] [--hidden N] [--epochs N] [--batch N] [--lr F] [--max-len N] [--seed N]
#   5,6  [--csv PATH] [--max-features N] [--C F] [--seed N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
fi

alg="${1:-}"
if [[ -n "$alg" ]]; then
  shift
fi

usage() {
  echo "用法: $0 <1-6> [Python 脚本选项...]" >&2
  echo "  1  BERT        → --csv --model --epochs --batch --lr --max-len --seed" >&2
  echo "  2  GloVe+CNN   → --csv --glove --filters --epochs --batch --lr --max-len --seed" >&2
  echo "  3  GloVe+MLP   → --csv --glove --hidden --epochs --batch --lr --max-len --seed" >&2
  echo "  4  RoBERTa     → --csv --model --epochs --batch --lr --max-len --seed" >&2
  echo "  5  TF-IDF+LR   → --csv --max-features --C --seed" >&2
  echo "  6  TF-IDF+SVM  → --csv --max-features --C --seed" >&2
  exit 1
}

[[ -n "$alg" ]] || usage

case "$alg" in
  # [--csv] [--model 默认 bert-base-uncased] [--epochs] [--batch] [--lr] [--max-len] [--seed]
  1) exec "$PY" "$SCRIPT_DIR/classifier_bert.py" "$@" ;;
  # [--csv] [--glove] [--filters] [--epochs] [--batch] [--lr] [--max-len] [--seed]
  2) exec "$PY" "$SCRIPT_DIR/classifier_glove_cnn.py" "$@" ;;
  # [--csv] [--glove] [--hidden] [--epochs] [--batch] [--lr] [--max-len] [--seed]
  3) exec "$PY" "$SCRIPT_DIR/classifier_glove_mlp.py" "$@" ;;
  # [--csv] [--model 默认 roberta-base] [--epochs] [--batch] [--lr] [--max-len] [--seed]
  4) exec "$PY" "$SCRIPT_DIR/classifier_roberta.py" "$@" ;;
  # [--csv] [--max-features] [--C] [--seed]
  5) exec "$PY" "$SCRIPT_DIR/classifier_tfidf_logistic_regression.py" "$@" ;;
  # [--csv] [--max-features] [--C] [--seed]
  6) exec "$PY" "$SCRIPT_DIR/classifier_tfidf_svm.py" "$@" ;;
  *)
    echo "错误: 第一个参数必须是 1–6，收到: $alg" >&2
    usage
    ;;
esac
