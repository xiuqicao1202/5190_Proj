#!/usr/bin/env bash
# Usage: ./run_classifier.sh <1-6|all> [additional arguments for the corresponding Python script...]
#   Recommended: ./run_classifier.sh … or bash run_classifier.sh …
#   Do not use `sh run_classifier.sh`: On Debian/Ubuntu `sh` is often dash, which does not support `set -o pipefail`.
#   If you still call it via sh, the script will auto-detect and `exec` into bash below.
#   1 — BERT (classifier_bert.py)
#   2 — GloVe + CNN (classifier_glove_cnn.py)
#   3 — GloVe + MLP (classifier_glove_mlp.py)
#   4 — RoBERTa (classifier_roberta.py)
#   5 — TF-IDF + Logistic Regression (classifier_tfidf_logistic_regression.py)
#   6 — TF-IDF + SVM (classifier_tfidf_svm.py)
#   all — Run 1→6 sequentially (extra arguments are passed to every script)
#
# Extra arguments supported by each script (same as running python classifier_*.py directly):
#   1,2,3,4  all support [--device auto|cuda|cpu]; when called via run_classifier.sh the default is
#        --device "${CLASSIFIER_DEVICE:-auto}", you can also override with --device cpu at the end of extra args.
#   1,4  [--csv PATH] [--model STR] [--epochs N] [--batch N] [--lr F] [--max-len N] [--seed N] [--device ...]
#        (1 defaults to model=bert-base-uncased; 4 defaults to roberta-base)
#   2    [--csv] [--glove] [--skip-glove-download] [--filters] ... (if glove.6B.100d.txt is missing, the zip will be downloaded automatically)
#   3    [--csv] [--glove] [--skip-glove-download] [--hidden] ...
#   5,6  [--csv PATH] [--max-features N] [--C F] [--seed N]

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

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
  echo "Usage: $0 <1-6|all> [Python script options...]" >&2
  echo "  1  BERT        → --csv --model --epochs --batch --lr --max-len --seed [--device] (default \$CLASSIFIER_DEVICE or auto)" >&2
  echo "  2  GloVe+CNN   → --csv --glove --filters --epochs ... [--device] [--skip-glove-download] (if GloVe file is missing, will download automatically)" >&2
  echo "  3  GloVe+MLP   → --csv --glove --hidden --epochs ... [--device] [--skip-glove-download]" >&2
  echo "  4  RoBERTa     → --csv --model --epochs --batch --lr --max-len --seed [--device] (default \$CLASSIFIER_DEVICE or auto)" >&2
  echo "  5  TF-IDF+LR   → --csv --max-features --C --seed" >&2
  echo "  6  TF-IDF+SVM  → --csv --max-features --C --seed" >&2
  echo "  all            → run 1→6 in order; options will be passed to every script" >&2
  exit 1
}

[[ -n "$alg" ]] || usage

case "$alg" in
  all|ALL)
    for i in 1 2 3 4 5 6; do
      echo "=== run_classifier: $i/6 ===" >&2
      bash "${BASH_SOURCE[0]}" "$i" "$@" || exit $?
    done
    ;;
  # [--csv] [--model default bert-base-uncased] [--epochs] [--batch] [--lr] [--max-len] [--seed] [--device] (see CLASSIFIER_DEVICE)
  1) exec "$PY" "$SCRIPT_DIR/classifier_bert.py" --device "${CLASSIFIER_DEVICE:-auto}" "$@" ;;
  # [--csv] [--glove] [--filters] [--epochs] [--batch] [--lr] [--max-len] [--seed] [--device]
  2) exec "$PY" "$SCRIPT_DIR/classifier_glove_cnn.py" --device "${CLASSIFIER_DEVICE:-auto}" "$@" ;;
  # [--csv] [--glove] [--hidden] [--epochs] [--batch] [--lr] [--max-len] [--seed] [--device]
  3) exec "$PY" "$SCRIPT_DIR/classifier_glove_mlp.py" --device "${CLASSIFIER_DEVICE:-auto}" "$@" ;;
  # [--csv] [--model default roberta-base] [--epochs] [--batch] [--lr] [--max-len] [--seed] [--device] (see CLASSIFIER_DEVICE)
  4) exec "$PY" "$SCRIPT_DIR/classifier_roberta.py" --device "${CLASSIFIER_DEVICE:-auto}" "$@" ;;
  # [--csv] [--max-features] [--C] [--seed]
  5) exec "$PY" "$SCRIPT_DIR/classifier_tfidf_logistic_regression.py" "$@" ;;
  # [--csv] [--max-features] [--C] [--seed]
  6) exec "$PY" "$SCRIPT_DIR/classifier_tfidf_svm.py" "$@" ;;
  *)
    echo "Error: The first argument must be 1–6 or all, got: $alg" >&2
    usage
    ;;
esac
