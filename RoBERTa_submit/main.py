"""
Local eval: load RoBERTa Model and score a CSV with processed_title / label columns.
Run from this directory: python main.py
# 执行完整命令行如下（假设当前目录为 RoBERTa_submit）：
# python main.py --csv test.csv --batch-size 32 --device cuda

"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from model import Model, resolve_roberta_pth


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


# classifier_roberta.py：label_from_url → 1=FoxNews、0=NBC；写 csv 时 label_map={1:"FoxNews",0:"NBC"}，
# 与 model.Model.predict 的字符串约定一致，无需再交换。


def load_processed_title_label(csv_path: Path) -> tuple[list[str], list[str]]:
    titles: list[str] = []
    labels: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {csv_path}")
        # expect columns: headline,label
        if "headline" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns headline and label; got {reader.fieldnames}"
            )
        for row in reader:
            titles.append("" if row.get("headline") is None else str(row["headline"]))
            raw = "" if row.get("label") is None else str(row["label"])
            labels.append(raw)
    return titles, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa on test.csv-style CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=_script_dir() / "test.csv",
        help="CSV with columns processed_title and label (default: ./test.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Predict batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    pth = resolve_roberta_pth()
    if pth is None:
        print("Warning: no roberta.pth beside model_RoBERTa.py or under params_submit/ — using base weights only.")

    titles, labels = load_processed_title_label(csv_path)

    device = torch.device(args.device)
    model = Model()
    model.classifier.to(device)
    model.eval()

    preds: list[str] = []
    bs = max(1, args.batch_size)
    for i in range(0, len(titles), bs):
        preds.extend(model.predict(titles[i : i + bs]))

    n = len(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    acc = correct / n if n else 0.0
    print(f"n={n}  accuracy={acc:.4f}  device={device}")
    if pth is not None:
        print(f"weights: {pth}")


if __name__ == "__main__":
    main()
