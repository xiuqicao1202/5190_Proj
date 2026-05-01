"""
RoBERTa（报告 2.3.1）
h = RoBERTa(x)，ˆy = softmax(W h + b)。依赖 transformers。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as e:
    raise SystemExit("请安装: pip install transformers") from e


def label_from_url(url: str) -> int:
    return 1 if "foxnews.com" in url.lower() else 0


class HeadlineDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "Newsheadlines" / "url_with_headlines.csv")
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.csv)
    texts = df["headline"].fillna("").astype(str).tolist()
    labels = [label_from_url(str(u)) for u in df["url"].tolist()]

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=args.seed, stratify=np.array(labels)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=np.array(y_temp)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def make_loader(X: List[str], y: List[int], shuffle: bool) -> DataLoader:
        ds = HeadlineDataset(X, y, tokenizer, args.max_len)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    steps = len(train_loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            sched.step()

    def evaluate(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        correct = 0
        total = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"].to(device)
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                tp += ((pred == 1) & (labels == 1)).sum().item()
                fp += ((pred == 1) & (labels == 0)).sum().item()
                fn += ((pred == 0) & (labels == 1)).sum().item()
        acc = correct / max(1, total)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        return acc, f1

    for name, loader in [("val", val_loader), ("test", test_loader)]:
        acc, f1 = evaluate(loader)
        print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")


if __name__ == "__main__":
    main()
