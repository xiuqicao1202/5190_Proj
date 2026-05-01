"""
RoBERTa（报告 2.3.1）
h = RoBERTa(x)，ˆy = softmax(W h + b)。依赖 transformers。
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc,redef]
        return iterable

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


def _in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _print_cuda_install_hints() -> None:
    print("[hint] WSL2：在 Windows 上安装支持 WSL 的最新 NVIDIA 驱动；在发行版里执行 `nvidia-smi` 应能看到 GPU。")
    print("[hint] PyTorch 需安装 **CUDA 版** wheel（CPU 版会一直 cuda_available=False）。示例：")
    print('       pip3 install torch --index-url https://download.pytorch.org/whl/cu124')
    print("       （版本号请对照 https://pytorch.org/get-started/locally/ ）")


def _print_cuda_diagnostics() -> None:
    print(f"[diag] torch={torch.__version__!r}  torch.version.cuda={torch.version.cuda!r}")
    if torch.version.cuda is None:
        print("[diag] 当前 PyTorch 很可能是 **CPU 构建**，需换用带 CUDA 的安装包。")
    if _in_wsl() and not torch.cuda.is_available():
        _print_cuda_install_hints()


def resolve_torch_device(mode: str) -> torch.device:
    m = mode.lower().strip()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            print("[error] 请求 --device cuda，但 torch.cuda.is_available() 为 False。")
            _print_cuda_diagnostics()
            raise SystemExit(1)
        return torch.device("cuda")
    if m != "auto":
        raise ValueError(f"unknown device mode: {mode!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    dev = torch.device("cpu")
    print("[device] auto → CPU（未检测到可用 CUDA）")
    _print_cuda_diagnostics()
    return dev


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
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto：有 CUDA 则用 GPU；cuda：强制 GPU（不可用则退出）；cpu：强制 CPU",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    t0 = time.perf_counter()
    df = pd.read_csv(args.csv)
    texts = df["headline"].fillna("").astype(str).tolist()
    labels = [label_from_url(str(u)) for u in df["url"].tolist()]

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=args.seed, stratify=np.array(labels)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=np.array(y_temp)
    )
    print(f"[timing] CSV read + stratified split: {time.perf_counter() - t0:.2f}s")
    print(f"[data] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[timing] AutoTokenizer.from_pretrained: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    print(f"[timing] AutoModelForSequenceClassification.from_pretrained: {time.perf_counter() - t0:.2f}s")

    device = resolve_torch_device(args.device)
    model.to(device)
    if device.type == "cuda":
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)} (device_count={torch.cuda.device_count()})")
    elif args.device == "cpu":
        print("[device] CPU（--device cpu）")

    pin = device.type == "cuda"

    def make_loader(X: List[str], y: List[int], shuffle: bool) -> DataLoader:
        ds = HeadlineDataset(X, y, tokenizer, args.max_len)
        return DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=shuffle,
            pin_memory=pin,
            num_workers=0,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    steps = len(train_loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    train_all_t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_t0 = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=True,
        )
        for batch in pbar:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            sched.step()
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")
        print(f"[timing] train epoch {epoch + 1}/{args.epochs}: {time.perf_counter() - epoch_t0:.2f}s")
    print(f"[timing] all training epochs: {time.perf_counter() - train_all_t0:.2f}s")

    def evaluate(name: str, loader: DataLoader) -> tuple[float, float]:
        model.eval()
        correct = 0
        total = 0
        tp = fp = fn = 0
        ev_t0 = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval {name}", unit="batch", leave=True):
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
        print(f"[timing] evaluate {name}: {time.perf_counter() - ev_t0:.2f}s")
        acc = correct / max(1, total)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        return acc, f1

    for name, loader in [("val", val_loader), ("test", test_loader)]:
        acc, f1 = evaluate(name, loader)
        print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")


if __name__ == "__main__":
    main()
