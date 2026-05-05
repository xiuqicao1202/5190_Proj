"""
BERT (Section 2.3.1 Report)
h = BERT(x),  ŷ = softmax(W h + b), cross-entropy loss. Requires transformers.
"""
from __future__ import annotations

import argparse
import os
import sys
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
    raise SystemExit("Please install: pip install transformers") from e

from param_storage import (
    copy_script_source,
    finetune_base,
    finetune_prefix,
    huggingface_pretrained_cache,
    project_root,
    write_training_txt,
)


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
    print("[hint] WSL2: Install the latest NVIDIA driver for WSL on Windows; running `nvidia-smi` inside your distribution should show the GPU.")
    print("[hint] PyTorch must be installed **with CUDA support** (CPU version will always result in cuda_available=False). For example:")
    print('       pip3 install torch --index-url https://download.pytorch.org/whl/cu124')
    print("       (Check the version number on https://pytorch.org/get-started/locally/ )")


def _print_cuda_diagnostics() -> None:
    print(f"[diag] torch={torch.__version__!r}  torch.version.cuda={torch.version.cuda!r}")
    if torch.version.cuda is None:
        print("[diag] The current PyTorch is likely a **CPU build**; you should use a CUDA-enabled build.")
    if _in_wsl() and not torch.cuda.is_available():
        _print_cuda_install_hints()


def resolve_torch_device(mode: str) -> torch.device:
    m = mode.lower().strip()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            print("[error] Requested --device cuda, but torch.cuda.is_available() is False.")
            _print_cuda_diagnostics()
            raise SystemExit(1)
        return torch.device("cuda")
    if m != "auto":
        raise ValueError(f"unknown device mode: {mode!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    dev = torch.device("cpu")
    print("[device] auto → CPU (CUDA not detected)")
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
    ap.add_argument("--model", type=str, default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Contains Pretrained_Params/huggingface and Finetune_Params/; defaults to the directory of this script.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: Use GPU if CUDA is available; cuda: force GPU (exit if unavailable); cpu: force CPU",
    )
    args = ap.parse_args()

    proj = project_root(args.project_dir, script_dir=Path(__file__).parent)
    hf_cache = str(huggingface_pretrained_cache(proj))
    print(f"[paths] project={proj}\n        HuggingFace pretrained cache: {hf_cache}")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=hf_cache)
    print(f"[timing] AutoTokenizer.from_pretrained: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, cache_dir=hf_cache
    )
    print(f"[timing] AutoModelForSequenceClassification.from_pretrained: {time.perf_counter() - t0:.2f}s")

    device = resolve_torch_device(args.device)
    model.to(device)
    if device.type == "cuda":
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)} (device_count={torch.cuda.device_count()})")
    elif args.device == "cpu":
        print("[device] CPU (--device cpu)")

    pin = device.type == "cuda"

    def make_loader(X: List[str], y: List[int], shuffle: bool) -> DataLoader:
        ds = HeadlineDataset(X, y, tokenizer, args.max_len)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, pin_memory=pin)

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

    metrics: dict = {}
    for name, loader in [("val", val_loader), ("test", test_loader)]:
        acc, f1 = evaluate(name, loader)
        metrics[name] = {"accuracy": acc, "f1_fox": f1}
        print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")

    prefix = finetune_prefix(Path(__file__).stem)
    run_dir = finetune_base(proj) / prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    hf_out = run_dir / f"{prefix}_finetuned_hf"
    hf_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_out)
    tokenizer.save_pretrained(hf_out)
    meta_py = Path(__file__).resolve()
    write_training_txt(
        run_dir / f"{prefix}.txt",
        {
            "timestamp_and_note": f"prefix={prefix}",
            "cmd_args": " ".join(sys.argv),
            "pretrained_model": {"model_id_or_path": args.model, "hf_cache_dir": hf_cache},
            "train_hyperparameters": {
                "csv": str(args.csv.resolve()),
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "max_len": args.max_len,
                "seed": args.seed,
                "device_arg": args.device,
            },
            "validation_and_test_metrics": metrics,
            "saved_files": {"finetuned_huggingface_dir": str(hf_out)},
        },
    )
    copy_script_source(meta_py, run_dir / f"{prefix}.py")
    print(f"[save] Finetune_Params/{prefix}/ written: {prefix}_finetuned_hf, {prefix}.txt, {prefix}.py")


if __name__ == "__main__":
    main()
