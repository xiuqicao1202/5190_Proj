"""
GloVe + CNN (Report 2.3.1)
E ∈ R^{T×k}, 1D convolution + ReLU, max pooling, linear layer ŷ = σ(w^T h + b).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from param_storage import (
    copy_script_source,
    default_glove_txt,
    ensure_glove_100d,
    finetune_base,
    finetune_prefix,
    project_root,
    resolve_glove_txt,
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
    print("[hint] For WSL2: Install the latest NVIDIA driver for Windows with WSL support; running `nvidia-smi` in your distribution should allow you to see the GPU.")
    print("[hint] PyTorch requires the **CUDA version** wheel (CPU version will always have cuda_available=False). Example:")
    print('       pip3 install torch --index-url https://download.pytorch.org/whl/cu124')
    print("       (Please check the version at https://pytorch.org/get-started/locally/ )")


def _print_cuda_diagnostics() -> None:
    print(f"[diag] torch={torch.__version__!r}  torch.version.cuda={torch.version.cuda!r}")
    if torch.version.cuda is None:
        print("[diag] Your current PyTorch installation is probably a **CPU build**. You need to install a CUDA-enabled package.")
    if _in_wsl() and not torch.cuda.is_available():
        _print_cuda_install_hints()


def resolve_torch_device(mode: str) -> torch.device:
    m = mode.lower().strip()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            print("[error] --device cuda requested, but torch.cuda.is_available() is False.")
            _print_cuda_diagnostics()
            raise SystemExit(1)
        return torch.device("cuda")
    if m != "auto":
        raise ValueError(f"unknown device mode: {mode!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    dev = torch.device("cpu")
    print("[device] auto → CPU (no available CUDA detected)")
    _print_cuda_diagnostics()
    return dev


def tokenize(s: str) -> List[str]:
    return [t for t in s.lower().replace("'", " ").split() if t]


def load_glove_map(path: Path, dim: int) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            emb[w] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return emb


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for w in tokenize(t):
            freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def texts_to_indices(texts: List[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    unk = vocab["<unk>"]
    rows = []
    for t in texts:
        ids = [vocab.get(w, unk) for w in tokenize(t)][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


class GloveCNN(nn.Module):
    def __init__(
        self,
        embedding: torch.Tensor,
        num_filters: int,
        kernel_sizes: List[int],
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=0)
        d = embedding.shape[1]
        self.convs = nn.ModuleList([nn.Conv1d(d, num_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            pooled.append(h.max(dim=2)[0])
        h = torch.cat(pooled, dim=1)
        return self.fc(h).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "Newsheadlines" / "url_with_headlines.csv")
    ap.add_argument("--glove", type=Path, default=None, help="Path to GloVe file; default is Pretrained_Params/glove/glove.6B.100d.txt")
    ap.add_argument(
        "--skip-glove-download",
        action="store_true",
        help="Do not auto-download: exit if glove.6B.100d.txt is missing (default: auto-download glove.6B.zip if needed)",
    )
    ap.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Directory containing Pretrained_Params/ and Finetune_Params/; defaults to the script directory",
    )
    ap.add_argument("--filters", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: Use GPU if CUDA is available; cuda: force GPU; cpu: force CPU",
    )
    args = ap.parse_args()

    proj = project_root(args.project_dir, script_dir=Path(__file__).parent)
    glove_txt = resolve_glove_txt(proj, args.glove)
    ensure_glove_100d(glove_txt, auto_download=not args.skip_glove_download)
    print(f"[paths] project={proj}\n        GloVe -> {glove_txt}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)} (device_count={torch.cuda.device_count()})")
    elif args.device == "cpu":
        print("[device] CPU (--device cpu)")

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

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    vocab = build_vocab(X_train)
    dim = 100
    n_vocab = len(vocab)
    mat = np.random.randn(n_vocab, dim).astype(np.float32) * 0.01
    mat[vocab["<pad>"]] = 0.0

    if glove_txt.is_file():
        glove = load_glove_map(glove_txt, dim)
        for w, i in vocab.items():
            if w in glove:
                mat[i] = glove[w]
    else:
        print(f"GloVe file {glove_txt} not found, using randomly initialized embeddings.")

    emb = torch.tensor(mat)
    kernel_sizes = [3, 4, 5]
    model = GloveCNN(emb, args.filters, kernel_sizes)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(Xs: List[str], yt_cpu: torch.Tensor, train: bool) -> None:
        model.train(train)
        idx = torch.randperm(len(Xs), device="cpu") if train else torch.arange(len(Xs))
        for start in range(0, len(Xs), args.batch):
            sel = idx[start : start + args.batch]
            batch_x = [Xs[i] for i in sel.tolist()]
            batch_y = yt_cpu[sel].to(device)
            xb = texts_to_indices(batch_x, vocab, args.max_len).to(device)
            logits = model(xb)
            loss = loss_fn(logits, batch_y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

    train_all_t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        run_epoch(X_train, y_train_t, True)
        print(f"[timing] train epoch {epoch + 1}/{args.epochs}: {time.perf_counter() - epoch_t0:.2f}s")
    print(f"[timing] all training epochs: {time.perf_counter() - train_all_t0:.2f}s")

    model.eval()
    metrics: dict = {}
    with torch.no_grad():
        for name, Xs, yt_cpu in [("val", X_val, y_val_t), ("test", X_test, y_test_t)]:
            ev_t0 = time.perf_counter()
            pred_all: List[torch.Tensor] = []
            idx = torch.arange(len(Xs))
            for start in range(0, len(Xs), args.batch):
                sel = idx[start : start + args.batch]
                batch_x = [Xs[i] for i in sel.tolist()]
                xb = texts_to_indices(batch_x, vocab, args.max_len).to(device)
                pred_all.append((torch.sigmoid(model(xb)) >= 0.5).float().cpu())
            pred = torch.cat(pred_all, dim=0)
            yt = yt_cpu
            acc = (pred == yt).float().mean().item()
            tp = ((pred == 1) & (yt == 1)).sum().item()
            fp = ((pred == 1) & (yt == 0)).sum().item()
            fn = ((pred == 0) & (yt == 1)).sum().item()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            metrics[name] = {"accuracy": acc, "f1_fox": f1}
            print(f"[timing] evaluate {name}: {time.perf_counter() - ev_t0:.2f}s")
            print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")

    prefix = finetune_prefix(Path(__file__).stem)
    run_dir = finetune_base(proj) / prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_path = run_dir / f"{prefix}_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vocab": vocab,
            "num_filters": args.filters,
            "kernel_sizes": kernel_sizes,
            "max_len": args.max_len,
            "embedding_dim": dim,
            "glove_pretrained_path": str(glove_txt),
        },
        ck_path,
    )
    meta_py = Path(__file__).resolve()
    write_training_txt(
        run_dir / f"{prefix}.txt",
        {
            "timestamp & description": f"prefix={prefix}",
            "command line": " ".join(sys.argv),
            "pretrained embeddings": {"path": str(glove_txt), "default_location": str(default_glove_txt(proj))},
            "training hyperparameters": {
                "csv": str(args.csv.resolve()),
                "filters": args.filters,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "max_len": args.max_len,
                "kernel_sizes": kernel_sizes,
                "seed": args.seed,
                "device_arg": args.device,
            },
            "validation and test metrics": metrics,
            "saved files": {"checkpoint_pt": str(ck_path)},
        },
    )
    copy_script_source(meta_py, run_dir / f"{prefix}.py")
    print(f"[save] Finetune_Params/{prefix}/ has written {prefix}_checkpoint.pt, {prefix}.txt, and {prefix}.py")


if __name__ == "__main__":
    main()
