"""
GloVe + MLP（报告 2.3.1）
词序列嵌入 {e_i}，平均池化 h0 = (1/T)Σ e_i，MLP：h1 = ReLU(W1 h0 + b1)，ˆy = σ(W2 h1 + b2)。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def label_from_url(url: str) -> int:
    return 1 if "foxnews.com" in url.lower() else 0


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


class GloveMLP(nn.Module):
    def __init__(self, embedding: torch.Tensor, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=0)
        d = embedding.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids != 0
        e = self.embed(token_ids)
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        h = (e * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.mlp(h).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).parent / "Newsheadlines" / "url_with_headlines.csv")
    ap.add_argument("--glove", type=Path, default=Path("glove.6B.100d.txt"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.csv)
    texts = df["headline"].fillna("").astype(str).tolist()
    y = torch.tensor([label_from_url(str(u)) for u in df["url"].tolist()], dtype=torch.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(texts, y, test_size=0.3, random_state=args.seed, stratify=y.numpy())
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=args.seed, stratify=y_temp.numpy()
    )

    vocab = build_vocab(X_train)
    dim = 100
    n_vocab = len(vocab)
    mat = np.random.randn(n_vocab, dim).astype(np.float32) * 0.01
    mat[vocab["<pad>"]] = 0.0

    if args.glove.is_file():
        glove = load_glove_map(args.glove, dim)
        for w, i in vocab.items():
            if w in glove:
                mat[i] = glove[w]
    else:
        print(f"未找到 GloVe 文件 {args.glove}，使用随机初始化嵌入（请下载 glove.6B.100d.txt）。")

    emb = torch.tensor(mat)
    model = GloveMLP(emb, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(Xs: List[str], yt: torch.Tensor, train: bool) -> float:
        model.train(train)
        idx = torch.randperm(len(Xs)) if train else torch.arange(len(Xs))
        total_loss = 0.0
        for start in range(0, len(Xs), args.batch):
            sel = idx[start : start + args.batch]
            batch_x = [Xs[i] for i in sel.tolist()]
            batch_y = yt[sel]
            xb = texts_to_indices(batch_x, vocab, args.max_len)
            logits = model(xb)
            loss = loss_fn(logits, batch_y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += float(loss) * len(sel)
        return total_loss / max(1, len(Xs))

    for _ in range(args.epochs):
        run_epoch(X_train, y_train, True)
    model.eval()
    with torch.no_grad():
        for name, Xs, yt in [
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            xb = texts_to_indices(Xs, vocab, args.max_len)
            pred = (torch.sigmoid(model(xb)) >= 0.5).float()
            acc = (pred == yt).float().mean().item()
            tp = ((pred == 1) & (yt == 1)).sum().item()
            fp = ((pred == 1) & (yt == 0)).sum().item()
            fn = ((pred == 0) & (yt == 1)).sum().item()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            print(f"{name}: accuracy={acc:.4f} f1_fox={f1:.4f}")


if __name__ == "__main__":
    main()
