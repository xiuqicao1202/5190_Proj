from collections import Counter
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from model import NewsClassifier
from preprocess import prepare_data


LABEL_TO_ID = {"FoxNews": 0, "NBC": 1}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data_path(path: str) -> Path:
    data_path = Path(path)
    if data_path.is_absolute():
        return data_path

    local_path = Path(__file__).resolve().parent / data_path
    if local_path.exists():
        return local_path

    baseline_path = Path(__file__).resolve().parent.parent / "TF-IDF_and_Logistic_Regression" / data_path
    if baseline_path.exists():
        return baseline_path

    return local_path


def build_vocab(model: NewsClassifier, texts: Iterable[str], min_count: int, max_vocab: int) -> dict:
    counts = Counter()
    for text in texts:
        counts.update(model._tokens(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common(max_vocab - len(vocab)):
        if count >= min_count:
            vocab[token] = len(vocab)

    return vocab


def build_matrix(model: NewsClassifier, texts: List[str]) -> torch.Tensor:
    return torch.stack([model._encode(text) for text in texts])


def evaluate(model: NewsClassifier, texts: List[str], labels: List[str]) -> Tuple[float, float]:
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return accuracy, f1


def train_model(
    X_train: List[str],
    y_train: List[str],
    X_val: List[str] | None,
    y_val: List[str] | None,
    max_len: int,
    max_vocab: int,
    min_count: int,
    embedding_dim: int,
    num_filters: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> NewsClassifier:
    device = choose_device()
    print(f"Using device: {device}")

    model = NewsClassifier(load_weights=False)
    model.max_len = max_len
    model.embedding_dim = embedding_dim
    model.num_filters = num_filters
    model.dropout = dropout
    model.vocab = build_vocab(model, X_train, min_count=min_count, max_vocab=max_vocab)
    model.network = model._make_network()

    X_tensor = build_matrix(model, X_train).to(device)
    y_tensor = torch.tensor([LABEL_TO_ID[label] for label in y_train], dtype=torch.long).to(device)

    counts = Counter(y_train)
    class_weights = torch.tensor(
        [len(y_train) / (len(model.classes) * counts[label]) for label in model.classes],
        dtype=torch.float32,
    ).to(device)

    model.network.to(device)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        model.network.train()
        optimizer.zero_grad()
        logits = model.network(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
        optimizer.step()

        message = f"Epoch {epoch + 1}/{epochs} train_loss={loss.item():.4f}"

        if X_val is not None and y_val is not None:
            model.network.cpu()
            model.eval()
            accuracy, f1 = evaluate(model, X_val, y_val)
            model.network.to(device)
            message += f" val_accuracy={accuracy:.4f} val_f1={f1:.4f}"

            if f1 > best_f1:
                best_f1 = f1
                best_state = deepcopy(model.network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= patience:
                print(message)
                print(f"Early stopping after {epoch + 1} epochs")
                break

        print(message)

    model.network.cpu()
    if best_state is not None:
        model.network.load_state_dict(best_state)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="url_dataset_headlines.csv")
    parser.add_argument("--output", default="model.pt")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=5000)
    parser.add_argument("--num-filters", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    X, y = prepare_data(str(data_path))

    if args.final:
        model = train_model(
            X,
            y,
            None,
            None,
            max_len=args.max_len,
            max_vocab=args.max_vocab,
            min_count=args.min_count,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            max_len=args.max_len,
            max_vocab=args.max_vocab,
            min_count=args.min_count,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    torch.save(
        {
            "classes": model.classes,
            "vocab": model.vocab,
            "max_len": model.max_len,
            "embedding_dim": model.embedding_dim,
            "num_filters": model.num_filters,
            "kernel_sizes": model.kernel_sizes,
            "dropout": model.dropout,
            "network": model.network.state_dict(),
        },
        output_path,
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
