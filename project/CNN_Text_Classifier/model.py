import re
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        num_classes: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            features = F.relu(conv(x))
            pooled.append(F.max_pool1d(features, kernel_size=features.size(2)).squeeze(2))

        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class NewsClassifier(nn.Module):
    def __init__(self, load_weights: bool = True) -> None:
        super().__init__()

        self.classes = ["FoxNews", "NBC"]
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.max_len = 32
        self.embedding_dim = 64
        self.num_filters = 64
        self.kernel_sizes = [2, 3, 4]
        self.dropout = 0.3
        self.network = self._make_network()

        if load_weights:
            model_path = Path(__file__).resolve().parent / "model.pt"
            if model_path.exists():
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                self.load_state_dict(state)

        self.eval()

    def eval(self):
        super().eval()
        return self

    def load_state_dict(self, state, strict=True):
        self.classes = [str(label) for label in state["classes"]]
        self.vocab = {str(token): int(index) for token, index in state["vocab"].items()}
        self.max_len = int(state["max_len"])
        self.embedding_dim = int(state["embedding_dim"])
        self.num_filters = int(state["num_filters"])
        self.kernel_sizes = [int(size) for size in state["kernel_sizes"]]
        self.dropout = float(state["dropout"])
        self.network = self._make_network()
        self.network.load_state_dict(state["network"])
        self.eval()

    def _make_network(self) -> TextCNN:
        return TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
            num_classes=len(self.classes),
        )

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def _encode(self, text: str) -> torch.Tensor:
        token_ids = [self.vocab.get(token, 1) for token in self._tokens(text)[: self.max_len]]
        token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def predict(self, batch: Iterable[str]) -> List[str]:
        if isinstance(batch, str):
            batch = [batch]

        X = torch.stack([self._encode(text) for text in batch])

        with torch.no_grad():
            logits = self.network(X)
            pred_ids = torch.argmax(logits, dim=1).tolist()

        return [self.classes[i] for i in pred_ids]


def get_model():
    return NewsClassifier()


Model = NewsClassifier
