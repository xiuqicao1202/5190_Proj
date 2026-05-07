import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn


class NewsClassifier(nn.Module):
    def __init__(self, load_weights: bool = True) -> None:
        super().__init__()

        self.classes = ["FoxNews", "NBC"]
        self.vocab = {}
        self.num_features = 20000
        self.hidden_dims = [256, 128]
        self.dropout = 0.3
        self.classifier = self._make_classifier()

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
        self.num_features = int(state["num_features"])
        self.hidden_dims = [int(dim) for dim in state["hidden_dims"]]
        self.dropout = float(state["dropout"])
        self.classifier = self._make_classifier()
        self.classifier.load_state_dict(state["classifier"])
        self.eval()

    def _make_classifier(self) -> nn.Sequential:
        layers = []
        input_dim = self.num_features

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, len(self.classes)))
        return nn.Sequential(*layers)

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def _vectorize(self, text: str) -> torch.Tensor:
        x = torch.zeros(self.num_features)
        tokens = [token for token in self._tokens(text) if token in self.vocab]
        if not tokens:
            return x

        counts = Counter(tokens)
        for token, count in counts.items():
            x[self.vocab[token]] = count / len(tokens)

        norm = torch.linalg.vector_norm(x)
        if norm > 0:
            x = x / norm

        return x

    def predict(self, batch: Iterable[str]) -> List[str]:
        if isinstance(batch, str):
            batch = [batch]

        X = torch.stack([self._vectorize(text) for text in batch])

        with torch.no_grad():
            logits = self.classifier(X)
            pred_ids = torch.argmax(logits, dim=1).tolist()

        return [self.classes[i] for i in pred_ids]


def get_model():
    return NewsClassifier()


Model = NewsClassifier
