import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch


class NewsClassifier:
    def __init__(self, load_weights: bool = True) -> None:
        self.num_features = 5000
        self.classes = ["FoxNews", "NBC"]
        self.idf = torch.ones(self.num_features)
        self.coef = torch.zeros(len(self.classes), self.num_features)
        self.intercept = torch.zeros(len(self.classes))

        if load_weights:
            model_path = Path(__file__).resolve().parent / "model.pt"
            if model_path.exists():
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                self.load_state_dict(state)

        self.eval()

    def eval(self):
        return self

    def load_state_dict(self, state, strict=True):
        self.num_features = state["num_features"]
        self.classes = [str(label) for label in state["classes"]]
        self.idf = state["idf"]
        self.coef = state["coef"]
        self.intercept = state["intercept"]
        self.eval()

    def _hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.num_features

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def _tfidf(self, text: str) -> torch.Tensor:
        tokens = self._tokens(text)

        features = []
        for token in tokens:
            features.append(self._hash("word:" + token))

        for i in range(len(tokens) - 1):
            features.append(self._hash("bigram:" + tokens[i] + "_" + tokens[i + 1]))

        x = torch.zeros(self.num_features)
        if not features:
            return x

        counts = Counter(features)
        for index, count in counts.items():
            tf = count / len(features)
            x[index] = tf * self.idf[index]

        norm = torch.linalg.vector_norm(x)
        if norm > 0:
            x = x / norm

        return x

    def predict(self, batch: Iterable[str]) -> List[str]:
        if isinstance(batch, str):
            batch = [batch]

        X = torch.stack([self._tfidf(text) for text in batch])

        with torch.no_grad():
            scores = X @ self.coef.T + self.intercept
            pred_ids = torch.argmax(scores, dim=1).tolist()

        return [self.classes[i] for i in pred_ids]


def get_model():
    return NewsClassifier()


Model = NewsClassifier
