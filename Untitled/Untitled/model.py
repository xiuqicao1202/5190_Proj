import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn


class NewsClassifier(nn.Module):
    def __init__(self, load_weights: bool = True) -> None:
        super().__init__()

        self.num_features = 8192
        self.classes = ["FoxNews", "NBC"]
        self.idf = torch.ones(self.num_features)
        self.classifier = nn.Linear(self.num_features, len(self.classes))

        if load_weights:
            model_path = Path(__file__).resolve().parent / "model.pt"
            if model_path.exists():
                state = torch.load(model_path, map_location="cpu")
                self.load_state_dict(state)

        self.eval()

    def eval(self):
        super().eval()
        return self

    def load_state_dict(self, state, strict=True):
        self.num_features = state["num_features"]
        self.classes = state["classes"]
        self.idf = state["idf"]

        self.classifier = nn.Linear(self.num_features, len(self.classes))
        self.classifier.load_state_dict(state["classifier"])
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
        print("batch ", batch)
        print("type of batch ", type(batch))
        '''
        'indiana police make announcement unsolved 2017 delphi murders 2 teenag', 
        'gaza burials dead israel hamas war', 
        'kentucky derby how bet horses home', 
        'fbi releases declassified document saudi 9 11 links after biden', 
        'iran attack israel israel war gaza hamas drone missile', 
        'us israeli citizen kidnapped oct 7 confirmed dead idf says',
        'hurricane milton storm surge florida', 
        'marco rubio cuba fight socialist government leaders evil incompetent', 
        'prosecutors trump sentencing delay hush money election', 
        'walzs handling blm riots strict covid rules under microscope after harris vp pick', 
        'ny ballot measure abortion transgender culture wars']
        
        '''
        
        if isinstance(batch, str):
            batch = [batch]

        X = torch.stack([self._tfidf(text) for text in batch])

        with torch.no_grad():
            logits = self.classifier(X)
            pred_ids = torch.argmax(logits, dim=1).tolist()

        print("result ", [self.classes[i] for i in pred_ids])
        '''FoxNews', 'NBC'''
        return [self.classes[i] for i in pred_ids]


def get_model():
    return NewsClassifier()


Model = NewsClassifier
