"""
RoBERTa submission model.
Loads fine-tuned weights and tokenizer from params_submit/roberta_model.pt and provides predict().
"""
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    raise SystemExit("Please install: pip install transformers") from e

_MODEL_PT = Path(__file__).resolve().parent / "roberta_model.pt"


class NewsClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if not _MODEL_PT.exists():
            raise FileNotFoundError(
                f"roberta_model.pt not found, please run classifier_roberta.py to generate parameters: {_MODEL_PT}"
            )

        # Check if CUDA is available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(_MODEL_PT, map_location="cpu", weights_only=False)
        self.classes: List[str] = ckpt["classes"]
        self._max_len: int = ckpt["max_len"]

        # Rebuild model architecture and load weights
        cfg_dict: dict = ckpt["config"]
        config = AutoConfig.for_model(
            cfg_dict["model_type"],
            **{k: v for k, v in cfg_dict.items() if k != "model_type"},
        )
        self._roberta = AutoModelForSequenceClassification.from_config(config)
        self._roberta.load_state_dict(ckpt["state_dict"])
        self._roberta.eval()
        self._roberta.to(self._device)  # Move model to CUDA if available

        # Rebuild tokenizer from embedded files in .pt
        with tempfile.TemporaryDirectory() as tmp:
            for fname, data in ckpt["tokenizer_files"].items():
                with open(os.path.join(tmp, fname), "wb") as f:
                    f.write(data)
            self._tokenizer = AutoTokenizer.from_pretrained(tmp)

    def eval(self):
        super().eval()
        if hasattr(self, "_roberta"):
            self._roberta.eval()
        return self

    def predict(self, batch: Iterable[str]) -> List[str]:
        if isinstance(batch, str):
            batch = [batch]
        batch = list(batch)

        enc = self._tokenizer(
            batch,
            truncation=True,
            max_length=self._max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Move input to the same device as the model
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            pred_ids = torch.argmax(logits, dim=1).tolist()

        return [self.classes[i] for i in pred_ids]


def get_model():
    return NewsClassifier()


Model = NewsClassifier
