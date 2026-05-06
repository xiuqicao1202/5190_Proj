import torch
from torch import nn
from typing import Any, Iterable, List
import tempfile
import os
from pathlib import Path


class Model(nn.Module):
    """
    Transformer-based text classifier loaded from roberta.pth bundle.
    The bundle contains: state_dict, config_dict, tokenizer_files, max_len.
    Labels: 1 → "FoxNews", 0 → "NBC"
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = None
        self._tokenizer = None
        self._max_len = 128
        self._device = torch.device("cpu")
        self._label_map = {1: "FoxNews", 0: "NBC"}
        self._load_from_bundle()

    def _load_from_bundle(self) -> None:
        bundle_path = Path(__file__).parent / "roberta.pth"
        if not bundle_path.exists():
            raise FileNotFoundError(f"roberta.pth not found at {bundle_path}")

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        state_dict = bundle["state_dict"]
        config_dict = bundle["config_dict"]
        tokenizer_files = bundle["tokenizer_files"]  # dict[str, bytes]
        self._max_len = bundle.get("max_len", 128)

        # ── Reconstruct tokenizer from bundled file bytes ──
        with tempfile.TemporaryDirectory() as tok_tmp:
            tok_dir = Path(tok_tmp)
            for rel_path, file_bytes in tokenizer_files.items():
                dest = tok_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(file_bytes)
            self._tokenizer = self._load_tokenizer(tok_dir, config_dict)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ── Reconstruct model from config_dict + state_dict ──
        self._model = self._build_model(config_dict, num_labels=2)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    @staticmethod
    def _load_tokenizer(tok_dir: Path, config_dict: dict):
        """Load tokenizer using slow (Python) backend to avoid fast-tokenizer issues."""
        from transformers import (
            BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AutoTokenizer
        )
        model_type = config_dict.get("model_type", "")
        s = str(tok_dir)
        if model_type == "bert":
            return BertTokenizer.from_pretrained(s)
        if model_type == "roberta":
            return RobertaTokenizer.from_pretrained(s)
        if model_type == "distilbert":
            return DistilBertTokenizer.from_pretrained(s)
        return AutoTokenizer.from_pretrained(s, use_fast=False)

    @staticmethod
    def _build_model(config_dict: dict, num_labels: int):
        """Reconstruct the HuggingFace classification model from its config dict."""
        from transformers import (
            BertConfig, BertForSequenceClassification,
            RobertaConfig, RobertaForSequenceClassification,
            DistilBertConfig, DistilBertForSequenceClassification,
            AutoConfig, AutoModelForSequenceClassification,
        )
        model_type = config_dict.get("model_type", "")
        if model_type == "bert":
            cfg = BertConfig(**{**config_dict, "num_labels": num_labels})
            return BertForSequenceClassification(cfg)
        if model_type == "roberta":
            cfg = RobertaConfig(**{**config_dict, "num_labels": num_labels})
            return RobertaForSequenceClassification(cfg)
        if model_type == "distilbert":
            cfg = DistilBertConfig(**{**config_dict, "num_labels": num_labels})
            return DistilBertForSequenceClassification(cfg)
        # Fallback: let AutoConfig handle it
        cfg = AutoConfig.for_model(model_type, **{**config_dict, "num_labels": num_labels})
        return AutoModelForSequenceClassification.from_config(cfg)

    def eval(self) -> "Model":
        if self._model is not None:
            self._model.eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Args:
            batch: Iterable of text strings (pseudo-titles or headlines),
                   as produced by preprocess.py.
        Returns:
            List of label strings: "FoxNews" or "NBC".
        """
        texts = list(batch)
        if not texts:
            return []

        encodings = self._tokenizer(
            texts,
            truncation=True,
            max_length=self._max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self._device)
        attention_mask = encodings["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        preds = logits.argmax(dim=-1).tolist()
        return [self._label_map[p] for p in preds]


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Loads weights from roberta.pth in the same directory.
    """
    return Model()