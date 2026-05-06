"""
RoBERTa leaderboard model: matches preprocess.py string inputs (pseudo titles).
Weights from classifier_roberta.py:
  torch.save(model.state_dict(), ...) -> Finetune_Params/.../{prefix}_weights.pth
  and a copy at params_submit/roberta.pth.
Load paths:
  - Evaluator: torch.load(path); model.load_state_dict(sd)
  - Local auto-load (see resolve_roberta_pth): prefers roberta.pth next to this file,
    otherwise params_submit/roberta.pth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import torch
from torch import nn

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    raise ImportError("Please install: pip install transformers") from e


def _project_root() -> Path:
    # This file lives in 5190_Proj/model_RoBERTa.py
    return Path(__file__).resolve().parent


def _hf_cache_dir() -> str:
    return str(_project_root() / "Pretrained_Params" / "huggingface")


def default_roberta_pth() -> Path:
    """Path used by classifier_roberta.py copy: params_submit/roberta.pth."""
    return _project_root() / "params_submit" / "roberta.pth"


def resolve_roberta_pth() -> Optional[Path]:
    """
    Weight file for auto-load in Model.__init__:
    1) roberta.pth in the same directory as this module (5190_Proj/ next to model_RoBERTa.py)
    2) else params_submit/roberta.pth (training script output)
    """
    root = _project_root()
    same_dir = root / "roberta.pth"
    if same_dir.is_file():
        return same_dir
    submit = root / "params_submit" / "roberta.pth"
    if submit.is_file():
        return submit
    return None


DEFAULT_MODEL_ID = "roberta-base"
MAX_LEN = 128


class Model(nn.Module):
    """
    Requirements (model_template.py):
    - No-arg __init__ for evaluator.
    - predict(batch) returns labels aligned with preprocess.py outputs ('FoxNews' | 'NBC').
    - PyTorch weights: evaluator loads state_dict into this module; keys match HF
      AutoModelForSequenceClassification, so load_state_dict forwards to `classifier`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cache = _hf_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, cache_dir=cache)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_MODEL_ID, num_labels=2, cache_dir=cache
        )
        pth = resolve_roberta_pth()
        if pth is not None:
            # classifier_roberta.py: torch.save(model.state_dict(), ...)
            sd = torch.load(pth, map_location="cpu")
            self.classifier.load_state_dict(sd, strict=True)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.classifier.load_state_dict(state_dict, strict=strict)

    def eval(self) -> "Model":
        super().eval()
        self.classifier.eval()
        return self

    def train(self, mode: bool = True) -> "Model":
        super().train(mode)
        self.classifier.train(mode)
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [("" if t is None else str(t)) for t in batch]
        device = next(self.classifier.parameters()).device
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(**enc).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        # Training script: 1 = foxnews.com (FoxNews), 0 = NBC
        return ["FoxNews" if int(p) == 1 else "NBC" for p in pred_ids]


def get_model() -> Model:
    return Model()
