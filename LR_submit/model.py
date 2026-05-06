"""
TF-IDF + Logistic Regression 评测包装：载入 `classifier_tfidf_logistic_regression.py`
导出的 sklearn Pipeline（joblib）。

查找顺序（与训练脚本 `LR_SUBMIT_PIPELINE_NAME` 一致）：
1) 本文件同目录 `tfidf_lr_pipeline.joblib`（训练后会 mirror 到这里）
2) 项目根下 `LR_submit/tfidf_lr_pipeline.joblib`

请先运行：
  python classifier_tfidf_logistic_regression.py ...
以生成权重文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional

import joblib
from torch import nn

# 必须与 classifier_tfidf_logistic_regression.py 中 LR_SUBMIT_PIPELINE_NAME 相同
LR_SUBMIT_PIPELINE_NAME = "tfidf_lr_pipeline.joblib"


def resolve_lr_pipeline_path() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    same_dir = here / LR_SUBMIT_PIPELINE_NAME
    if same_dir.is_file():
        return same_dir
    root_lr = here.parent / "LR_submit" / LR_SUBMIT_PIPELINE_NAME
    if root_lr.is_file():
        return root_lr
    return None


class Model(nn.Module):
    """
    无 Torch 参数的薄包装：`predict` 走 sklearn Pipeline，与 preprocess.py 输出的字符串序列对齐。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        p = resolve_lr_pipeline_path()
        if p is None:
            raise FileNotFoundError(
                "未找到 TF-IDF LR pipeline。请先训练并生成以下之一：\n"
                f"  - {Path(__file__).resolve().parent / LR_SUBMIT_PIPELINE_NAME}\n"
                f"  - {Path(__file__).resolve().parent.parent / 'LR_submit' / LR_SUBMIT_PIPELINE_NAME}"
            )
        self._pipeline_path = p
        self._pipeline = joblib.load(p)

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [("" if x is None else str(x)) for x in batch]
        out = self._pipeline.predict(texts)
        return [str(x) for x in out]


def get_model() -> Model:
    return Model()
