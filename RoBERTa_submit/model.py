"""
RoBERTa 榜单模型：用于和 preprocess.py 产出的字符串输入（伪标题）匹配。
权重来自 classifier_roberta.py 训练导出：
  torch.save(model.state_dict(), ...) -> Finetune_Params/.../{prefix}_weights.pth
  并复制一份在 params_submit/roberta.pth。
权重自动加载路径：
  - 评测器: torch.load(path); model.load_state_dict(sd)
  - 本地自动加载（参见 resolve_roberta_pth）：优先选择与本文件同目录下的 roberta.pth,
    否则选 params_submit/roberta.pth。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import torch
from torch import nn

try:
    # transformers库用于加载HuggingFace的RoBERTa模型和分词器
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging
except ImportError as e:
    raise ImportError("Please install: pip install transformers") from e

# 微调后权重若直接 from_pretrained(base) 会出现检查点 structural mismatch，产生噪声日志。
# 推荐先用 from_config 初始化模型结构，再加载权重 load_state_dict，可避免 UNEXPECTED/MISSING 日志。
hf_logging.set_verbosity_error()

def _project_root() -> Path:
    # 返回当前脚本所在目录
    return Path(__file__).resolve().parent

def _hf_cache_dir() -> str:
    # HuggingFace模型缓存路径，存放在 "Pretrained_Params/huggingface" 下
    return str(_project_root() / "Pretrained_Params" / "huggingface")

def default_roberta_pth() -> Path:
    """classifier_roberta.py 导出权重的默认路径：params_submit/roberta.pth。"""
    return _project_root() / "params_submit" / "roberta.pth"

def resolve_roberta_pth() -> Optional[Path]:
    """
    自动寻找模型权重文件（用于 Model.__init__ 里）：
    1) 若本目录下（5190_Proj/roberta.pth）存在，则优先用它
    2) 否则查找 params_submit/roberta.pth（训练脚本输出路径）
    """
    root = _project_root()
    same_dir = root / "roberta.pth"
    if same_dir.is_file():
        return same_dir
    submit = root / "params_submit" / "roberta.pth"
    if submit.is_file():
        return submit
    return None

DEFAULT_MODEL_ID = "roberta-base"  # 预设使用的HuggingFace模型名
MAX_LEN = 128  # 输入文本的最大分词长度

class Model(nn.Module):
    """
    模型要求（model_template.py 约定）：
    - 必须支持无参数初始化（评测器调用）。
    - predict(batch) 需输出与 preprocess.py 产出一致的标签（'FoxNews'｜'NBC'）。
    - 支持加载PyTorch权重；权重键与 HF AutoModelForSequenceClassification 对齐，
      本类 load_state_dict 直接传递给 self.classifier。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cache = _hf_cache_dir()  # HuggingFace transformer本地缓存路径
        # 加载分词器，若无 pad_token，设为 eos_token 保证padding正常
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, cache_dir=cache)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 尝试自动寻找微调权重
        pth = resolve_roberta_pth()
        if pth is not None:
            # 如果找到权重，则先用 AutoConfig 构建模型结构，不立即加载原始预训练权重
            cfg = AutoConfig.from_pretrained(DEFAULT_MODEL_ID, cache_dir=cache)
            cfg.num_labels = 2  # 二分类问题
            self.classifier = AutoModelForSequenceClassification.from_config(cfg)
            try:
                # 优先使用新版本 PyTorch 的 weights_only 参数，只载入权重层字典
                sd = torch.load(pth, map_location="cpu", weights_only=True)
            except TypeError:
                # 兼容PyTorch老版本；无 weights_only 关键字时退回常规加载
                sd = torch.load(pth, map_location="cpu")
            # 加载权重到分类器
            self.classifier.load_state_dict(sd, strict=True)
        else:
            # 若无微调权重，则加载HuggingFace公开的roberta-base，输出2分类头
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                DEFAULT_MODEL_ID, num_labels=2, cache_dir=cache
            )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # 直接将state_dict交给self.classifier（HF模型）来加载模型参数
        return self.classifier.load_state_dict(state_dict, strict=strict)

    def eval(self) -> "Model":
        # 设置为eval推理模式，冻结Dropout等
        super().eval()
        self.classifier.eval()
        return self

    # def train(self, mode: bool = True) -> "Model":
    #     # 如果需要训练模式，可解除注释
    #     super().train(mode)
    #     self.classifier.train(mode)
    #     return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        批量推理函数：输入batch（若为空时补""），输出同长度的标签列表
        Args:
            batch: 预处理好的输入文本列表（如多条新闻标题）
        Returns:
            标签结果，顺序与输入一一对应（元素为 'FoxNews' 或 'NBC'）
        """
        # 保证 batch 不含 None 类型，将None转为空字符串
        texts = [("" if t is None else str(t)) for t in batch]
        # 获取分类器参数所在设备（支持cuda/cpu一致性）
        device = next(self.classifier.parameters()).device
        # 用tokenizer编码输入文本并pad/truncate，使之对齐到固定MAX_LEN
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",  # 返回pytorch tensor
        )
        # 将所有输入张量移动到模型相同设备
        enc = {k: v.to(device) for k, v in enc.items()}
        self.classifier.eval()  # 明确进入推理模式
        with torch.no_grad():  # 关闭梯度，节省内存加速推理
            logits = self.classifier(**enc).logits  # 得到原始输出分数（未softmax）
        pred_ids = logits.argmax(dim=-1).tolist()  # 选最大分数的类别下标
        # 训练时规定：1 = foxnews.com (输出 'FoxNews')，0 = NBC
        return ["FoxNews" if int(p) == 1 else "NBC" for p in pred_ids]


def get_model() -> Model:
    """
    工厂函数：返回一个未初始化的模型对象，供评测器等外部调用
    """
    return Model()
