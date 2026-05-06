"""
诊断：确认 roberta.pth 可被 transformers 从本地 bundle 还原（不访问 HuggingFace 缓存）。
从 5190_Proj/RoBERTa_submit/ 目录运行：  python verify_bundle.py
"""
from pathlib import Path

import torch

from model import Model, _build_hf_from_bundle, _is_bundle, _safe_torch_load

bundle_path = Path(__file__).resolve().parent / "roberta.pth"
bundle = _safe_torch_load(bundle_path)

if not _is_bundle(bundle):
    raise SystemExit(
        "roberta.pth 缺少 tokenizer_files。请运行 classifier_roberta.py 重新导出 bundle。"
    )

sd = bundle["state_dict"]
cfg = bundle["config_dict"]

print("=== 1. state_dict 键 ===")
keys = list(sd.keys())
print(f"  total keys: {len(keys)}")
print(f"  first 5: {keys[:5]}")
print(f"  last 5:  {keys[-5:]}")

print("\n=== 2. config 关键字段 ===")
for k in [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "type_vocab_size",
    "layer_norm_eps",
    "pad_token_id",
    "num_labels",
    "hidden_act",
]:
    print(f"  {k}: {cfg.get(k, 'MISSING')}")

print("\n=== 3. tokenizer 文件 ===")
for name in sorted(bundle["tokenizer_files"].keys()):
    print(f"  {name}")

print("\n=== 4. Model 前向（与本目录 model.Model 一致）===")
m = Model()
sample = ["trump immigration plan"]
pred = m.predict(sample)
print(f"  predict({sample!r}) -> {pred}")

import tempfile

_tmp = tempfile.TemporaryDirectory(prefix="verify_roberta_")
tok, clf, max_len = _build_hf_from_bundle(bundle, tokenizer_tmp=_tmp)
enc = tok(
    sample,
    truncation=True,
    max_length=max_len,
    padding="max_length",
    return_tensors="pt",
)
clf.eval()
with torch.no_grad():
    logits = clf(**enc).logits
print(f"  logits: {logits}")
