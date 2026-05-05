# 本次改动说明

## 1. GloVe + CNN / GloVe + MLP

- **标签与划分**：`train_test_split` 改为与 BERT 一致——用 **`List[int]` 标签** + **`stratify=np.array(labels)`**，再在 CPU 上建 `torch.tensor(y_train, ...)`，避免 `sklearn` 返回 numpy 与 `torch` 索引混用导致的问题。
- **GPU**：增加 **`--device auto|cuda|cpu`** 与 **`resolve_torch_device`**；**模型、`input_ids`、batch 标签** 在训练与按 batch 验证时均搬到选定设备；`auto` 在有 CUDA 时用 GPU。
- **其它**：补充耗时与数据规模打印；验证集/测试集按 batch 前向，指标在 CPU 上与标签对齐。

## 2. RoBERTa

- **Tokenizer**：若 **`pad_token` 未设置**，则设为 **`eos_token`**，避免 `padding="max_length"` 时缺 pad。
- **DataLoader**：**`num_workers=0`**，减少 Windows/WSL 下多进程问题；训练仍通过 **`pin_memory` + `.to(device)`** 使用 GPU（与此前逻辑一致）。

## 3. `run_classifier.sh`

- **选项 2、3**：与 1、4 相同，默认插入 **`--device "${CLASSIFIER_DEVICE:-auto}"`**，便于 GloVe 脚本默认走 GPU；可用环境变量或行末 **`--device cpu`** 覆盖。
- **用法**：在 bash 下执行 **`bash run_classifier.sh <1-6> [额外参数...]`**（或已 `chmod +x` 时 **`./run_classifier.sh <1-6>`**），数字 **1–6** 分别对应 BERT、GloVe+CNN、GloVe+MLP、RoBERTa、TF-IDF+LR、TF-IDF+SVM，均可直接跑通对应 Python 脚本。
