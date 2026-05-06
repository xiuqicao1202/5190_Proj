# 本次改动说明

## 1. `run_roberta.sh`（RoBERTa 超参数实验脚本）

新增独立的批量实验脚本，用于系统化对比 RoBERTa 不同超参数配置的效果。

- **实验设计**：共 4 组（编号 1–4），基线（编号 0）已注释保留以供参考：

  | 编号 | 说明 | 关键参数变化 |
  |------|------|-------------|
  | 0（注释） | 基线 | `roberta-base`, lr=2e-5, batch=16, epochs=3, max-len=128 |
  | 1 | 更多 epochs | epochs=**5**，其余同基线 |
  | 2 | 更高学习率 | lr=**3e-5**，epochs=3 |
  | 3 | 短文本优化 | max-len=**64**, batch=**32** |
  | 4 | 更复杂模型 | model=**roberta-large**, batch=**8**, lr=**1e-5** |

- **Python 解释器检测**：与 `run_classifier.sh` 一致，优先使用环境变量 `$PYTHON`，回退至 `python3` / `python`；同样以 `set -euo pipefail` 保证任意步骤失败即终止。
- **结果归档**：每次 `classifier_roberta.py` 运行会在 `Finetune_Params/` 下写出带时间戳的子目录（检查点、脚本副本、评估摘要），各组实验互不覆盖。
- **用法**：`bash run_roberta.sh`（无需参数，顺序执行所有未注释实验）。

## 2. `run_classifier.sh`（统一启动脚本）

在前序提交的基础上，当前版本功能已完整，在此做一次全量说明以便后续参考：

- **统一入口**：单脚本覆盖全部 6 种分类器，通过数字参数 `1–6` 或 `all` 选择：

  | 编号 | 对应脚本 | 额外支持参数 |
  |------|----------|-------------|
  | 1 | `classifier_bert.py` | `--csv --model --epochs --batch --lr --max-len --seed --device` |
  | 2 | `classifier_glove_cnn.py` | `--csv --glove --filters --epochs … --device [--skip-glove-download]` |
  | 3 | `classifier_glove_mlp.py` | `--csv --glove --hidden --epochs … --device [--skip-glove-download]` |
  | 4 | `classifier_roberta.py` | `--csv --model --epochs --batch --lr --max-len --seed --device` |
  | 5 | `classifier_tfidf_logistic_regression.py` | `--csv --max-features --C --seed` |
  | 6 | `classifier_tfidf_svm.py` | `--csv --max-features --C --seed` |

- **设备默认**：选项 1–4 均自动追加 `--device "${CLASSIFIER_DEVICE:-auto}"`；可通过环境变量 `CLASSIFIER_DEVICE=cpu` 全局覆盖，也可在命令末尾追加 `--device cpu` 逐次覆盖。
- **`all` 模式**：顺序执行 1→6，额外参数会透传给每个子脚本；任意一步失败立即中止。
- **Shell 兼容**：检测到非 bash 解释器（如 Ubuntu 下 `sh` 实为 `dash`）时，自动 `exec` 切换至 bash，规避 `pipefail` 不兼容问题；CRLF 已统一为 LF。
- **用法示例**：
  ```bash
  bash run_classifier.sh 2               # GloVe+CNN，默认 GPU
  bash run_classifier.sh 4 --epochs 5    # RoBERTa，指定 epochs
  CLASSIFIER_DEVICE=cpu bash run_classifier.sh all  # 全部在 CPU 上跑
  ```
