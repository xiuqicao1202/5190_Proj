# 本次改动说明

## 1. `run_classifier.sh`（WSL / Linux）

- **问题**：脚本在 Windows 下保存为 CRLF 换行时，shebang 会变成 `#!/usr/bin/env bash\r`，在 WSL 中报错：`bash\r: No such file or directory`。
- **处理**：将文件换行统一为 LF（Unix 风格），便于在 `./run_classifier.sh` 下直接执行。

## 2. 空标题 / CSV 中的 NaN 与 sklearn

- **问题**：默认数据 `Newsheadlines/url_with_headlines.csv` 中部分行 `headline` 为空，pandas 读成 `NaN`。`TfidfVectorizer` 不接受 `np.nan`，运行 TF-IDF 相关脚本时在 `fit` 阶段报错：`np.nan is an invalid document`。
- **处理**：在转为字符串列表前对文本列使用 **`fillna("")`**，将缺失标题视为空字符串。
- **涉及文件**：
  - `classifier_tfidf_logistic_regression.py`、`classifier_tfidf_svm.py`：`processed_title` 与 `headline` 两路读取。
  - `classifier_bert.py`、`classifier_roberta.py`、`classifier_glove_cnn.py`、`classifier_glove_mlp.py`：`headline` 列同样处理，避免同一份 CSV 在其他实验脚本中复现同类错误。
