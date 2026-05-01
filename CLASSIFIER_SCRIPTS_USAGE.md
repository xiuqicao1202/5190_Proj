# 分类器脚本使用说明

本文说明 `5190_Proj` 目录下六个独立实验脚本的用途、环境、数据格式与命令行用法。它们用于在本地 **训练 + 在验证集/测试集上打印 accuracy 与 F1（FoxNews 为正类）**，与课程提交用的 `model_template.py` / `preprocess_template.py` **无直接耦合**；若需上交权重，需自行在训练中保存 checkpoint 并接到评测模板。

---

## 1. 数据格式

CSV 至少包含两列：

| 列名     | 含义 |
|----------|------|
| `url`    | 文章链接；标签由脚本根据域名推断：`foxnews.com` → **FoxNews**，否则 **NBC**。 |
| `headline` | 模型使用的文本（标题或伪标题）。 |

默认路径（脚本未改 `--csv` 时）：

`Newsheadlines/url_with_headlines.csv`

---

## 2. 环境与依赖

**通用**

- Python 3.10+（建议）
- `pandas`
- `scikit-learn`
- `torch`

**仅 BERT / RoBERTa 额外需要**

```bash
pip install transformers
```

首次运行会在有网络时从 Hugging Face Hub 拉取对应预训练模型；无网时需提前把模型目录下载到本地，并用 `--model` 指向该目录。

**GloVe + MLP / CNN**

- 需要与代码维度一致的 GloVe 文本文件：**`glove.6B.100d.txt`**（100 维）。
- 下载：[GloVe - Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)，解压后把 `glove.6B.100d.txt` 放到已知路径。
- 若未提供有效 `--glove` 路径，脚本会用 **随机初始化词向量** 并打印提示，结果不代表真实 GloVe 实验。

---

## 3. 数据划分说明

所有脚本使用相同策略：

1. 先划出 **30%** 作为临时集合；
2. 在临时集合内按 **1 : 2** 拆分：约 **10% 验证**、**20% 测试**；
3. 剩余约 **70% 训练**。

分层依据：TF-IDF 脚本按字符串标签 `FoxNews`/`NBC`；神经网络脚本按 0/1 标签。

---

## 4. 各脚本用法

在 `5190_Proj` 下执行（Windows 可将 `python` 换成 `py`）。

### `run_classifier.sh`（Linux / WSL）

可用 **`./run_classifier.sh <1-6>`** 启动对应分类器（环境与直接运行各 `classifier_*.py` 相同），编号：**1** BERT，**2** GloVe+CNN，**3** GloVe+MLP，**4** RoBERTa，**5** TF-IDF+逻辑回归，**6** TF-IDF+SVM。第一个参数之后的选项会原样传给对应 Python 脚本（例如 `./run_classifier.sh 5 --csv data.csv`）。

**5 与 6 可正常跑通**（默认 CSV `Newsheadlines/url_with_headlines.csv`；需已修复空 `headline` 的 `fillna` 与脚本的 LF 换行）。下面是一次本机运行打印的 **accuracy** 与 **f1_fox**（FoxNews 为正类的二元 F1），**仅供参考**，复现时可能因环境略有差异：

| 命令 | 验证集 | 测试集 |
|------|--------|--------|
| `./run_classifier.sh 5` | accuracy **0.7763**，f1_fox **0.8046** | accuracy **0.7848**，f1_fox **0.8071** |
| `./run_classifier.sh 6` | accuracy **0.7947**，f1_fox **0.8152** | accuracy **0.8123**，f1_fox **0.8258** |

---

### 4.1 TF-IDF + 逻辑回归 — `classifier_tfidf_logistic_regression.py`

```bash
python classifier_tfidf_logistic_regression.py
python classifier_tfidf_logistic_regression.py --csv path/to/data.csv --max-features 20000 --C 1.0 --seed 42
# Linux / WSL 快捷（等同上行）：
./run_classifier.sh 5
./run_classifier.sh 5 --csv path/to/data.csv --max-features 20000 --C 1.0 --seed 42
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--csv` | `Newsheadlines/url_with_headlines.csv` | 数据路径 |
| `--max-features` | `20000` | TF-IDF 最大特征数（对应报告中的维数上限之一） |
| `--C` | `1.0` | 逻辑回归正则强度的倒数（越大越弱正则） |
| `--seed` | `42` | 划分随机种子 |

无预训练文件；向量器与分类器均在训练集上 `fit`。

---

### 4.2 TF-IDF + SVM — `classifier_tfidf_svm.py`

```bash
python3 classifier_tfidf_svm.py
python3 classifier_tfidf_svm.py --csv path/to/data.csv --max-features 20000 --C 1.0
# Linux / WSL 快捷：
./run_classifier.sh 6
./run_classifier.sh 6 --csv path/to/data.csv --max-features 20000 --C 1.0
```

参数含义与上表相同；分类器为 `LinearSVC`（hinge 损失）。

---

### 4.3 GloVe + MLP — `classifier_glove_mlp.py`

```bash
python3 classifier_glove_mlp.py
python3 classifier_glove_mlp.py --glove D:\data\glove.6B.100d.txt --epochs 15 --hidden 128
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--csv` | 同上 | 数据路径 |
| `--glove` | `glove.6B.100d.txt`（当前工作目录下相对路径） | GloVe 100d 词向量文件 |
| `--hidden` | `128` | MLP 隐层宽度 |
| `--epochs` | `15` | 训练轮数 |
| `--batch` | `64` | 批大小 |
| `--lr` | `0.001` | Adam 学习率 |
| `--max-len` | `64` | 每条样本截断/填充到的词数上限 |
| `--seed` | `42` | 随机种子 |

词嵌入由 GloVe 初始化，脚本中为 **可训练**（非冻结）。

---

### 4.4 GloVe + CNN — `classifier_glove_cnn.py`

```bash
python3 classifier_glove_cnn.py
python3 classifier_glove_cnn.py --glove D:\data\glove.6B.100d.txt --filters 100
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--filters` | `100` | 每个卷积核尺寸下的滤波器个数 |
| 其余 | 同 MLP | `--csv`、`--glove`、`--epochs`、`--batch`、`--lr`、`--max-len`、`--seed` |

卷积核尺寸在代码中固定为多尺度 `[3, 4, 5]`。

---

### 4.5 BERT — `classifier_bert.py`

```bash
python3 classifier_bert.py
python3 classifier_bert.py --model bert-base-uncased --epochs 3 --batch 16 --lr 2e-5
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--csv` | 同上 | 数据路径 |
| `--model` | `bert-base-uncased` | Hugging Face 模型名或 **本地模型目录** |
| `--epochs` | `3` | 微调轮数 |
| `--batch` | `16` | 批大小 |
| `--lr` | `2e-5` | AdamW 学习率 |
| `--max-len` | `128` | 分词最大长度 |
| `--seed` | `42` | 随机种子 |

**预训练参数**：由 `transformers` 在 `from_pretrained` 时自动下载或从 `--model` 本地路径加载；分类头为 2 类，在您的数据上微调。

---

### 4.6 RoBERTa — `classifier_roberta.py`

```bash
python3 classifier_roberta.py
python3 classifier_roberta.py --model roberta-base --epochs 3 --batch 16
```

参数与 BERT 相同；默认 `--model` 为 `roberta-base`。

---

## 5. 运行输出

每个脚本结束时会在标准输出打印两行类似：

```text
val: accuracy=0.xxxx f1_fox=0.xxxx
test: accuracy=0.xxxx f1_fox=0.xxxx
```

其中 `f1_fox` 以 **FoxNews 为正类** 计算二元 F1。

---

## 6. 与课程提交物的关系

| 项目 | 说明 |
|------|------|
| 这些 `classifier_*.py` | 本地实验与报告数值复现；不替换 `model_template.py`。 |
| `model_template.py` | 评测器要求 `get_model()`、`predict()` 等接口；若最终模型是 BERT，需写包装类并在 `predict` 里做 tokenizer + `forward`。 |
| 权重保存 | 当前脚本 **未** 写入 `torch.save`；需要时请在训练循环后自行保存 `state_dict` 或 transformers 的 `save_pretrained`。 |

---

## 7. 常见问题

**Q：`--glove` 已指定仍像随机效果？**  
检查文件是否为 **100 维**、路径是否可读；工作目录不同时应使用 **绝对路径**。

**Q：BERT 下载慢或失败？**  
配置 Hugging Face 镜像或离线缓存；或将模型完整克隆到本地， `--model` 指向该文件夹。

**Q：显存不足？**  
减小 `--batch` 或 `--max-len`；CPU 可跑但较慢。

---

*文档对应仓库内六个脚本；若脚本内参数默认值变更，请以 `python3 script.py -h` 为准。*
