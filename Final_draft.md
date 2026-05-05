

| 模型 | 关键参数 | 准确率 | F1 | 是否加载预训练/外部权重 | 设备（GPU/CPU） |
|------|----------|--------|-----|--------------------------|-----------------|
| TF-IDF + 逻辑回归 | | | | 否：仅用 sklearn 从语料拟合 TF-IDF，不加载词向量或 Hugging Face 模型 | CPU（sklearn，脚本未使用 GPU） |
| TF-IDF + SVM | | | | 同上 | CPU |
| GloVe + MLP | | | | 是：从本地 GloVe 文本文件加载词向量（`load_glove_map`），再训练 MLP | PyTorch：`--device auto` 时若 `torch.cuda.is_available()` 则用 GPU，否则 CPU；可 `--device cpu` / `cuda` 强制 |
| GloVe + CNN | | | | 同上（GloVe 文件 + 卷积分类头） | 同上 |
| BERT | | | | 是：`transformers` 的 `AutoTokenizer.from_pretrained` 与 `AutoModelForSequenceClassification.from_pretrained`（默认如 `bert-base-uncased`） | 同上 |
| RoBERTa | | | | 是：同上 API 加载 RoBERTa 权重与分词器 | 同上 |

**简要归纳**

- **未加载大规模预训练模型**：TF-IDF + 逻辑回归、TF-IDF + SVM（无外部 `.txt`/checkpoint，`TfidfVectorizer` 在训练集上统计得到表示）。
- **加载外部词向量（非 Transformer）**：GloVe + MLP、GloVe + CNN（需预先下载 GloVe 文件路径，见各脚本参数）。
- **加载 Hugging Face 预训练 Transformer**：BERT、RoBERTa（下载/缓存由 `transformers` 管理）。
- **GPU 使用**：仅 **PyTorch 四类**（GloVe+MLP/CNN、BERT、RoBERTa）支持 CUDA；默认 **`--device auto`** 有 NVIDIA GPU 且安装的是 **CUDA 版 PyTorch** 时用 GPU，否则回退 CPU。**TF-IDF 两类始终 CPU**。
