# Finetune_Params / Pretrained_Params — 目录分工原理（简述）

## 总体思路

项目在 `param_storage.py` 里约定两套本地目录：**输入侧**放「通用、可复用的预训练资源」，**输出侧**放「某次训练跑出来的结果」。这样职责清晰：预训练权重可被多个实验共享；微调产物按运行隔离，互不覆盖。

## Pretrained_Params（预训练参数 / 缓存）

- **含义**：从外部获取、**跨运行复用**的资源，例如 Hugging Face 下载的基座模型缓存（`Pretrained_Params/huggingface`）、GloVe 词向量（`Pretrained_Params/glove/`）等。
- **特点**：体积大、下载成本高；一般不随单次实验频繁变动；脚本会在需要时自动创建子目录或触发下载（如 GloVe）。
- **为何常忽略进 Git**：二进制/大文件多，且可通过脚本或官方链接重新获得，不适合纳入版本库。

## Finetune_Params（微调产物）

- **含义**：每次训练运行结束后写入的**实验输出**，由时间戳 + 脚本名等组成前缀目录（如 `{时间戳}_classifier_bert`）。
- **典型内容**：检查点（`.pt`）、复制的一份训练脚本（`.py`）、训练/评估摘要文本（`.txt`）；BERT/RoBERTa 等还会写出 `{前缀}_finetuned_hf` 等 Hugging Face 兼容目录。
- **特点**：随运行累积、可能很大；与「某一次」超参与数据划分绑定，属于**生成物**而非源码。
- **为何忽略进 Git**：体量与次数线性增长，且可由代码与数据重新训练复现；纳入仓库会拖慢 clone 并易产生 merge 噪声。

## 与 `.gitignore` 的关系

将 `Finetune_Params/` 与 `Pretrained_Params/` 写入 `.gitignore`，是让 Git **只跟踪代码与配置**，而把本地大文件与运行缓存留给各开发者机器自行管理；已在索引中的路径需配合 `git rm --cached` 才能真正停止跟踪。
