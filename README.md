# IMDB 影评情感分类

基于 [Arian Latifi](https://github.com/ArianLatifi/Sentiment-Classification-PyTorch) 的 PyTorch 项目改进，提供本地与 Google Colab 版本，使用 LSTM 将影评分为正面或负面。

维护者：[木子白白白](https://github.com/hnlyzxf)

---

## 目录

- [项目概览](#项目概览)
- [IMDB 情感分析](#imdb-情感分析)
  - [数据集](#数据集)
  - [预处理流水线](#预处理流水线)
  - [模型结构](#模型结构)
  - [训练配置](#训练配置)
  - [实验结果](#实验结果)
- [NLP 预处理流水线](#nlp-预处理流水线)
- [安装](#安装)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [关键词](#关键词)
- [项目来源与维护](#项目来源与维护)

---

## 项目概览

| 项目 | 说明 |
|---|---|
| **数据** | IMDB 影评 50,000 条，正负样本各半 |
| **模型** | PyTorch LSTM，二分类 |
| **运行方式** | 本地 notebook / Google Colab notebook |
| **测试准确率** | **88.62%**（Colab 记录，测试集 5,000 条） |

---

## IMDB 情感分析

模型使用训练集构建的词表和端到端训练的 LSTM 分类器。

### 流水线概览

```
原始影评文本
      │
      ▼
分层划分训练 / 验证 / 测试集（80% / 10% / 10%）
      │
      ▼
清理 HTML、保留否定词、按句批量词性标注
      │
      ▼
仅用训练集构建词表（最多 50,000 项，<PAD>=0，<UNK>=1）
      │
      ▼
整数编码（最大长度 = 256，batch 内动态 Padding）
      │
      ▼
嵌入层（词表大小 → 128 维）
      │
      ▼
LSTM 层（128 个隐藏单元，pack_padded_sequence）
      │
      ▼
验证集早停与最佳 checkpoint
      │
      ▼
全连接 logits → BCEWithLogitsLoss
```

### 数据集

- **来源：** [Kaggle 上的 IMDB 数据集](https://www.kaggle.com/datasets/rehanliaqat17/imbd-dataset)
- **规模：** 50,000 条电影评论 —— 25,000 条正面、25,000 条负面（完全均衡）
- **划分：** 分层 80% 训练集（40,000 条）/ 10% 验证集（5,000 条）/ 10% 测试集（5,000 条）
- **格式：** CSV，包含 `review`（文本）与 `sentiment`（正面/负面）两列
- **获取：** `IMDB_Dataset.csv` 已放在项目目录下。本地版优先读取本地文件，缺失时回退到 KaggleHub；Colab 版固定从 Google Drive 项目目录读取

### 预处理流水线

每条评论在编码前都会经过以下步骤：

1. **先划分数据** —— 分层拆分训练、验证和测试集，避免测试集参与词表拟合
2. **清理文本** —— 解码 HTML 实体并移除 HTML 标签
3. **保留否定词** —— 停用词过滤时排除 `not`、`no`、`don't` 等否定形式
4. **词性标注与词形还原** —— 每条评论批量标注一次，再使用 `WordNetLemmatizer`
5. **构建词表** —— 仅使用训练集，最多保留 50,000 项并过滤低频词
6. **整数编码** —— 未登录词映射为 `<UNK>`
7. **动态填充 / 截断** —— 最长 256 个词元，按 batch 补齐

### 模型结构

```python
class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128, padding_idx=0)
        self.lstm      = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(128, 1)  # 返回 logits
```

| 层 | 输出形状 | 说明 |
|---|---|---|
| Embedding | `(batch, seq_len, 128)` | 可学习的词向量表示 |
| LSTM | `(batch, 128)` | 取最后一个隐藏状态 |
| Dropout | `(batch, 128)` | p=0.3，缓解过拟合 |
| Linear | `(batch,)` | 输出 logits；推理时再经 Sigmoid 转成概率 |

### 训练配置

| 超参数 | 取值 |
|---|---|
| 优化器 | Adam |
| 学习率 | 0.001 |
| 损失函数 | BCEWithLogitsLoss |
| 批大小 | 64 |
| 训练轮数 | 最多 10，验证损失连续 3 轮不改善时提前停止 |
| 最大序列长度 | 256 |
| 词表上限 | 50,000（包含特殊 token） |
| 数据划分 | 分层 80% / 10% / 10% |
| 梯度裁剪 | 1.0 |
| 随机种子 | 42 |

### 实验结果

Colab notebook 当前保存的运行记录如下。训练在第 7 轮触发早停，最终测试指标来自验证损失最低的 checkpoint。

| 指标 | 测试结果 |
|---|---:|
| 测试损失 | 0.2976 |
| 准确率 | 88.62% |
| Negative F1 | 0.8868 |
| Positive F1 | 0.8856 |

混淆矩阵（行是真实类别，列是预测类别，类别顺序为 negative / positive）：

```text
[[2229  271]
 [ 298 2202]]
```

以上是 notebook 中保留的 Colab 运行结果；更改数据、随机种子或训练配置后，应重新评估并更新此处数值。

---

## NLP 预处理流水线

预处理实现位于 notebook 的第 8 节：停用词集合只创建一次，否定词被保留，每条评论只调用一次词性标注器。训练、验证、测试划分发生在词表构建之前。

---

## 安装

本地运行依赖：

```bash
pip install torch torchvision nltk pandas numpy scikit-learn matplotlib seaborn kagglehub
```

本地 notebook 使用项目中的 `nltk_data/`。Colab notebook 首次运行时会把缺少的 NLTK 资源下载到 Drive 项目目录下的 `nltk_data/`。

---

## 使用方法

### 本地运行

```bash
jupyter notebook IMBD_classification.ipynb
```

从上到下运行单元格。数据文件缺失时，本地版会尝试使用 KaggleHub 下载；训练结束后会在项目目录中写入 `best_sentiment_model.pt` 和 `sentiment_model.pt`。

### Google Colab 运行

将 `IMBD_classification_colab.ipynb` 和 `IMDB_Dataset.csv` 放在 Google Drive 的以下目录：

```text
/content/drive/MyDrive/Sentiment-Classification-PyTorch-IMBD/
```

在 Colab 打开 notebook 并从上到下运行。notebook 会挂载 Drive，检查数据文件，并将 NLTK 资源及两个模型文件保存在该目录中。

### 加载已保存模型

先运行 notebook 中的模型定义单元格并设置 `device`，然后加载完整 bundle：

```python
checkpoint = torch.load("sentiment_model.pt", map_location=device)
vocab = checkpoint["vocab"]
MAX_LEN = checkpoint["max_len"]
model = SentimentModel(len(vocab)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

## 项目结构

```
Sentiment-Classification-PyTorch/
│
├── IMBD_classification.ipynb          # 本地运行版
├── IMBD_classification_colab.ipynb     # Google Colab 版，包含训练输出
├── IMDB_Dataset.csv                    # 50,000 条评论数据
├── nltk_data/                          # NLTK 资源
├── best_sentiment_model.pt             # 验证损失最低的 checkpoint
├── sentiment_model.pt                  # 推理用模型 bundle
└── README.md                           # 项目说明
```

---

## 技术栈

| 技术 | 用途 |
|---|---|
| **PyTorch** | LSTM 模型定义、训练循环、GPU 加速 |
| **NLTK** | 分词、去除停用词、词性标注、词形还原 |
| **pandas** | 数据加载、处理与标签编码 |
| **NumPy** | 数值运算与数组处理 |
| **scikit-learn** | 分层数据划分、混淆矩阵、分类报告 |
| **Matplotlib / Seaborn** | EDA 可视化、混淆矩阵热力图 |
| **KaggleHub** | 本地版在数据文件缺失时从 Kaggle 下载 |

---

## 关键词

`文本分类` · `LSTM` · `PyTorch` · `NLP` · `自然语言处理` · `情感分析` · `IMDB 数据集` · `电影评论分类` · `二分类` · `深度学习` · `循环神经网络` · `RNN` · `序列建模` · `词向量` · `词形还原` · `停用词过滤` · `词性标注` · `自建词表` · `pack_padded_sequence` · `BCEWithLogitsLoss` · `文本预处理流水线` · `Kaggle NLP`

---

## 项目来源与维护

本项目源自 Arian Latifi 的 [Sentiment-Classification-PyTorch](https://github.com/ArianLatifi/Sentiment-Classification-PyTorch)。上游目录包含 IMDB 情感分类和 HuffPost 新闻分类两个 notebook；当前项目聚焦 IMDB 情感分类，新闻分类 notebook 未纳入此目录。

**上游作者：** Arian Latifi，计算机科学专业学生。感谢作者提供原始 PyTorch LSTM 实现与项目基础。

**当前维护者：** 木子白白白（[GitHub](https://github.com/hnlyzxf)）。本项目的维护工作包括：

- 整理并改写 IMDB notebook 的中文说明、数据检查和运行流程。
- 优化文本预处理，保留否定词、清理 HTML、减少逐词词性标注开销，并限制训练词表规模。
- 重构训练评估流程，加入分层训练 / 验证 / 测试划分、验证集早停、最佳 checkpoint、分类报告和混淆矩阵。
- 增加 Google Colab 版本，适配指定的 Google Drive 目录，并将 NLTK 资源与模型产物保存在 Drive 中。
- 保存推理所需的模型权重、词表和最大序列长度，并根据实际 notebook 运行结果维护本文档。

当前仓库在上游 IMDB 实现基础上，持续维护中文说明、模型代码、训练评估流程与 Colab 运行支持。欢迎通过 issue 或 pull request 提出问题和改进建议。
