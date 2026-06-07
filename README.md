# 中文新闻文本挖掘与热点发现系统

本项目面向“文本挖掘领域的系统/算法”课程大作业，完成一个可复现的中文文本挖掘系统。系统包含新闻文本分类、主题发现、关键词抽取、相似文本检索和可视化演示，复杂度高于单一分类实验，适合写成论文形式的实验报告。

## 项目亮点

- 监督学习：词级 TF-IDF、字级 TF-IDF 与线性 SVM 的混合特征分类器；BERT 预训练嵌入 + 逻辑回归分类器。
- 无监督学习：基于 NMF 的新闻主题发现，输出每个主题的代表词。
- 文本挖掘应用：关键词抽取、相似新闻检索、模型解释性词表。
- 可复现交付：实验脚本会自动保存指标、混淆矩阵、主题词、关键词和模型文件。
- 系统展示：提供 Streamlit 页面，可输入新闻标题进行分类和相似检索。

## 数据集链接

推荐使用 CLUE Benchmark 的 TNEWS 中文新闻分类数据集：

- Hugging Face 数据集页：https://huggingface.co/datasets/clue/clue
- CLUE 官方 GitHub：https://github.com/CLUEbenchmark/CLUE

Hugging Face 页面中的 `tnews` 子集包含约 7.34 万条中文新闻标题，带有 train/validation/test 划分。由于作业要求“不要直接上传数据集”，本仓库只放一个极小样例文件 `data/sample_tnews.csv` 用于本地冒烟测试，正式实验请通过脚本自动下载公开数据集。

## 目录结构

```text
.
├── app.py                         # Streamlit 演示系统
├── data/
│   └── sample_tnews.csv           # 小样例，不是正式数据集
├── report/
│   └── experiment_report.md       # 论文式实验报告模板
├── scripts/
│   └── run_experiment.py          # 一键实验入口
├── src/text_mining_system/
│   ├── data.py                    # 数据读取
│   ├── models.py                  # 分类模型与评估
│   ├── preprocess.py              # 中文清洗与分词
│   ├── similarity.py              # 相似文本检索
│   └── topics.py                  # 主题发现与关键词抽取
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速运行

先用内置小样例检查环境：

```bash
python scripts/run_experiment.py --dataset sample --output results/sample
```

正式实验建议使用 TNEWS 数据集。完整训练集和验证集运行命令如下：

```bash
python scripts/run_experiment.py --dataset clue-tnews --max-train 53360 --max-eval 10000 --classifier hybrid_svm --output results/tnews_full
```

如需运行基线模型：

```bash
python scripts/run_experiment.py --dataset clue-tnews --max-train 53360 --max-eval 10000 --classifier word_lr --output results/tnews_word_lr
```

如需运行 BERT 预训练嵌入模型（需要 GPU，CPU 也可运行但较慢）：

```bash
python scripts/run_experiment.py --dataset clue-tnews --classifier bert --output results/tnews_bert
```

运行后主要结果会保存到输出目录：

- `metrics.json`：准确率、宏平均 F1、加权 F1。
- `classification_report.csv`：各类别 precision、recall、F1。
- `confusion_matrix.png`：混淆矩阵图。
- `topics.csv`：NMF 主题及代表词。
- `keywords.csv`：样本文本关键词。
- `top_features.csv`：分类模型每类重要特征。
- `model.joblib`：训练好的分类模型。

当前已完成一次完整实验，结果保存在 `results/tnews_full`：

- Accuracy：0.5526
- Macro-F1：0.5385
- Weighted-F1：0.5527

同时已完成 `word_lr` 基线实验，结果保存在 `results/tnews_word_lr`：

- Accuracy：0.5357
- Macro-F1：0.5215
- Weighted-F1：0.5374

## 启动演示系统

```bash
streamlit run app.py
```

页面会优先加载 `results/tnews_full/model.joblib` 中的完整分类模型；如果该文件不存在，则自动使用小样例快速训练一个演示模型。相似文本检索和主题查看默认基于内置小样例，便于课堂快速展示。

## 源代码

本项目源代码托管在 GitHub（请将仓库地址替换为你的实际地址）：

- GitHub: https://github.com/你的用户名/Chinese-News-Text-Mining

## 可写论文题目

《融合词级与字级 TF-IDF 特征的中文新闻文本挖掘系统设计与实现》

也可以改成：

《面向中文新闻标题的分类、主题发现与相似检索系统研究》
