# A-MEM 复现项目

这是论文 "A-MEM: Agentic Memory for LLM Agents" 的复现实现，使用 DeepSeek API 作为 LLM 后端。

## 论文简介

A-MEM 提出了一种智能记忆系统，可以让 LLM 代理动态组织和管理记忆。该系统基于 Zettelkasten 方法，通过以下方式工作：

1. **记忆分析**：自动提取关键词、上下文和标签
2. **记忆进化**：根据相关记忆动态更新记忆之间的链接
3. **混合检索**：结合 BM25 和语义嵌入进行记忆检索

## 环境要求

- Python 3.8+
- DeepSeek API Key

## 安装

```bash
# 克隆或下载本项目
cd "E:\Amem reproducing"

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# 或
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 快速开始

```bash
# 设置 API Key（方式一：环境变量）
set DEEPSEEK_API_KEY=your_api_key_here

# 或直接在命令行中指定（方式二）
python evaluate.py --api_key YOUR_API_KEY
```

### 完整参数

```bash
python evaluate.py \
    --api_key YOUR_API_KEY \
    --model deepseek-v4-flash \
    --base_url https://api.deepseek.com \
    --dataset data/locomo10.json \
    --output_dir results \
    --ratio 1.0 \
    --retrieve_k 10 \
    --temperature_c5 0.5 \
    --embedding_model all-MiniLM-L6-v2 \
    --backend deepseek
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | 环境变量 | DeepSeek API Key |
| `--model` | deepseek-v4-flash | 模型名称 |
| `--base_url` | https://api.deepseek.com | API 地址 |
| `--dataset` | data/locomo10.json | 数据集路径 |
| `--output_dir` | results | 结果输出目录 |
| `--ratio` | 1.0 | 评估数据比例（0.0-1.0） |
| `--retrieve_k` | 10 | 检索记忆数量 |
| `--temperature_c5` | 0.5 | 类别5问题的温度 |
| `--embedding_model` | all-MiniLM-L6-v2 | 嵌入模型 |
| `--backend` | deepseek | 后端类型 |

### 快速测试

如果只想快速测试，可以使用 `--ratio 0.1` 只评估 10% 的数据：

```bash
python evaluate.py --api_key YOUR_API_KEY --ratio 0.1
```

## 数据集

使用 LoCoMo 数据集进行评估，包含以下类别：

- **Category 1**: 单跳问题 (Single-hop)
- **Category 2**: 时间问题 (Temporal)
- **Category 3**: 开放域问题 (Open-domain)
- **Category 4**: 多跳问题 (Multi-hop)
- **Category 5**: 对抗性问题 (Adversarial)

## 评估指标

- Exact Match (精确匹配)
- F1 Score
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- Sentence-BERT Similarity

## 项目结构

```
E:\Amem reproducing\
├── README.md           # 本文件
├── requirements.txt    # 依赖列表
├── config.py          # 配置文件
├── llm_controllers.py # LLM 控制器
├── memory_layer.py    # 记忆层实现
├── load_dataset.py    # 数据集加载器
├── metrics.py         # 评估指标
├── evaluate.py        # 主评估脚本
└── data/
    └── locomo10.json  # LoCoMo 数据集
```

## 参考文献

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 许可证

本项目仅用于学术研究目的。
