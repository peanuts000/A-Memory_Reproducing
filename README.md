# A-Mem：面向 LLM 智能体的自主记忆系统

对论文 **"A-Mem: Agentic Memory for LLM Agents"**（arXiv:2502.12110）的忠实复现。

## 概述

A-Mem 是一种面向 LLM 智能体的自主记忆系统，能够实现动态记忆结构化，无需依赖静态、预定义的记忆操作。该系统受 **Zettelkasten 方法** 启发，通过以下机制构建互联的知识网络：

1. **笔记构建（Note Construction）** — 生成结构化记忆笔记，由 LLM 提取关键词、上下文和标签
2. **链接生成（Link Generation）** — 在相关记忆之间建立语义连接
3. **记忆演化（Memory Evolution）** — 随着新经验的融入，动态更新已有记忆
4. **记忆检索（Memory Retrieval）** — 使用嵌入向量的余弦相似度检索相关记忆

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM 智能体                              │
│                      交互层                                  │
│                       │                                      │
│                       ▼                                      │
│              ┌────────────────┐                              │
│              │  笔记构建       │  K_i, G_i, X_i ← LLM(c_i)  │
│              │  Note           │  e_i = f_enc(concat(...))    │
│              │  Construction   │                              │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  链接生成       │  查找 top-k 最近邻            │
│              │  Link           │  LLM 决定连接关系            │
│              │  Generation     │                              │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  记忆演化       │  更新已有记忆的上下文和标签    │
│              │  Memory         │                              │
│              │  Evolution      │                              │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  记忆存储       │  互联的知识网络               │
│              │  Memory Store   │  （嵌入向量）                 │
│              │  (Embeddings)   │                              │
│              └────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/A-Memory-Reproducing.git
cd A-Memory-Reproducing

# 安装依赖
pip install -r requirements.txt

# 复制环境配置文件并填入你的 API 密钥
cp .env.example .env
```

### 配置

编辑 `.env` 文件，填入你的 API 凭据：

```env
# 豆包（火山引擎 Ark API）— 国内推荐
DOUBAO_API_KEY=你的Ark-API密钥
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-seed-2.0-lite-260215

# 或 OpenAI
OPENAI_API_KEY=sk-你的API密钥
```

### 运行示例

#### 豆包 Doubao（国内推荐）

```bash
# 使用豆包模型运行示例（从 .env 自动读取配置）
python examples/doubao_demo.py

# 或指定参数
python examples/doubao_demo.py --api-key "你的Key" --model "doubao-seed-2.0-lite-260215"
```

#### OpenAI

```bash
python examples/basic_demo.py --backend openai --model gpt-4o-mini
```

#### Ollama（本地部署）

```bash
# 先启动 Ollama 并拉取模型
ollama serve
ollama pull llama3.2

# 运行示例
python examples/basic_demo.py --backend ollama --model llama3.2
```

#### 交互式对话

```bash
python examples/conversation_demo.py --interactive
```

### Web 可视化

启动 Web 可视化界面，直观查看记忆网络：

```bash
# 启动 Web 服务器
python web/app.py

# 在浏览器中打开
# http://localhost:5000
```

Web 界面功能：
- 📝 **添加记忆** — 输入内容，实时查看笔记构建过程
- 🔍 **检索记忆** — 查询相关记忆，观察检索结果
- 🌐 **网络图** — 可视化记忆节点和连接关系
- 🔄 **演化记录** — 查看记忆标签和上下文的动态更新
- 💾 **持久化** — 记忆自动保存到 `web/data/` 目录

## 项目结构

```
├── amem/
│   ├── __init__.py              # 包导出
│   ├── memory_note.py           # MemoryNote 数据类（m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}）
│   ├── retriever.py             # SimpleEmbeddingRetriever（余弦相似度检索）
│   ├── llm_controller.py        # LLM 后端抽象层（OpenAI/Ollama/LiteLLM/豆包）
│   ├── agentic_memory.py        # AgenticMemorySystem（核心编排器）
│   ├── prompt_templates.py      # 提示词模板（Ps1, Ps2, Ps3）
│   └── parsers.py               # LLM 响应解析器（含回退机制）
├── web/
│   ├── app.py                   # Flask Web 服务器
│   ├── templates/
│   │   └── index.html           # Web 可视化界面
│   └── data/
│       ├── memories.json        # 持久化的记忆数据
│       └── meta.json            # 系统元数据
├── examples/
│   ├── basic_demo.py            # 基础用法示例
│   ├── doubao_demo.py           # 豆包模型示例
│   └── conversation_demo.py     # 多轮对话示例
├── evaluation/
│   ├── evaluate.py              # LoCoMo 数据集评估脚本
│   ├── metrics.py               # F1、BLEU-1、ROUGE-L、ROUGE-2、METEOR、SBERT
│   └── load_dataset.py          # LoCoMo 数据集加载器
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## 评估

### 在 LoCoMo 数据集上评估

```bash
python evaluation/evaluate.py \
    --data_dir ./data/locomo \
    --backend openai \
    --model gpt-4o-mini \
    --top_k 10 \
    --output results.json
```

### 评估指标

| 指标 | 说明 |
|------|------|
| F1 | 词级 F1 分数（精确率 + 召回率） |
| BLEU-1 | 一元组重叠精确率 |
| ROUGE-L | 最长公共子序列 |
| ROUGE-2 | 二元组重叠 |
| METEOR | 对齐一元组（含同义词匹配） |
| SBERT | 句子嵌入余弦相似度 |

## LLM 后端

### 豆包 Doubao（国内推荐）

使用火山引擎的豆包模型，适合国内用户：

1. 注册 [火山引擎](https://www.volcengine.com/) 账号
2. 开通 [Ark API 服务](https://console.volcengine.com/ark)
3. 创建 API Key 并获取 endpoint
4. 配置 `.env` 文件：

```env
DOUBAO_API_KEY=你的Ark-API密钥
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-seed-2.0-lite-260215
```

```python
memory = AgenticMemorySystem(
    llm_backend="doubao",
    llm_model="doubao-seed-2.0-lite-260215",
    api_key="你的API密钥",
    api_base="https://ark.cn-beijing.volces.com/api/v3"
)
```

### OpenAI

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

memory = AgenticMemorySystem(llm_backend="openai", llm_model="gpt-4o-mini")
```

### Ollama（本地部署）

```bash
# 启动 Ollama 服务
ollama serve

# 拉取模型
ollama pull llama3.2
```

```python
memory = AgenticMemorySystem(llm_backend="ollama", llm_model="llama3.2")
```

### LiteLLM（通用适配）

```python
memory = AgenticMemorySystem(
    llm_backend="litellm",
    llm_model="ollama/llama3.2",
    api_base="http://localhost:11434"
)
```

## 论文引用

```bibtex
@article{xu2025amem,
  title={A-Mem: Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## 致谢

- 原始论文：[A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- 原始代码：[WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)
- 嵌入模型：[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
