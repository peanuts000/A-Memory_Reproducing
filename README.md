# A-Mem: Agentic Memory for LLM Agents

A faithful reproduction of the paper **"A-Mem: Agentic Memory for LLM Agents"** (arXiv:2502.12110).

## Overview

A-Mem is an agentic memory system for LLM agents that enables dynamic memory structuring without relying on static, predetermined memory operations. Inspired by the **Zettelkasten method**, the system creates interconnected knowledge networks through:

1. **Note Construction** - Generate structured memory notes with LLM-extracted keywords, context, and tags
2. **Link Generation** - Establish semantic connections between related memories
3. **Memory Evolution** - Dynamically update existing memories as new experiences are integrated
4. **Memory Retrieval** - Retrieve relevant memories using cosine similarity on embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Agent                              │
│                    Interaction                               │
│                       │                                      │
│                       ▼                                      │
│              ┌────────────────┐                              │
│              │  Note           │  K_i, G_i, X_i ← LLM(c_i)  │
│              │  Construction   │  e_i = f_enc(concat(...))    │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  Link           │  Find top-k nearest neighbors│
│              │  Generation     │  LLM determines connections  │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  Memory         │  Update context & tags of    │
│              │  Evolution      │  existing memories           │
│              └───────┬────────┘                              │
│                      │                                       │
│                      ▼                                       │
│              ┌────────────────┐                              │
│              │  Memory Store   │  Interconnected knowledge    │
│              │  (Embeddings)   │  network                     │
│              └────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from amem import AgenticMemorySystem

# Initialize the memory system
memory = AgenticMemorySystem(
    llm_backend="openai",       # or "ollama", "litellm"
    llm_model="gpt-4o-mini",    # or "llama3.2", etc.
    top_k=5,
)

# Add memories (triggers Note Construction + Link Generation + Evolution)
memory.add_note("Speaker Dave says: I've taken up photography and it's been great.")
memory.add_note("Speaker Calvin says: I've been experimenting with electronic music.")

# Retrieve relevant memories
results = memory.retrieve("What hobby did Dave pick up?", k=3)
for mem in results:
    print(f"Content: {mem.content}")
    print(f"Context: {mem.context}")
    print(f"Keywords: {mem.keywords}")
    print(f"Tags: {mem.tags}")
```

### Running the Demo

```bash
# With OpenAI
python examples/basic_demo.py --backend openai --model gpt-4o-mini

# With Ollama (local)
python examples/basic_demo.py --backend ollama --model llama3.2

# Interactive conversation demo
python examples/conversation_demo.py --interactive
```

## Project Structure

```
├── amem/
│   ├── __init__.py              # Package exports
│   ├── memory_note.py           # MemoryNote data class (m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i})
│   ├── retriever.py             # SimpleEmbeddingRetriever (cosine similarity)
│   ├── llm_controller.py        # LLM backend abstraction (OpenAI/Ollama/LiteLLM)
│   ├── agentic_memory.py        # AgenticMemorySystem (core orchestrator)
│   ├── prompt_templates.py      # Prompt templates (Ps1, Ps2, Ps3)
│   └── parsers.py               # LLM response parsers with fallback
├── examples/
│   ├── basic_demo.py            # Basic usage demonstration
│   └── conversation_demo.py     # Multi-turn conversation demo
├── evaluation/
│   ├── evaluate.py              # LoCoMo dataset evaluation script
│   ├── metrics.py               # F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT
│   └── load_dataset.py          # LoCoMo dataset loader
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## Evaluation

### On LoCoMo Dataset

```bash
python evaluation/evaluate.py \
    --data_dir ./data/locomo \
    --backend openai \
    --model gpt-4o-mini \
    --top_k 10 \
    --output results.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| F1 | Token-level F1 score (precision + recall) |
| BLEU-1 | Unigram overlap precision |
| ROUGE-L | Longest common subsequence |
| ROUGE-2 | Bigram overlap |
| METEOR | Aligned unigrams with synonym matching |
| SBERT | Sentence embedding cosine similarity |

## LLM Backends

### OpenAI
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

memory = AgenticMemorySystem(llm_backend="openai", llm_model="gpt-4o-mini")
```

### Ollama (Local)
```bash
# Start Ollama
ollama serve
ollama pull llama3.2
```
```python
memory = AgenticMemorySystem(llm_backend="ollama", llm_model="llama3.2")
```

### LiteLLM (Universal)
```python
memory = AgenticMemorySystem(
    llm_backend="litellm",
    llm_model="ollama/llama3.2",
    api_base="http://localhost:11434"
)
```

## Paper Reference

```bibtex
@article{xu2025amem,
  title={A-Mem: Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## Acknowledgments

- Original paper: [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- Original code: [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)
- Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
