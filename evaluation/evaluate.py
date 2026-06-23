"""
A-Mem 在 LoCoMo 数据集上的评估脚本。

复现论文中的评估（第 4 节）：
  - 数据集：LoCoMo（5 个类别共 7,512 个 QA 对）
  - 指标：F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT 相似度
  - 类别：单跳、多跳、时间、开放领域、对抗性

使用方法：
    python evaluation/evaluate.py \
        --data_dir ./data/locomo \
        --backend doubao \
        --model doubao-seed-2-0-lite-260215 \
        --output results.json
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem
from evaluation.load_dataset import load_locomo_dataset, Conversation
from evaluation.metrics import calculate_all_metrics, aggregate_metrics

# LoCoMo 数据集的类别名称
CATEGORY_NAMES = {
    0: "单跳 (Single Hop)",
    1: "多跳 (Multi Hop)",
    2: "时间 (Temporal)",
    3: "开放领域 (Open Domain)",
    4: "对抗性 (Adversarial)",
}


def evaluate_conversation(
    conversation: Conversation,
    memory_system: AgenticMemorySystem,
    top_k: int = 10,
) -> List[Dict]:
    """在单个对话上评估 A-Mem。

    对于对话中的每一轮，将其添加为记忆笔记。
    然后对于每个 QA 对，检索相关记忆并生成答案。

    参数:
        conversation: 要评估的对话。
        memory_system: 初始化的 A-Mem 系统。
        top_k: 每个问题检索的记忆数量。

    返回:
        包含预测、参考和指标的结果字典列表。
    """
    results = []

    # 阶段 1：将所有对话轮次作为记忆摄入
    print(f"  正在摄入 {sum(len(s.turns) for s in conversation.sessions)} 轮对话...")
    for session in conversation.sessions:
        for turn in session.turns:
            content = f"Speaker {turn.speaker} says: {turn.content}"
            memory_system.add_note(content, time=turn.timestamp)

    print(f"  记忆总数: {memory_system.get_memory_count()}")

    # 阶段 2：使用检索到的记忆回答问题
    for qa in conversation.qa_pairs:
        # 检索相关记忆
        context = memory_system.retrieve_with_context(qa.question, k=top_k)

        # 使用 LLM 和检索到的上下文生成答案
        prompt = (
            f"基于以下对话记忆，简洁地回答问题。\n\n"
            f"记忆:\n{context}\n\n"
            f"问题: {qa.question}\n\n"
            f"请用短语或句子回答:"
        )

        try:
            response = memory_system.llm_controller.llm.get_completion(
                prompt, temperature=0.1
            )
            prediction = response.strip()
        except Exception as e:
            print(f"  生成答案时出错: {e}")
            prediction = ""

        # 计算指标
        metrics = calculate_all_metrics(prediction, qa.answer)

        results.append({
            "question": qa.question,
            "reference": qa.answer,
            "prediction": prediction,
            "category": qa.category,
            "category_name": qa.category_name or CATEGORY_NAMES.get(qa.category, "未知"),
            "metrics": metrics,
        })

    return results


def run_evaluation(
    data_dir: str,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = None,
    api_base: str = None,
    top_k: int = 10,
    output_file: str = "results.json",
    max_conversations: int = None,
):
    """在 LoCoMo 数据集上运行完整评估。

    参数:
        data_dir: LoCoMo 数据目录路径。
        backend: LLM 后端。
        model: LLM 模型。
        api_key: 可选的 API Key。
        top_k: 检索的记忆数量。
        output_file: 保存结果的路径。
        max_conversations: 限制对话数量（用于测试）。
    """
    print("=" * 60)
    print("A-Mem 在 LoCoMo 数据集上的评估")
    print("=" * 60)
    print(f"后端: {backend}, 模型: {model}, Top-k: {top_k}")
    print()

    # 加载数据集
    conversations = load_locomo_dataset(data_dir)
    if not conversations:
        print(f"在 {data_dir} 中未找到对话数据")
        return

    if max_conversations:
        conversations = conversations[:max_conversations]

    print(f"已加载 {len(conversations)} 个对话")

    all_results = []
    all_metrics = []
    all_categories = []
    total_time = 0

    for i, conv in enumerate(conversations):
        print(f"\n评估对话 {i + 1}/{len(conversations)}: {conv.conversation_id}")

        # 为每个对话创建新的记忆系统
        memory_system = AgenticMemorySystem(
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            top_k=top_k,
        )

        start_time = time.time()
        results = evaluate_conversation(conv, memory_system, top_k)
        elapsed = time.time() - start_time
        total_time += elapsed

        for r in results:
            all_metrics.append(r["metrics"])
            all_categories.append(r["category"])

        all_results.extend(results)
        print(f"  完成耗时 {elapsed:.1f}s, {len(results)} 个 QA 对")

    # 聚合结果
    aggregated = aggregate_metrics(all_metrics, all_categories)

    # 打印摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)

    overall = aggregated.get("overall", {})
    print("\n总体:")
    for metric_name, stats in overall.items():
        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")

    print("\n各类别:")
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_key = f"category_{cat_id}"
        if cat_key in aggregated:
            print(f"\n  {cat_name}:")
            for metric_name, stats in aggregated[cat_key].items():
                print(f"    {metric_name}: {stats['mean']:.4f}")

    print(f"\n总评估时间: {total_time:.1f}s")
    print(f"平均每 QA 耗时: {total_time / max(len(all_results), 1):.2f}s")

    # 保存结果
    output = {
        "config": {
            "backend": backend,
            "model": model,
            "top_k": top_k,
            "num_conversations": len(conversations),
        },
        "aggregated": aggregated,
        "detailed_results": all_results,
        "total_time_seconds": total_time,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


def main():
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215") if default_backend == "doubao" else "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="在 LoCoMo 上评估 A-Mem")
    parser.add_argument("--data_dir", type=str, required=True, help="LoCoMo 数据路径")
    parser.add_argument("--backend", type=str, default=default_backend, choices=["openai", "ollama", "litellm", "doubao"])
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL（从 .env 自动读取）")
    parser.add_argument("--top_k", type=int, default=10, help="检索的记忆数量")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max_conversations", type=int, default=None, help="限制对话数量用于测试")
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        api_base=args.base_url,
        top_k=args.top_k,
        output_file=args.output,
        max_conversations=args.max_conversations,
    )


if __name__ == "__main__":
    main()
