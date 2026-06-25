"""
A-Mem 在 LoCoMo 数据集上的评估脚本。

复现论文中的评估（第 4 节）：
  - 数据集：LoCoMo（5 个类别共 7,512 个 QA 对）
  - 指标：F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT 相似度
  - 类别：单跳、多跳、时间、开放领域、对抗性

增强功能：
  - LLM 查询关键词生成：从问题中提取检索关键词
  - LLM 记忆筛选：从检索结果中筛选最相关部分
  - 分类别 prompt：不同问题类型使用不同提示词
  - 记忆缓存：避免重复构建记忆

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
import pickle
import random
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem, LLMController
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


class MemoryAgent:
    """增强的记忆代理，封装 AgenticMemorySystem 并提供高级检索功能。

    增强功能：
      - 查询关键词生成：使用 LLM 从问题中提取检索关键词
      - LLM 记忆筛选：使用 LLM 从检索结果中筛选最相关部分
      - 分类别 prompt：不同问题类型使用不同的回答提示词
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = None,
        llm_model: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieve_k: int = 10,
        temperature_c5: float = 0.5,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.5,
    ):
        self.memory_system = AgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            api_base=api_base,
            use_hybrid=use_hybrid,
            hybrid_alpha=hybrid_alpha,
        )
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content: str, time: str = None):
        """添加记忆。"""
        self.memory_system.add_note(content, time=time)

    def generate_query_keywords(self, question: str) -> str:
        """使用 LLM 从问题中生成检索关键词。

        参数:
            question: 用户问题。

        返回:
            逗号分隔的关键词字符串。
        """
        prompt = f"""Given the following question, generate several keywords for memory retrieval.

Question: {question}

Please respond with a JSON object containing a "keywords" field with comma-separated keywords.
Example: {{"keywords": "keyword1, keyword2, keyword3"}}"""

        try:
            response = self.llm_controller.llm.get_completion(
                prompt, temperature=0.3
            )
            # 尝试解析 JSON
            result = json.loads(response.strip().strip("`").replace("json\n", "").replace("```", ""))
            keywords = result.get("keywords", question)
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            return keywords
        except Exception:
            # 回退：直接使用问题作为查询
            return question

    def retrieve_memory_llm(self, memories_text: str, query: str) -> str:
        """使用 LLM 从检索到的记忆中筛选最相关的部分。

        参数:
            memories_text: 检索到的记忆文本。
            query: 用户问题。

        返回:
            筛选后的相关记忆文本。
        """
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts. If no parts are relevant, return the original text unchanged.
Please respond with a JSON object containing a "relevant_parts" field.
Example: {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""

        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.3)
            result = json.loads(response.strip().strip("`").replace("json\n", "").replace("```", ""))
            return result.get("relevant_parts", memories_text)
        except Exception:
            return memories_text

    def retrieve_memory(self, query: str, k: int = None) -> str:
        """检索相关记忆并返回格式化字符串。

        参数:
            query: 查询文本。
            k: 检索数量。

        返回:
            格式化的记忆文本。
        """
        k = k or self.retrieve_k
        return self.memory_system._find_related_memories_raw(query, k)

    def answer_question(
        self, question: str, category: int, use_llm_filter: bool = False
    ) -> Tuple[str, str, str]:
        """生成问题的回答。

        参数:
            question: 问题文本。
            category: 问题类别（0-4）。
            use_llm_filter: 是否使用 LLM 筛选记忆。

        返回:
            (回答, prompt, 原始上下文) 元组。
        """
        # 生成检索关键词
        keywords = self.generate_query_keywords(question)

        # 检索相关记忆
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)

        # 可选：使用 LLM 筛选记忆
        context = raw_context
        if use_llm_filter and raw_context:
            context = self.retrieve_memory_llm(raw_context, question)

        # 根据类别选择不同的 prompt
        temperature = 0.1
        if category == 4:  # 对抗性问题
            temperature = self.temperature_c5
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer or indicate if it's not mentioned in the conversation.
Short answer:"""
        elif category == 2:  # 时间问题
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible.

Question: {question} Short answer:"""
        elif category == 1:  # 多跳问题
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. This question may require information from multiple conversations.

Question: {question} Short answer:"""
        else:  # 单跳 (0) 和开放领域 (3)
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""

        try:
            response = self.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature
            )
            prediction = response.strip()
            # 尝试解析 JSON 格式的回答
            try:
                result = json.loads(prediction.strip().strip("`").replace("json\n", "").replace("```", ""))
                prediction = result.get("answer", prediction)
            except (json.JSONDecodeError, ValueError):
                pass
        except Exception as e:
            print(f"  生成答案时出错: {e}")
            prediction = ""

        return prediction, user_prompt, raw_context


def evaluate_conversation(
    conversation: Conversation,
    agent: MemoryAgent,
    use_llm_filter: bool = False,
    use_memory_cache: bool = True,
    cache_dir: str = None,
    sample_idx: int = 0,
) -> List[Dict]:
    """在单个对话上评估 A-Mem。

    参数:
        conversation: 要评估的对话。
        agent: 初始化的记忆代理。
        use_llm_filter: 是否使用 LLM 筛选记忆。
        use_memory_cache: 是否使用记忆缓存。
        cache_dir: 缓存目录。
        sample_idx: 样本索引（用于缓存文件名）。

    返回:
        包含预测、参考和指标的结果字典列表。
    """
    results = []

    # 尝试加载缓存的记忆
    memory_loaded = False
    if use_memory_cache and cache_dir:
        cache_file = os.path.join(cache_dir, f"memory_cache_sample_{sample_idx}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_memories = pickle.load(f)
                agent.memory_system.memories = cached_memories
                # 重建检索器索引
                agent.memory_system.consolidate_memories()
                print(f"  已从缓存加载 {len(cached_memories)} 条记忆")
                memory_loaded = True
            except Exception as e:
                print(f"  加载缓存失败: {e}，将重新构建记忆")

    # 如果没有缓存，摄入对话轮次
    if not memory_loaded:
        total_turns = sum(len(s.turns) for s in conversation.sessions)
        print(f"  正在摄入 {total_turns} 轮对话...")
        for session in conversation.sessions:
            for turn in session.turns:
                content = f"Speaker {turn.speaker} says: {turn.content}"
                agent.add_memory(content, time=turn.timestamp)

        # 保存缓存
        if use_memory_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"memory_cache_sample_{sample_idx}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            print(f"  已缓存 {len(agent.memory_system.memories)} 条记忆")

    print(f"  记忆总数: {agent.memory_system.get_memory_count()}")

    # 使用检索到的记忆回答问题
    for qa in conversation.qa_pairs:
        prediction, user_prompt, raw_context = agent.answer_question(
            qa.question, qa.category, use_llm_filter=use_llm_filter
        )

        # 计算指标
        metrics = calculate_all_metrics(prediction, qa.answer)

        results.append({
            "question": qa.question,
            "reference": qa.answer,
            "prediction": prediction,
            "category": qa.category,
            "category_name": qa.category_name or CATEGORY_NAMES.get(qa.category, "未知"),
            "metrics": metrics,
            "raw_context": raw_context[:500],  # 截断以节省空间
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
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5,
    use_llm_filter: bool = False,
    use_memory_cache: bool = True,
    temperature_c5: float = 0.5,
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
        use_hybrid: 是否使用混合检索。
        hybrid_alpha: 混合检索权重。
        use_llm_filter: 是否使用 LLM 筛选记忆。
        use_memory_cache: 是否使用记忆缓存。
        temperature_c5: 对抗性问题的温度参数。
    """
    print("=" * 60)
    print("A-Mem 在 LoCoMo 数据集上的评估")
    print("=" * 60)
    print(f"后端: {backend}, 模型: {model}, Top-k: {top_k}")
    print(f"混合检索: {use_hybrid}, Alpha: {hybrid_alpha}")
    print(f"LLM 筛选: {use_llm_filter}, 记忆缓存: {use_memory_cache}")
    print()

    # 加载数据集
    conversations = load_locomo_dataset(data_dir)
    if not conversations:
        print(f"在 {data_dir} 中未找到对话数据")
        return

    if max_conversations:
        conversations = conversations[:max_conversations]

    print(f"已加载 {len(conversations)} 个对话")

    # 缓存目录
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        f"cached_memories_{backend}_{model.replace('/', '_')}"
    ) if use_memory_cache else None

    all_results = []
    all_metrics = []
    all_categories = []
    total_time = 0

    for i, conv in enumerate(conversations):
        print(f"\n评估对话 {i + 1}/{len(conversations)}: {conv.conversation_id}")

        # 为每个对话创建新的记忆代理
        agent = MemoryAgent(
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            retrieve_k=top_k,
            temperature_c5=temperature_c5,
            use_hybrid=use_hybrid,
            hybrid_alpha=hybrid_alpha,
        )

        start_time = time.time()
        results = evaluate_conversation(
            conv, agent,
            use_llm_filter=use_llm_filter,
            use_memory_cache=use_memory_cache,
            cache_dir=cache_dir,
            sample_idx=i,
        )
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
            "use_hybrid": use_hybrid,
            "hybrid_alpha": hybrid_alpha,
            "use_llm_filter": use_llm_filter,
            "temperature_c5": temperature_c5,
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
    default_model = (
        os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215")
        if default_backend == "doubao"
        else "gpt-4o-mini"
    )

    parser = argparse.ArgumentParser(description="在 LoCoMo 上评估 A-Mem")
    parser.add_argument("--data_dir", type=str, required=True, help="LoCoMo 数据路径")
    parser.add_argument(
        "--backend",
        type=str,
        default=default_backend,
        choices=["openai", "ollama", "litellm", "doubao"],
    )
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL")
    parser.add_argument("--top_k", type=int, default=10, help="检索的记忆数量")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max_conversations", type=int, default=None, help="限制对话数量用于测试")
    parser.add_argument("--no_hybrid", action="store_true", help="禁用混合检索，使用纯语义检索")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5, help="混合检索语义权重 (0-1)")
    parser.add_argument("--use_llm_filter", action="store_true", help="使用 LLM 筛选记忆")
    parser.add_argument("--no_cache", action="store_true", help="禁用记忆缓存")
    parser.add_argument("--temperature_c5", type=float, default=0.5, help="对抗性问题温度")
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
        use_hybrid=not args.no_hybrid,
        hybrid_alpha=args.hybrid_alpha,
        use_llm_filter=args.use_llm_filter,
        use_memory_cache=not args.no_cache,
        temperature_c5=args.temperature_c5,
    )


if __name__ == "__main__":
    main()
