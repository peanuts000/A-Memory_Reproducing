"""A-Mem 评估脚本 - 限制每个对话只读取前 N 个 session。"""

import json
import os
import sys
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amem import AgenticMemorySystem, LLMController
from evaluation.load_dataset import load_locomo_dataset, Conversation, Session, Turn, QAPair
from evaluation.metrics import calculate_all_metrics, aggregate_metrics


CATEGORY_NAMES = {
    0: "单跳 (SingleHop)",
    1: "多跳 (MultiHop)",
    2: "时间 (Temporal)",
    3: "开放领域 (OpenDomain)",
    4: "对抗性 (Adversarial)",
}


def load_limited_sessions(data_dir: str, max_sessions: int = 0, max_conversations: int = 0) -> list:
    """加载对话，可选限制 session 和 conversation 数量。

    参数:
        data_dir: 数据目录
        max_sessions: 每个对话最大 session 数，0 表示不限制
        max_conversations: 最大对话数，0 表示不限制
    """
    conversations = []

    if not os.path.exists(data_dir):
        print(f"警告: 未找到数据目录: {data_dir}")
        return conversations

    filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])
    if max_conversations > 0:
        filenames = filenames[:max_conversations]

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 解析对话
        conv = Conversation(conversation_id=data.get("conversation_id", "unknown"))

        # 限制 session 数量（如果指定）
        sessions_data = data.get("sessions", [])
        if max_sessions > 0:
            sessions_data = sessions_data[:max_sessions]

        for session_data in sessions_data:
            session = Session(session_id=session_data.get("session_id", 0))
            for turn_data in session_data.get("turns", []):
                turn = Turn(
                    speaker=turn_data.get("speaker", "Unknown"),
                    content=turn_data.get("content", ""),
                    timestamp=turn_data.get("timestamp", ""),
                )
                session.turns.append(turn)
            conv.sessions.append(session)

        # QA 对保持不变
        for qa_data in data.get("qa_pairs", data.get("questions", [])):
            qa = QAPair(
                question=qa_data.get("question", ""),
                answer=qa_data.get("answer", qa_data.get("reference", "")),
                category=qa_data.get("category", 0),
                category_name=CATEGORY_NAMES.get(qa_data.get("category", 0), "未知"),
            )
            conv.qa_pairs.append(qa)

        conversations.append(conv)

    return conversations


def ingest_conversation(memory, conversation):
    """将对话摄入记忆系统。"""
    total_turns = 0
    for session in conversation.sessions:
        for turn in session.turns:
            timestamp = turn.timestamp if turn.timestamp else ""
            content = f"[{timestamp}] Speaker {turn.speaker} says: {turn.content}"
            memory.add_note(content, time=timestamp)
            total_turns += 1
    return total_turns


def answer_question(memory, llm, question, category):
    """使用记忆系统回答问题。"""
    # 使用多查询策略提高召回率
    queries = [question]
    try:
        variant_prompt = f"""Generate 3 different search queries for: "{question}"
Respond with JSON: {{"queries": ["q1", "q2", "q3"]}}"""
        resp = llm.llm.get_completion(variant_prompt, temperature=0.3)
        import json as _json
        result = _json.loads(resp.strip().strip("`").replace("json\n", "").replace("```", ""))
        variants = result.get("queries", [])
        if variants:
            queries = [question] + variants[:3]
    except Exception:
        pass

    # 合并多查询检索结果
    seen_ids = set()
    all_related = []
    for q in queries:
        results = memory.retrieve(q, k=5)
        for m in results:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                all_related.append(m)

    context_parts = []
    for m in all_related[:10]:  # 最多10条
        timestamp = m.timestamp if m.timestamp else "unknown"
        context_parts.append(f"[{timestamp}] {m.content}")
    context = "\n".join(context_parts)

    if category == 2:  # 时间问题
        prompt = f"""Based ONLY on the following conversation memories, answer the question about WHEN something happened.

Context:
{context}

Question: {question}

RULES:
- Use the CONVERSATION DATE in the memories to calculate the absolute date.
- Answer with a specific date: "DD Month YYYY" or "Month YYYY"
- Do NOT use relative terms like "yesterday", "last week"
- If the date cannot be determined, say "Not mentioned"

Short answer:"""
    elif category == 4:  # 对抗性问题
        prompt = f"""Based ONLY on the following conversation memories, answer the question.

Context:
{context}

Question: {question}

RULES:
- If the answer IS in the memories, provide a short answer.
- If the answer is NOT in the memories, respond with exactly: "Not mentioned"
- Do NOT guess or infer information not explicitly stated.

Short answer:"""
    elif category == 1:  # 多跳问题
        prompt = f"""Based ONLY on the following conversation memories, answer the question.

Context:
{context}

Question: {question}

RULES:
- This question may require combining information from MULTIPLE sessions.
- Look through ALL memories carefully.
- Answer with a short phrase using exact words from context.
- If not found, say "Not mentioned"

Short answer:"""
    else:  # 单跳 (0) 和开放领域 (3)
        prompt = f"""Based ONLY on the following conversation memories, answer the question.

Context:
{context}

Question: {question}

RULES:
- Find the EXACT answer from the memories.
- Answer with a short phrase using exact words from context.
- If not found, say "Not mentioned"
- Do NOT output JSON.

Short answer:"""

    try:
        answer = llm.llm.get_completion(prompt, temperature=0.1)
        answer = answer.strip()
        # 清理 JSON 格式的回答
        try:
            import json as _json
            result = _json.loads(answer.strip().strip("`").replace("json\n", "").replace("```", ""))
            answer = result.get("answer", result.get("response", answer))
        except Exception:
            pass
        answer = answer.strip().strip('"').strip("'")
        return answer
    except Exception as e:
        print(f"    生成答案失败: {e}")
        return ""


def run_evaluation(
    data_dir="./data/locomo",
    backend="doubao",
    model="doubao-seed-2-0-lite-260215",
    max_conversations=2,
    max_sessions=2,
    max_qa=None,
    use_cache=True,
    output_file="results_limited.json",
):
    """运行评估。"""
    print("=" * 60)
    print("A-Mem 评估 (限制 session 数量)")
    print("=" * 60)
    print(f"后端: {backend}, 模型: {model}")
    print(f"最大对话数: {max_conversations}")
    print(f"每个对话最大 session 数: {max_sessions}")
    if max_qa:
        print(f"每对话最大 QA 数: {max_qa}")
    print()

    # 加载数据集（限制 session 和 conversation 数量）
    conversations = load_limited_sessions(data_dir, max_sessions, max_conversations)
    if not conversations:
        print(f"在 {data_dir} 中未找到对话数据")
        return

    print(f"已加载 {len(conversations)} 个对话")

    # 打印统计信息
    for i, conv in enumerate(conversations):
        total_turns = sum(len(s.turns) for s in conv.sessions)
        print(f"  对话 {i+1}: {len(conv.sessions)} 个 session, {total_turns} 轮, {len(conv.qa_pairs)} 个 QA")

    # 缓存目录
    cache_dir = f"./cache_limited_{backend}"
    os.makedirs(cache_dir, exist_ok=True)

    # LLM 控制器
    llm = LLMController(backend, model)

    all_results = []
    all_metrics = []
    all_categories = []
    total_time = 0

    for i, conv in enumerate(conversations):
        print(f"\n{'='*60}")
        print(f"对话 {i + 1}/{len(conversations)}: {conv.conversation_id}")
        total_turns = sum(len(s.turns) for s in conv.sessions)
        print(f"Session 数: {len(conv.sessions)}, 轮次: {total_turns}, QA 对数: {len(conv.qa_pairs)}")
        print(f"{'='*60}")

        # 检查缓存
        cache_file = os.path.join(cache_dir, f"{conv.conversation_id}_s{max_sessions}.pkl")
        memory = None

        if use_cache and os.path.exists(cache_file):
            print("加载缓存的记忆...")
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                # 创建新的记忆系统并加载缓存数据
                memory = AgenticMemorySystem(
                    llm_backend=backend,
                    llm_model=model,
                )
                memory.memories = cache_data['memories']
                memory.evo_cnt = cache_data.get('evo_cnt', 0)
                # 重建检索器索引
                memory.consolidate_memories()
                print(f"已加载 {memory.get_memory_count()} 条记忆")
            except Exception as e:
                print(f"加载缓存失败: {e}")
                memory = None

        # 如果没有缓存，摄入记忆
        if memory is None:
            memory = AgenticMemorySystem(
                llm_backend=backend,
                llm_model=model,
            )

            print("正在摄入记忆...")
            start_time = time.time()
            turns_ingested = ingest_conversation(memory, conv)
            elapsed = time.time() - start_time
            print(f"已摄入 {turns_ingested} 轮对话，耗时 {elapsed:.1f}s")
            print(f"记忆总数: {memory.get_memory_count()}")

            # 保存缓存（只保存记忆数据，不保存 LLM 控制器）
            if use_cache:
                try:
                    cache_data = {
                        'memories': memory.memories,
                        'evo_cnt': memory.evo_cnt,
                    }
                    with open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f)
                    print(f"已保存缓存")
                except Exception as e:
                    print(f"保存缓存失败: {e}")

        # 评估 QA 对
        qa_pairs = conv.qa_pairs
        if max_qa:
            qa_pairs = qa_pairs[:max_qa]

        print(f"\n评估 {len(qa_pairs)} 个 QA 对...")

        for qa_idx, qa in enumerate(qa_pairs):
            if (qa_idx + 1) % 20 == 0:
                print(f"  进度: {qa_idx + 1}/{len(qa_pairs)}")

            start_time = time.time()
            prediction = answer_question(memory, llm, qa.question, qa.category)
            elapsed = time.time() - start_time
            total_time += elapsed

            metrics = calculate_all_metrics(prediction, qa.answer)

            all_metrics.append(metrics)
            all_categories.append(qa.category)
            all_results.append({
                "question": qa.question,
                "reference": qa.answer,
                "prediction": prediction,
                "category": qa.category,
                "category_name": CATEGORY_NAMES.get(qa.category, "未知"),
                "metrics": metrics,
            })

    # 聚合结果
    aggregated = aggregate_metrics(all_metrics, all_categories)

    # 打印摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)

    overall = aggregated.get("overall", {})
    print("\n总体指标:")
    for metric_name, stats in overall.items():
        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")

    print("\n各类别指标:")
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_key = f"category_{cat_id}"
        if cat_key in aggregated:
            print(f"\n  {cat_name}:")
            for metric_name, stats in aggregated[cat_key].items():
                print(f"    {metric_name}: {stats['mean']:.4f}")

    print(f"\n总评估时间: {total_time:.1f}s")
    if all_results:
        print(f"平均每 QA 耗时: {total_time / len(all_results):.2f}s")

    # 保存结果
    output = {
        "config": {
            "backend": backend,
            "model": model,
            "max_conversations": max_conversations,
            "max_sessions": max_sessions,
            "max_qa": max_qa,
        },
        "aggregated": aggregated,
        "detailed_results": all_results[:100],
        "total_time_seconds": total_time,
        "total_qa_count": len(all_results),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 与论文结果对比
    print("\n" + "=" * 60)
    print("论文 A-MEM 结果 (Table 1)")
    print("=" * 60)
    print("  MultiHop F1: 27.02, BLEU-1: 20.09")
    print("  Temporal F1: 45.85, BLEU-1: 36.67")
    print("  OpenDomain F1: 12.14, BLEU-1: 12.00")
    print("  SingleHop F1: 44.65, BLEU-1: 37.06")
    print("  Adversarial F1: 50.03, BLEU-1: 49.47")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-Mem 评估 (限制 session 数量)")
    parser.add_argument("--data_dir", type=str, default="./data/locomo")
    parser.add_argument("--backend", type=str, default="doubao")
    parser.add_argument("--model", type=str, default="doubao-seed-2-0-lite-260215")
    parser.add_argument("--max_conversations", type=int, default=2)
    parser.add_argument("--max_sessions", type=int, default=0, help="每个对话最大 session 数（0 表示不限制）")
    parser.add_argument("--max_qa", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--output", type=str, default="results_limited.json")

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        backend=args.backend,
        model=args.model,
        max_conversations=args.max_conversations,
        max_sessions=args.max_sessions,
        max_qa=args.max_qa,
        use_cache=not args.no_cache,
        output_file=args.output,
    )
