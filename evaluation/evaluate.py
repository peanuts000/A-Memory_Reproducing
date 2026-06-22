"""
A-Mem Evaluation Script on LoCoMo Dataset.

Reproduces the evaluation from the paper (Section 4):
  - Dataset: LoCoMo (7,512 QA pairs across 5 categories)
  - Metrics: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT Similarity
  - Categories: Single-hop, Multi-hop, Temporal, Open-domain, Adversarial

Usage:
    python evaluation/evaluate.py \
        --data_dir ./data/locomo \
        --backend openai \
        --model gpt-4o-mini \
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

# Category names for the LoCoMo dataset
CATEGORY_NAMES = {
    0: "Single Hop",
    1: "Multi Hop",
    2: "Temporal",
    3: "Open Domain",
    4: "Adversarial",
}


def evaluate_conversation(
    conversation: Conversation,
    memory_system: AgenticMemorySystem,
    top_k: int = 10,
) -> List[Dict]:
    """Evaluate A-Mem on a single conversation.

    For each turn in the conversation, add it as a memory note.
    Then for each QA pair, retrieve relevant memories and generate an answer.

    Args:
        conversation: The conversation to evaluate.
        memory_system: Initialized A-Mem system.
        top_k: Number of memories to retrieve for each question.

    Returns:
        List of result dicts with predictions, references, and metrics.
    """
    results = []

    # Phase 1: Ingest all conversation turns as memories
    print(f"  Ingesting {sum(len(s.turns) for s in conversation.sessions)} turns...")
    for session in conversation.sessions:
        for turn in session.turns:
            content = f"Speaker {turn.speaker} says: {turn.content}"
            memory_system.add_note(content, time=turn.timestamp)

    print(f"  Total memories: {memory_system.get_memory_count()}")

    # Phase 2: Answer questions using retrieved memories
    for qa in conversation.qa_pairs:
        # Retrieve relevant memories
        context = memory_system.retrieve_with_context(qa.question, k=top_k)

        # Generate answer using LLM with retrieved context
        prompt = (
            f"Based on the following conversation memories, answer the question concisely.\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {qa.question}\n\n"
            f"Answer with a short phrase or sentence:"
        )

        try:
            response = memory_system.llm_controller.llm.get_completion(
                prompt, temperature=0.1
            )
            prediction = response.strip()
        except Exception as e:
            print(f"  Error generating answer: {e}")
            prediction = ""

        # Calculate metrics
        metrics = calculate_all_metrics(prediction, qa.answer)

        results.append({
            "question": qa.question,
            "reference": qa.answer,
            "prediction": prediction,
            "category": qa.category,
            "category_name": qa.category_name or CATEGORY_NAMES.get(qa.category, "unknown"),
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
    """Run full evaluation on the LoCoMo dataset.

    Args:
        data_dir: Path to LoCoMo data directory.
        backend: LLM backend.
        model: LLM model.
        api_key: Optional API key.
        top_k: Number of memories to retrieve.
        output_file: Path to save results.
        max_conversations: Limit number of conversations (for testing).
    """
    print("=" * 60)
    print("A-Mem Evaluation on LoCoMo Dataset")
    print("=" * 60)
    print(f"Backend: {backend}, Model: {model}, Top-k: {top_k}")
    print()

    # Load dataset
    conversations = load_locomo_dataset(data_dir)
    if not conversations:
        print(f"No conversations found in {data_dir}")
        return

    if max_conversations:
        conversations = conversations[:max_conversations]

    print(f"Loaded {len(conversations)} conversations")

    all_results = []
    all_metrics = []
    all_categories = []
    total_time = 0

    for i, conv in enumerate(conversations):
        print(f"\nEvaluating conversation {i + 1}/{len(conversations)}: {conv.conversation_id}")

        # Create a fresh memory system for each conversation
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
        print(f"  Completed in {elapsed:.1f}s, {len(results)} QA pairs")

    # Aggregate results
    aggregated = aggregate_metrics(all_metrics, all_categories)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    overall = aggregated.get("overall", {})
    print("\nOverall:")
    for metric_name, stats in overall.items():
        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")

    print("\nPer Category:")
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_key = f"category_{cat_id}"
        if cat_key in aggregated:
            print(f"\n  {cat_name}:")
            for metric_name, stats in aggregated[cat_key].items():
                print(f"    {metric_name}: {stats['mean']:.4f}")

    print(f"\nTotal evaluation time: {total_time:.1f}s")
    print(f"Average time per QA: {total_time / max(len(all_results), 1):.2f}s")

    # Save results
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
    print(f"\nResults saved to: {output_file}")


def main():
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215") if default_backend == "doubao" else "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="Evaluate A-Mem on LoCoMo")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to LoCoMo data")
    parser.add_argument("--backend", type=str, default=default_backend, choices=["openai", "ollama", "litellm", "doubao"])
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API base URL (auto-read from .env)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of memories to retrieve")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max_conversations", type=int, default=None, help="Limit conversations for testing")
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
