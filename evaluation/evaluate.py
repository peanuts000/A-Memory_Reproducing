"""
Evaluation script for A-Mem on the LoCoMo dataset.

Reproduces the evaluation from the paper (Section 4):
  - Dataset: LoCoMo (5 categories, 7,512 QA pairs)
  - Metrics: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT similarity
  - Categories: Single-hop, Multi-hop, Temporal, Open-domain, Adversarial

Enhancements:
  - LLM query keyword generation: Extract retrieval keywords from questions
  - LLM memory filtering: Filter most relevant parts from retrieval results
  - Category-specific prompts: Different prompts for different question types
  - Memory caching: Avoid redundant memory construction

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
import pickle
import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem, LLMController
from amem.parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
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

logger = logging.getLogger("amem_eval")


class MemoryAgent:
    """Enhanced memory agent wrapping AgenticMemorySystem.

    Enhancements:
      - Multi-query retrieval: Generate multiple query variants for better recall
      - LLM reranking: Score and filter retrieval results by relevance
      - Category-specific prompts: Different answer prompts per question type
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
        hybrid_alpha: float = 0.6,
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
        """Add a memory."""
        self.memory_system.add_note(content, time=time)

    def generate_query_variants(self, question: str) -> List[str]:
        """Generate multiple query variants using LLM for better retrieval recall.

        Args:
            question: User question.

        Returns:
            List of query variants (including the original question).
        """
        prompt = f"""Given the following question, generate 3 different search queries to find relevant conversation memories.

Question: {question}

Each query should approach the question from a different angle:
1. Query 1: Focus on the main entity/concept (use exact names)
2. Query 2: Focus on the action/event being asked about
3. Query 3: Focus on the context/situation described

Respond with a JSON object:
{{"queries": ["query1", "query2", "query3"]}}"""

        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.3)
            result = json.loads(response.strip().strip("`").replace("json\n", "").replace("```", ""))
            queries = result.get("queries", [question])
            if isinstance(queries, list) and queries:
                # Ensure the original question is in the list
                if question not in queries:
                    queries.insert(0, question)
                return queries[:5]  # Max 5 queries
        except Exception:
            pass
        return [question]

    def retrieve_memory_multi_query(self, question: str, k: int = None) -> str:
        """Retrieve memories using multi-query strategy, merge and deduplicate.

        Args:
            question: User question.
            k: Number of results per query.

        Returns:
            Merged and deduplicated formatted memory text.
        """
        k = k or self.retrieve_k
        queries = self.generate_query_variants(question)

        # Collect all retrieval results (larger candidate pool)
        candidate_k = min(k * 2, 20)
        all_memories = list(self.memory_system.memories.values())
        seen_indices = set()
        memory_entries = []

        for query in queries:
            results = self.memory_system.retriever.search_with_scores(query, candidate_k)
            for idx, score in results:
                if idx < len(all_memories) and idx not in seen_indices:
                    seen_indices.add(idx)
                    m = all_memories[idx]
                    memory_entries.append((idx, score, m))

        # Sort by score, take top-k
        memory_entries.sort(key=lambda x: x[1], reverse=True)
        top_entries = memory_entries[:k]

        # Format output
        memory_str = ""
        for idx, score, m in top_entries:
            memory_str += (
                f"talk start time: {m.timestamp}\t"
                f"memory content: {m.content}\t"
                f"memory context: {m.context}\t"
                f"memory keywords: {m.keywords}\t"
                f"memory tags: {m.tags}\n"
            )
            # Include linked neighbors
            for neighbor_id in m.links:
                if neighbor_id in self.memory_system.memories:
                    nm = self.memory_system.memories[neighbor_id]
                    memory_str += (
                        f"  [link] talk start time: {nm.timestamp}\t"
                        f"memory content: {nm.content}\t"
                        f"memory context: {nm.context}\n"
                    )

        return memory_str

    def rerank_memories(self, memories_text: str, question: str, top_n: int = 10) -> str:
        """Rerank retrieved memories using LLM, keeping the most relevant parts.

        Args:
            memories_text: Retrieved memory text.
            question: User question.
            top_n: Number of memory entries to keep.

        Returns:
            Filtered relevant memory text.
        """
        if not memories_text.strip():
            return memories_text

        # Split memory text into individual entries
        lines = memories_text.strip().split("\n")
        memory_entries = []
        current_entry = ""

        for line in lines:
            if line.startswith("talk start time:"):
                if current_entry:
                    memory_entries.append(current_entry)
                current_entry = line
            elif line.strip():
                current_entry += "\n" + line
        if current_entry:
            memory_entries.append(current_entry)

        if len(memory_entries) <= top_n:
            return memories_text

        # Use LLM for reranking
        entries_text = ""
        for i, entry in enumerate(memory_entries):
            entries_text += f"[{i}] {entry}\n\n"

        prompt = f"""Given the following conversation memories and a question, rank each memory by relevance (0-10 score).

Question: {question}

Memories:
{entries_text}

For each memory, assign a relevance score from 0 (irrelevant) to 10 (directly answers the question).
Respond with a JSON object mapping memory index to score:
{{"scores": {{"0": 8, "1": 3, "2": 9, ...}}}}"""

        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            result = json.loads(response.strip().strip("`").replace("json\n", "").replace("```", ""))
            scores = result.get("scores", {})

            # Sort by score
            scored_entries = []
            for i, entry in enumerate(memory_entries):
                score = scores.get(str(i), scores.get(i, 0))
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0
                scored_entries.append((score, entry))

            scored_entries.sort(key=lambda x: x[0], reverse=True)

            # Return top_n
            selected = [entry for _, entry in scored_entries[:top_n]]
            return "\n".join(selected)
        except Exception:
            # Fallback: return original text
            return memories_text

    def retrieve_memory(self, query: str, k: int = None) -> str:
        """Retrieve related memories and return formatted string (legacy interface).

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            Formatted memory text.
        """
        k = k or self.retrieve_k
        return self.memory_system._find_related_memories_raw(query, k)

    def retrieve_memory_llm(self, memories_text: str, query: str) -> str:
        """Select relevant parts of conversation memories using LLM.

        Args:
            memories_text: Retrieved memory text.
            query: User question.

        Returns:
            Filtered relevant memory text.
        """
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        try:
            response = self.llm_controller.llm.get_completion(prompt)
            return parse_relevant_parts(response)
        except Exception as e:
            logger.warning("retrieve_memory_llm failed: %s — returning original", e)
            return memories_text

    def generate_query_llm(self, question: str) -> str:
        """Generate query keywords using LLM.

        Args:
            question: User question.

        Returns:
            Comma-separated keywords string.
        """
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        try:
            response = self.llm_controller.llm.get_completion(prompt)
            result = parse_keywords_response(response)
            logger.debug("generate_query_llm response: %s", result)
            return result
        except Exception as e:
            logger.warning("generate_query_llm failed: %s — returning question", e)
            return question

    def answer_question(
        self, question: str, category: int, use_llm_filter: bool = False
    ) -> Tuple[str, str, str]:
        """Generate an answer for a question.

        Args:
            question: Question text.
            category: Question category (0-4).
            use_llm_filter: Whether to use LLM to filter memories.

        Returns:
            (answer, prompt, raw_context) tuple.
        """
        # Use multi-query strategy to retrieve relevant memories
        raw_context = self.retrieve_memory_multi_query(question, k=self.retrieve_k)

        # Use LLM reranking to filter most relevant memories
        context = raw_context
        if raw_context:
            context = self.rerank_memories(raw_context, question, top_n=self.retrieve_k)

        # Select different prompts based on category
        temperature = 0.1
        if category == 4:  # Adversarial questions
            temperature = self.temperature_c5
            user_prompt = f"""Based ONLY on the following conversation memories, answer the question.

Conversation memories:
{context}

Question: {question}

RULES:
- If the answer IS in the memories, provide a short answer using exact words from the memories.
- If the answer is NOT in the memories, respond with exactly: "Not mentioned"
- Do NOT guess or infer information not explicitly stated.
- Do NOT output JSON format.

Short answer:"""
        elif category == 2:  # Temporal questions
            user_prompt = f"""Based ONLY on the following conversation memories, answer the question about WHEN something happened.

Conversation memories:
{context}

Question: {question}

RULES:
- Use the CONVERSATION DATE (talk start time) in the memories to calculate the absolute date.
- Answer with a specific date format: "DD Month YYYY" or "Month YYYY" or "YYYY"
- Do NOT use relative terms like "yesterday", "last week", "next month"
- If the date cannot be determined, respond with: "Not mentioned"
- Use exact words from the memories when possible.

Short answer:"""
        elif category == 1:  # Multi-hop questions
            user_prompt = f"""Based ONLY on the following conversation memories, answer the question.

Conversation memories:
{context}

Question: {question}

RULES:
- This question may require combining information from MULTIPLE conversation sessions.
- Look through ALL the memories carefully for relevant information.
- Answer with a short phrase using exact words from the memories.
- If the information is not in the memories, respond with: "Not mentioned"
- Do NOT output JSON format.

Short answer:"""
        else:  # Single-hop (0) and Open-domain (3)
            user_prompt = f"""Based ONLY on the following conversation memories, answer the question.

Conversation memories:
{context}

Question: {question}

RULES:
- Find the EXACT answer from the memories.
- Answer with a short phrase using exact words from the memories.
- If the information is not in the memories, respond with: "Not mentioned"
- Do NOT output JSON format.
- Do NOT add explanations.

Short answer:"""

        try:
            response = self.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature
            )
            prediction = response.strip()
            # Try to parse JSON-formatted answers (some LLMs return JSON)
            try:
                result = json.loads(prediction.strip().strip("`").replace("json\n", "").replace("```", ""))
                prediction = result.get("answer", result.get("response", prediction))
            except (json.JSONDecodeError, ValueError):
                pass
            # Clean answer: remove extra quotes and whitespace
            prediction = prediction.strip().strip('"').strip("'")
        except Exception as e:
            logger.error("Error generating answer: %s", e)
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
    """Evaluate A-Mem on a single conversation.

    Args:
        conversation: Conversation to evaluate.
        agent: Initialized memory agent.
        use_llm_filter: Whether to use LLM to filter memories.
        use_memory_cache: Whether to use memory caching.
        cache_dir: Cache directory.
        sample_idx: Sample index (for cache file naming).

    Returns:
        List of result dicts with predictions, references, and metrics.
    """
    results = []

    # Try loading cached memories
    memory_loaded = False
    if use_memory_cache and cache_dir:
        cache_file = os.path.join(cache_dir, f"memory_cache_sample_{sample_idx}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_memories = pickle.load(f)
                agent.memory_system.memories = cached_memories
                # Rebuild retriever index
                agent.memory_system.consolidate_memories()
                logger.info("Loaded %d memories from cache", len(cached_memories))
                memory_loaded = True
            except Exception as e:
                logger.warning("Failed to load cache: %s — rebuilding memories", e)

    # If no cache, ingest conversation turns
    if not memory_loaded:
        total_turns = sum(len(s.turns) for s in conversation.sessions)
        logger.info("Ingesting %d conversation turns...", total_turns)
        for session in conversation.sessions:
            for turn in session.turns:
                content = f"Speaker {turn.speaker} says: {turn.content}"
                agent.add_memory(content, time=turn.timestamp)

        # Save cache
        if use_memory_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"memory_cache_sample_{sample_idx}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            logger.info("Cached %d memories", len(agent.memory_system.memories))

    logger.info("Total memories: %d", agent.memory_system.get_memory_count())

    # Answer questions using retrieved memories
    for qa in conversation.qa_pairs:
        prediction, user_prompt, raw_context = agent.answer_question(
            qa.question, qa.category, use_llm_filter=use_llm_filter
        )

        # Calculate metrics
        metrics = calculate_all_metrics(prediction, qa.answer)

        results.append({
            "question": qa.question,
            "reference": qa.answer,
            "prediction": prediction,
            "category": qa.category,
            "category_name": qa.category_name or CATEGORY_NAMES.get(qa.category, "Unknown"),
            "metrics": metrics,
            "raw_context": raw_context[:500],  # Truncate to save space
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
    hybrid_alpha: float = 0.6,
    use_llm_filter: bool = False,
    use_memory_cache: bool = True,
    temperature_c5: float = 0.5,
):
    """Run full evaluation on the LoCoMo dataset.

    Args:
        data_dir: Path to LoCoMo data directory.
        backend: LLM backend.
        model: LLM model.
        api_key: Optional API key.
        top_k: Number of memories to retrieve.
        output_file: Path to save results.
        max_conversations: Limit conversation count (for testing).
        use_hybrid: Whether to use hybrid retrieval.
        hybrid_alpha: Hybrid retrieval weight.
        use_llm_filter: Whether to use LLM to filter memories.
        use_memory_cache: Whether to use memory caching.
        temperature_c5: Temperature for adversarial questions.
    """
    print("=" * 60)
    print("A-Mem Evaluation on LoCoMo Dataset")
    print("=" * 60)
    print(f"Backend: {backend}, Model: {model}, Top-k: {top_k}")
    print(f"Hybrid retrieval: {use_hybrid}, Alpha: {hybrid_alpha}")
    print(f"LLM filter: {use_llm_filter}, Memory cache: {use_memory_cache}")
    print()

    # Load dataset
    conversations = load_locomo_dataset(data_dir)
    if not conversations:
        print(f"No conversation data found in {data_dir}")
        return

    if max_conversations:
        conversations = conversations[:max_conversations]

    print(f"Loaded {len(conversations)} conversations")

    # Cache directory
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        f"cached_memories_{backend}_{model.replace('/', '_')}"
    ) if use_memory_cache else None

    all_results = []
    all_metrics = []
    all_categories = []
    total_time = 0

    for i, conv in enumerate(conversations):
        print(f"\nEvaluating conversation {i + 1}/{len(conversations)}: {conv.conversation_id}")

        # Create new memory agent for each conversation
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
        print(f"  Completed in {elapsed:.1f}s, {len(results)} QA pairs")

    # Aggregate results
    aggregated = aggregate_metrics(all_metrics, all_categories)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    overall = aggregated.get("overall", {})
    print("\nOverall:")
    for metric_name, stats in overall.items():
        print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")

    print("\nBy Category:")
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
    print(f"\nResults saved to: {output_file}")


def main():
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = (
        os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215")
        if default_backend == "doubao"
        else "gpt-4o-mini"
    )

    parser = argparse.ArgumentParser(description="Evaluate A-Mem on LoCoMo")
    parser.add_argument("--data_dir", type=str, required=True, help="LoCoMo data path")
    parser.add_argument(
        "--backend",
        type=str,
        default=default_backend,
        choices=["openai", "ollama", "litellm", "doubao", "deepseek", "sglang", "vllm"],
    )
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL")
    parser.add_argument("--top_k", type=int, default=10, help="Number of memories to retrieve")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max_conversations", type=int, default=None, help="Limit conversation count for testing")
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid retrieval, use pure semantic")
    parser.add_argument("--hybrid_alpha", type=float, default=0.6, help="Hybrid retrieval semantic weight (0-1)")
    parser.add_argument("--use_llm_filter", action="store_true", help="Use LLM to filter memories")
    parser.add_argument("--no_cache", action="store_true", help="Disable memory caching")
    parser.add_argument("--temperature_c5", type=float, default=0.5, help="Adversarial question temperature")
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
