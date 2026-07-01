"""
Main evaluation script for A-MEM reproduction.

Usage:
    python evaluate.py --api_key YOUR_API_KEY --model deepseek-v4-flash
    python evaluate.py --api_key YOUR_API_KEY --model deepseek-v4-flash --ratio 0.1
"""

import os
import json
import argparse
import logging
import pickle
import random
from datetime import datetime
from typing import Optional
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from config import DeepSeekConfig, ExperimentConfig
from llm_controllers import LLMControllerFactory, BaseLLMController
from memory_layer import AgenticMemorySystem, parse_plain_text_answer, parse_keywords_response
from load_dataset import load_locomo_dataset, LoCoMoSample
from metrics import calculate_metrics, aggregate_metrics


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('amem_eval')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MemAgent:
    """Agent using the A-MEM memory system."""

    def __init__(self, llm_controller: BaseLLMController,
                 retrieve_k: int = 10,
                 temperature_c5: float = 0.5,
                 embedding_model: str = 'all-MiniLM-L6-v2'):

        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model,
            llm_controller=llm_controller,
        )
        self.llm_controller = llm_controller
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content: str, time: str = None):
        """Add a memory to the system."""
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content: str, k: int = 10) -> str:
        """Retrieve related memories."""
        return self.memory_system.find_related_memories_raw(content, k=k)

    def generate_query_keywords(self, question: str) -> str:
        """Generate query keywords from a question."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.llm_controller.get_completion(prompt)
        result = parse_keywords_response(response)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question."""
        keywords = self.generate_query_keywords(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            # Adversarial: binary choice
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            # Temporal: use date
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            # Open-domain
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            # Single-hop (1) and Multi-hop (4)
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.llm_controller.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logging.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(config: ExperimentConfig, llm_controller: BaseLLMController):
    """Evaluate the agent on the LoCoMo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_{config.deepseek.model}_{timestamp}.log"
    log_path = os.path.join(config.output_dir, log_filename)

    eval_logger = setup_logger(log_path)
    eval_logger.info("=" * 60)
    eval_logger.info("A-MEM Evaluation with DeepSeek")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Model: {config.deepseek.model}")
    eval_logger.info(f"Dataset: {config.dataset_path}")
    eval_logger.info(f"Retrieve k: {config.retrieve_k}")
    eval_logger.info(f"Ratio: {config.ratio}")

    # Load dataset
    eval_logger.info(f"Loading dataset from {config.dataset_path}")
    samples = load_locomo_dataset(config.dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if config.ratio < 1.0:
        num_samples = max(1, int(len(samples) * config.ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({config.ratio*100:.1f}% of dataset)")

    # Create agent
    agent = MemAgent(
        llm_controller=llm_controller,
        retrieve_k=config.retrieve_k,
        temperature_c5=config.temperature_c5,
        embedding_model=config.embedding_model,
    )

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    memories_dir = os.path.join(config.output_dir, config.memory_cache_dir)
    os.makedirs(memories_dir, exist_ok=True)

    for sample_idx, sample in enumerate(samples):
        eval_logger.info(f"\n{'='*40}")
        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        eval_logger.info(f"{'='*40}")

        # Check for cached memories
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy")

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories

            if os.path.exists(retriever_cache_file):
                eval_logger.info("Loading cached retriever")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info("Rebuilding retriever from cached memories")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, config.embedding_model
                )
            eval_logger.info(f"Loaded {len(cached_memories)} memories from cache")
        else:
            eval_logger.info(f"Creating memories for sample {sample_idx}")

            # Create new agent for each sample
            agent = MemAgent(
                llm_controller=llm_controller,
                retrieve_k=config.retrieve_k,
                temperature_c5=config.temperature_c5,
                embedding_model=config.embedding_model,
            )

            # Add conversation turns as memories
            for _, turns in tqdm(sample.conversation.sessions.items(),
                                desc=f"Adding memories for sample {sample_idx}"):
                for turn in turns.turns:
                    turn_datetime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + " says: " + turn.text
                    agent.add_memory(conversation_tmp, time=turn_datetime)

            # Cache memories
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Cached {len(memories_to_cache)} memories")

        # Answer questions
        for qa in tqdm(sample.qa, desc=f"Answering questions for sample {sample_idx}"):
            if qa.category is None or qa.category not in [1, 2, 3, 4, 5]:
                continue

            total_questions += 1
            category_counts[qa.category] += 1

            prediction, user_prompt, raw_context = agent.answer_question(
                qa.question, qa.category, qa.final_answer
            )

            prediction = parse_plain_text_answer(prediction)

            eval_logger.info(f"\nQuestion {total_questions}: {qa.question}")
            eval_logger.info(f"Prediction: {prediction}")
            eval_logger.info(f"Reference: {qa.final_answer}")
            eval_logger.info(f"Category: {qa.category}")

            metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                "bleu4": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
            }

            all_metrics.append(metrics)
            all_categories.append(qa.category)

            result = {
                "sample_id": sample_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "category": qa.category,
                "metrics": metrics,
            }
            results.append(result)

            if total_questions % 10 == 0:
                eval_logger.info(f"Processed {total_questions} questions")

    # Calculate aggregate results
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": config.deepseek.model,
        "dataset": config.dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }

    # Save results
    output_file = os.path.join(
        config.output_dir,
        f"results_{config.deepseek.model}_{timestamp}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    eval_logger.info(f"\nResults saved to {output_file}")

    # Print summary
    eval_logger.info("\n" + "=" * 60)
    eval_logger.info("Evaluation Summary")
    eval_logger.info("=" * 60)
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"  Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("\nOverall Metrics:")
    if "overall" in aggregate_results:
        for metric_name, stats in aggregate_results["overall"].items():
            eval_logger.info(f"  {metric_name}: {stats['mean']:.4f} (std: {stats['std']:.4f})")

    return final_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate A-MEM with DeepSeek on LoCoMo dataset"
    )
    parser.add_argument("--api_key", type=str, default=None,
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash",
                        help="DeepSeek model name")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                        help="DeepSeek API base URL")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence embedding model name")
    parser.add_argument("--backend", type=str, default="deepseek",
                        choices=["deepseek", "openai"],
                        help="LLM backend to use")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    # Create configuration
    config = ExperimentConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        retrieve_k=args.retrieve_k,
        embedding_model=args.embedding_model,
        ratio=args.ratio,
        temperature_c5=args.temperature_c5,
        deepseek=DeepSeekConfig(
            api_key=args.api_key or "",
            base_url=args.base_url,
            model=args.model,
        ),
    )

    # Create LLM controller
    llm_controller = LLMControllerFactory.create(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Check connectivity
    print("Checking LLM connectivity...")
    try:
        llm_controller.check_connectivity()
        print("[OK] LLM connectivity check passed")
    except Exception as e:
        print(f"[FAIL] LLM connectivity check failed: {e}")
        print("Please check your API key and network connection.")
        return

    # Run evaluation
    results = evaluate_dataset(config, llm_controller)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Total questions: {results['total_questions']}")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
