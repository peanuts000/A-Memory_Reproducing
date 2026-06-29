"""
AgenticMemorySystem: Core memory management system for A-Mem.

Implements the three-phase memory storage pipeline:
  1. Note Construction (Section 3.1)
  2. Link Generation (Section 3.2)
  3. Memory Evolution (Section 3.3)

And memory retrieval mechanism (Section 3.4).

Uses plain-text LLM calls with section-marker parsing for maximum
backend compatibility (Ollama, SGLang, vLLM, OpenAI, etc.).

Usage:
    from amem import AgenticMemorySystem

    memory = AgenticMemorySystem(
        llm_backend="openai",
        llm_model="gpt-4o-mini"
    )
    memory.add_note("The caching system works well in production.")
    results = memory.retrieve("Tell me about the caching system")
"""

import json
import logging
from typing import List, Dict, Optional, Tuple

from .memory_note import MemoryNote
from .retriever import SimpleEmbeddingRetriever, HybridRetriever
from .llm_controller import LLMController
from .prompt_templates import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
)
from .parsers import (
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
    _parse_list_items,
    _heuristic_keywords,
    _heuristic_context,
)

logger = logging.getLogger("amem")


class AgenticMemorySystem:
    """Intelligent memory system for LLM Agents.

    Implements the complete A-Mem pipeline:
      - Note Construction: Generate structured notes via LLM (keywords, context, tags)
      - Link Generation: Find and establish connections between semantically similar memories
      - Memory Evolution: Update existing memories based on new information
      - Memory Retrieval: Retrieve relevant memories using cosine similarity

    Args:
        model_name: SentenceTransformer model for embeddings.
        llm_backend: LLM backend ('openai', 'ollama', 'litellm', 'doubao', 'sglang', 'vllm').
        llm_model: LLM model identifier.
        evo_threshold: Trigger consolidation every N evolutions.
        api_key: Optional LLM backend API key.
        api_base: Optional API base URL.
        top_k: Default number of nearest neighbors.
        use_hybrid: Whether to use BM25+semantic hybrid retrieval (default True).
        hybrid_alpha: Semantic retrieval weight in hybrid mode (0-1, default 0.6).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = None,
        llm_model: str = None,
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.6,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        if use_hybrid:
            self.retriever = HybridRetriever(model_name, alpha=hybrid_alpha)
        else:
            self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.top_k = top_k

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note to the system.

        Triggers the full pipeline:
          1. Note Construction (generate keywords, context, tags via LLM)
          2. Link Generation + Memory Evolution (3-step conditional evolution)

        Args:
            content: Raw interaction content.
            time: Optional timestamp string.
            **kwargs: Additional keyword arguments for MemoryNote.

        Returns:
            The newly created memory note ID.
        """
        # Step 1: Note Construction
        note = MemoryNote(content=content, timestamp=time, **kwargs)
        note = self._construct_note(note)

        # Step 2 & 3: Link Generation + Memory Evolution
        evolved = self._process_memory(note)

        # Store the note
        self.memories[note.id] = note

        # Add to retriever index
        doc = self._note_to_document(note)
        self.retriever.add_documents([doc])

        # Track evolution count for consolidation
        if evolved:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        return note.id

    def retrieve(self, query: str, k: int = None) -> List[MemoryNote]:
        """Retrieve top-k most relevant memories.

        Implements Section 3.4 memory retrieval:
          e_q = f_enc(q)
          s_{q,i} = (e_q · e_i) / (|e_q| |e_i|)
          M_retrieved = {m_i | rank(s_{q,i}) <= k}

        Records access count and time for each returned memory.

        Args:
            query: Query text.
            k: Number of results to return. Defaults to self.top_k.

        Returns:
            List of most relevant MemoryNote objects.
        """
        if not self.memories:
            return []

        k = k or self.top_k
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        results = []
        for idx in indices:
            if idx < len(all_memories):
                m = all_memories[idx]
                m.record_access()
                results.append(m)
        return results

    def retrieve_with_context(self, query: str, k: int = None) -> str:
        """Retrieve memories and format as context string.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            Formatted memory context string for prompt injection.
        """
        memories = self.retrieve(query, k)
        if not memories:
            return ""

        context_parts = []
        for m in memories:
            part = (
                f"talk start time: {m.timestamp}\n"
                f"memory content: {m.content}\n"
                f"memory context: {m.context}\n"
                f"memory keywords: {m.keywords}\n"
                f"memory tags: {m.tags}\n"
            )
            context_parts.append(part)
        return "\n---\n".join(context_parts)

    def get_memory_count(self) -> int:
        """Return the number of stored memories."""
        return len(self.memories)

    def consolidate_memories(self) -> None:
        """Rebuild the retriever index from all current memories.

        Called every evo_threshold evolutions to ensure the retriever
        embeddings are up-to-date after evolution changes context and tags.
        """
        try:
            model_name = self.retriever.model.get_config_dict()["model_name"]
        except (AttributeError, KeyError):
            model_name = "all-MiniLM-L6-v2"

        if self.use_hybrid:
            self.retriever = HybridRetriever(model_name, alpha=self.hybrid_alpha)
        else:
            self.retriever = SimpleEmbeddingRetriever(model_name)
        docs = [self._note_to_document(m) for m in self.memories.values()]
        if docs:
            self.retriever.add_documents(docs)

    # -----------------------------------------------------------------------
    # Internal pipeline
    # -----------------------------------------------------------------------

    def _construct_note(self, note: MemoryNote) -> MemoryNote:
        """Note Construction (Section 3.1).

        Uses LLM to generate keywords (K_i), context (X_i), and tags (G_i).
        Then computes embedding vector (e_i).

        K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ Ps1)
        e_i = f_enc[ concat(c_i, K_i, G_i, X_i) ]
        """
        prompt = ANALYZE_CONTENT_PROMPT.format(content=note.content)

        try:
            response = self.llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, note.content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=note.content)
                retry_response = self.llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, note.content)

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            analysis = {
                "keywords": _heuristic_keywords(note.content),
                "context": _heuristic_context(note.content),
                "tags": _heuristic_keywords(note.content, 3),
            }

        note.keywords = analysis["keywords"]
        note.context = analysis["context"]
        note.tags = analysis["tags"]

        return note

    def _process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)

        Args:
            note: Newly constructed memory note.

        Returns:
            True if evolution occurred, False otherwise.
        """
        neighbor_memory, indices = self._find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")
            evolved = False

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                try:
                    strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                        content=note.content,
                        keywords=note.keywords,
                        nearest_neighbors_memories=neighbor_memory,
                    )
                    strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                    strengthen = parse_strengthen_details(strengthen_response)
                    logger.debug("Strengthen details: %s", strengthen)

                    note.links.extend(strengthen["connections"])
                    if strengthen["tags"]:
                        note.tags = strengthen["tags"]
                    note.record_evolution("strengthen", f"connections: {len(strengthen['connections'])}")
                    evolved = True
                except Exception as e:
                    logger.error("Strengthen details error: %s — skipping strengthen step", e)

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                try:
                    update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                        content=note.content,
                        context=note.context,
                        nearest_neighbors_memories=neighbor_memory,
                        max_neighbor_idx=len(indices) - 1,
                        neighbor_count=len(indices),
                    )
                    update_response = self.llm_controller.llm.get_completion(update_prompt)
                    neighbor_updates = parse_update_neighbors(update_response, len(indices))
                    logger.debug("Neighbor updates: %s", neighbor_updates)

                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    updated_count = 0
                    for i in range(min(len(indices), len(neighbor_updates))):
                        upd = neighbor_updates[i]
                        memorytmp_idx = indices[i]
                        if memorytmp_idx >= len(noteslist):
                            continue
                        notetmp = noteslist[memorytmp_idx]
                        if upd["tags"]:
                            notetmp.tags = upd["tags"]
                        if upd["context"]:
                            notetmp.context = upd["context"]
                        self.memories[notes_id[memorytmp_idx]] = notetmp
                        updated_count += 1
                    note.record_evolution("update_neighbor", f"updated {updated_count} neighbors")
                    evolved = True
                except Exception as e:
                    logger.error("Update neighbors error: %s — skipping neighbor update step", e)

            return evolved

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False

    def _find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using the retriever.

        Args:
            query: Query text.
            k: Number of neighbors.

        Returns:
            Tuple of (formatted memory string, index list).
        """
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + m.timestamp +
                "\t memory content: " + m.content +
                "\t memory context: " + m.context +
                "\t memory keywords: " + str(m.keywords) +
                "\t memory tags: " + str(m.tags) + "\n"
            )
        return memory_str, indices

    def _find_related_memories_with_scores(
        self, query: str, k: int = 5
    ) -> Tuple[str, List[int], List[float]]:
        """Find related memories with similarity scores.

        Args:
            query: Query text.
            k: Number of neighbors.

        Returns:
            Tuple of (formatted memory string, index list, score list).
        """
        if not self.memories:
            return "", [], []

        results = self.retriever.search_with_scores(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        indices = []
        scores = []
        for idx, score in results:
            if idx >= len(all_memories):
                continue
            m = all_memories[idx]
            memory_str += (
                "memory index:" + str(idx) +
                "\t similarity:" + f"{score:.3f}" +
                "\t talk start time:" + m.timestamp +
                "\t memory content: " + m.content +
                "\t memory context: " + m.context +
                "\t memory keywords: " + str(m.keywords) +
                "\t memory tags: " + str(m.tags) + "\n"
            )
            indices.append(idx)
            scores.append(score)
        return memory_str, indices, scores

    def _find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with linked neighbors expanded.

        When a memory is retrieved, its linked neighbors are also visited.

        Args:
            query: Query text.
            k: Top-k result count.

        Returns:
            Formatted related memories string with links included.
        """
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""

        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                "talk start time:" + m.timestamp +
                " memory content: " + m.content +
                " memory context: " + m.context +
                " memory keywords: " + str(m.keywords) +
                " memory tags: " + str(m.tags) + "\n"
            )
            # Include linked neighbors
            for neighbor_id in m.links:
                if neighbor_id in self.memories:
                    nm = self.memories[neighbor_id]
                    memory_str += (
                        "  [link] talk start time:" + nm.timestamp +
                        " memory content: " + nm.content +
                        " memory context: " + nm.context +
                        " memory keywords: " + str(nm.keywords) +
                        " memory tags: " + str(nm.tags) + "\n"
                    )
        return memory_str

    @staticmethod
    def _note_to_document(note: MemoryNote) -> str:
        """Convert a memory note to a document string for the retriever.

        Combines content, context, keywords, and tags as per the paper.
        Includes timestamp for better BM25 matching on temporal queries.
        """
        return (
            f"content: {note.content} "
            f"timestamp: {note.timestamp} "
            f"context: {note.context} "
            f"keywords: {', '.join(note.keywords)} "
            f"tags: {', '.join(note.tags)}"
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save memory system state to disk.

        Args:
            path: Directory path to save state.
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save memories
        memories_data = {
            mid: note.to_dict() for mid, note in self.memories.items()
        }
        with open(os.path.join(path, "memories.json"), "w", encoding="utf-8") as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)

        # Save retriever
        self.retriever.save(
            os.path.join(path, "retriever_cache.pkl"),
            os.path.join(path, "retriever_embeddings.npy"),
        )

        # Save metadata
        meta = {
            "evo_cnt": self.evo_cnt,
            "evo_threshold": self.evo_threshold,
            "top_k": self.top_k,
            "memory_count": len(self.memories),
            "use_hybrid": self.use_hybrid,
            "hybrid_alpha": self.hybrid_alpha,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> "AgenticMemorySystem":
        """Load a saved memory system from disk.

        Args:
            path: Directory path of saved state.
            llm_backend: LLM backend for the restored system.
            llm_model: LLM model for the restored system.
            api_key: Optional API key.

        Returns:
            Restored AgenticMemorySystem instance.
        """
        import os

        # Load metadata
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        use_hybrid = meta.get("use_hybrid", True)
        hybrid_alpha = meta.get("hybrid_alpha", 0.5)

        system = cls(
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            evo_threshold=meta.get("evo_threshold", 100),
            top_k=meta.get("top_k", 5),
            use_hybrid=use_hybrid,
            hybrid_alpha=hybrid_alpha,
        )
        system.evo_cnt = meta.get("evo_cnt", 0)

        # Load memories
        with open(os.path.join(path, "memories.json"), "r", encoding="utf-8") as f:
            memories_data = json.load(f)
        system.memories = {
            mid: MemoryNote.from_dict(data) for mid, data in memories_data.items()
        }

        # Load retriever
        if use_hybrid:
            system.retriever = HybridRetriever.load(
                os.path.join(path, "retriever_cache.pkl"),
                os.path.join(path, "retriever_embeddings.npy"),
            )
        else:
            system.retriever = SimpleEmbeddingRetriever.load(
                os.path.join(path, "retriever_cache.pkl"),
                os.path.join(path, "retriever_embeddings.npy"),
            )

        return system
