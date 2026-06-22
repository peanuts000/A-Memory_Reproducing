"""
AgenticMemorySystem: The core A-Mem memory management system.

Implements the three-phase memory storage pipeline:
  1. Note Construction (Section 3.1)
  2. Link Generation (Section 3.2)
  3. Memory Evolution (Section 3.3)

And the memory retrieval mechanism (Section 3.4).

Usage:
    from amem import AgenticMemorySystem

    memory = AgenticMemorySystem(
        llm_backend="openai",
        llm_model="gpt-4o-mini"
    )
    memory.add_note("The cache system works great in production.")
    results = memory.retrieve("Tell me about the cache system")
"""

import json
from typing import List, Dict, Optional, Tuple

from .memory_note import MemoryNote
from .retriever import SimpleEmbeddingRetriever
from .llm_controller import LLMController
from .prompt_templates import (
    ANALYZE_CONTENT_PROMPT,
    LINK_GENERATION_PROMPT,
    EVOLUTION_PROMPT,
    ANALYSIS_SCHEMA,
    LINK_SCHEMA,
    EVOLUTION_SCHEMA,
)
from .parsers import (
    parse_analysis_response,
    parse_link_response,
    parse_evolution_response,
)


class AgenticMemorySystem:
    """Agentic Memory System for LLM Agents.

    Implements the full A-Mem pipeline:
      - Note Construction: Generate structured notes with LLM (keywords, context, tags)
      - Link Generation: Establish connections between related memories
      - Memory Evolution: Update existing memories with new context
      - Memory Retrieval: Retrieve relevant memories using cosine similarity

    Args:
        model_name: SentenceTransformer model for embeddings.
        llm_backend: LLM backend ('openai', 'ollama', 'litellm').
        llm_model: LLM model identifier.
        evo_threshold: Trigger consolidation every N evolutions.
        api_key: Optional API key for the LLM backend.
        api_base: Optional API base URL.
        top_k: Default number of nearest neighbors for operations.
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
    ):
        self.memories: Dict[str, MemoryNote] = {}
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

        This triggers the full pipeline:
          1. Note Construction (generate keywords, context, tags via LLM)
          2. Link Generation (find and establish connections)
          3. Memory Evolution (update related memories)

        Args:
            content: The raw interaction content.
            time: Optional timestamp string.
            **kwargs: Additional keyword arguments for MemoryNote.

        Returns:
            The ID of the newly created memory note.
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
        """Retrieve the top-k most relevant memories for a query.

        Implements the memory retrieval from Section 3.4:
          e_q = f_enc(q)
          s_{q,i} = (e_q · e_i) / (|e_q| |e_i|)
          M_retrieved = {m_i | rank(s_{q,i}) <= k}

        Args:
            query: The query text.
            k: Number of results to return. Defaults to self.top_k.

        Returns:
            List of the most relevant MemoryNote objects.
        """
        if not self.memories:
            return []

        k = k or self.top_k
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        results = []
        for idx in indices:
            if idx < len(all_memories):
                results.append(all_memories[idx])
        return results

    def retrieve_with_context(self, query: str, k: int = None) -> str:
        """Retrieve memories and format them as context string.

        Args:
            query: The query text.
            k: Number of results.

        Returns:
            Formatted string of retrieved memories for prompt injection.
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

        This is called periodically (every evo_threshold evolutions)
        to ensure the retriever's embeddings are up to date after
        memory evolution has changed contexts and tags.
        """
        try:
            model_name = self.retriever.model.get_config_dict()["model_name"]
        except (AttributeError, KeyError):
            model_name = "all-MiniLM-L6-v2"

        self.retriever = SimpleEmbeddingRetriever(model_name)
        docs = [self._note_to_document(m) for m in self.memories.values()]
        if docs:
            self.retriever.add_documents(docs)

    # -----------------------------------------------------------------------
    # Internal Pipeline
    # -----------------------------------------------------------------------

    def _construct_note(self, note: MemoryNote) -> MemoryNote:
        """Note Construction (Section 3.1).

        Uses LLM to generate keywords (K_i), context (X_i), and tags (G_i).
        Then computes the embedding vector (e_i).

        K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ Ps1)
        e_i = f_enc[ concat(c_i, K_i, G_i, X_i) ]
        """
        prompt = ANALYZE_CONTENT_PROMPT.format(content=note.content)

        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=ANALYSIS_SCHEMA, temperature=0.3
            )
            analysis = parse_analysis_response(response, note.content)
        except Exception as e:
            print(f"Error in note construction: {e}")
            analysis = {"keywords": [], "context": "", "tags": []}

        note.keywords = analysis["keywords"]
        note.context = analysis["context"]
        note.tags = analysis["tags"]

        return note

    def _process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note through Link Generation and Memory Evolution.

        Args:
            note: The newly constructed memory note.

        Returns:
            True if evolution occurred, False otherwise.
        """
        # Find nearest neighbors
        neighbor_str, indices = self._find_related_memories(note.content, self.top_k)

        if not indices:
            return False

        # Combined Link Generation + Evolution prompt
        prompt = EVOLUTION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_str,
            neighbor_number=len(indices),
        )

        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=EVOLUTION_SCHEMA, temperature=0.3
            )
            evolution_result = parse_evolution_response(response, len(indices))
        except Exception as e:
            print(f"Error in memory evolution: {e}")
            return False

        should_evolve = evolution_result["should_evolve"]
        if not should_evolve:
            return False

        actions = evolution_result["actions"]

        # Strengthen: add links and update tags for the new memory
        if "strengthen" in actions:
            connections = evolution_result["suggested_connections"]
            new_tags = evolution_result["tags_to_update"]
            note.links.extend(connections)
            if new_tags:
                note.tags = new_tags

        # Update Neighbor: update context and tags of existing memories
        if "update_neighbor" in actions:
            new_contexts = evolution_result["new_context_neighborhood"]
            new_tags_list = evolution_result["new_tags_neighborhood"]
            notes_list = list(self.memories.values())
            notes_ids = list(self.memories.keys())

            for i in range(min(len(indices), len(new_tags_list))):
                memory_idx = indices[i]
                if memory_idx >= len(notes_list):
                    continue

                tag = new_tags_list[i]
                context = new_contexts[i] if i < len(new_contexts) else ""

                notetmp = notes_list[memory_idx]
                if tag:
                    notetmp.tags = tag
                if context:
                    notetmp.context = context
                self.memories[notes_ids[memory_idx]] = notetmp

        return True

    def _find_related_memories(
        self, query: str, k: int = 5
    ) -> Tuple[str, List[int]]:
        """Find related memories using the retriever.

        Args:
            query: The query text.
            k: Number of neighbors.

        Returns:
            Tuple of (formatted memory string, list of indices).
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
                f"memory index: {i}\t"
                f"talk start time: {m.timestamp}\t"
                f"memory content: {m.content}\t"
                f"memory context: {m.context}\t"
                f"memory keywords: {m.keywords}\t"
                f"memory tags: {m.tags}\n"
            )
        return memory_str, indices

    def _find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories including linked neighbors.

        When a memory is retrieved, its linked neighbors are also accessed.

        Args:
            query: The query text.
            k: Number of top results.

        Returns:
            Formatted string of all related memories including links.
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
                f"talk start time: {m.timestamp}\t"
                f"memory content: {m.content}\t"
                f"memory context: {m.context}\t"
                f"memory keywords: {m.keywords}\t"
                f"memory tags: {m.tags}\n"
            )
            # Include linked neighbors
            for neighbor_id in m.links:
                if neighbor_id in self.memories:
                    nm = self.memories[neighbor_id]
                    memory_str += (
                        f"  [linked] talk start time: {nm.timestamp}\t"
                        f"memory content: {nm.content}\t"
                        f"memory context: {nm.context}\t"
                        f"memory keywords: {nm.keywords}\t"
                        f"memory tags: {nm.tags}\n"
                    )
        return memory_str

    @staticmethod
    def _note_to_document(note: MemoryNote) -> str:
        """Convert a memory note to a document string for the retriever.

        Combines content, context, keywords, and tags as specified in the paper.
        """
        return (
            f"content: {note.content} "
            f"context: {note.context} "
            f"keywords: {', '.join(note.keywords)} "
            f"tags: {', '.join(note.tags)}"
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the memory system state to disk.

        Args:
            path: Directory path to save the state.
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
            path: Directory path where state was saved.
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

        system = cls(
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            evo_threshold=meta.get("evo_threshold", 100),
            top_k=meta.get("top_k", 5),
        )
        system.evo_cnt = meta.get("evo_cnt", 0)

        # Load memories
        with open(os.path.join(path, "memories.json"), "r", encoding="utf-8") as f:
            memories_data = json.load(f)
        system.memories = {
            mid: MemoryNote.from_dict(data) for mid, data in memories_data.items()
        }

        # Load retriever
        system.retriever = SimpleEmbeddingRetriever.load(
            os.path.join(path, "retriever_cache.pkl"),
            os.path.join(path, "retriever_embeddings.npy"),
        )

        return system
