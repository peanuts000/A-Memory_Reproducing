"""
MemoryNote: The atomic memory unit in the A-Mem system.

Each memory note follows the Zettelkasten atomic note principle,
containing structured attributes: content, timestamp, keywords, tags,
context, links, and embedding (Equation 1 in the paper).

m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}

Extended attributes (not from the paper, for enhanced memory management):
  - importance_score: Memory importance score
  - retrieval_count: Number of times retrieved
  - last_accessed: Last access timestamp
  - evolution_history: Evolution history log
  - category: Category label
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any


class MemoryNote:
    """Basic memory unit with structured metadata.

    Attributes:
        id: Unique identifier for the memory note.
        content: Raw interaction content (c_i).
        timestamp: Interaction time (t_i).
        keywords: LLM-generated keywords capturing key concepts (K_i).
        tags: LLM-generated classification tags (G_i).
        context: LLM-generated context description (X_i).
        links: Set of linked memory IDs sharing semantic relationships (L_i).
        embedding: Dense vector representation for similarity matching (e_i).
        importance_score: Memory importance score (default 1.0).
        retrieval_count: Number of times retrieved.
        last_accessed: Last access timestamp.
        evolution_history: Evolution history log.
        category: Category label.
    """

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        links: Optional[List[int]] = None,
        embedding: Optional[List[float]] = None,
        timestamp: Optional[str] = None,
        importance_score: Optional[float] = None,
        retrieval_count: Optional[int] = None,
        last_accessed: Optional[str] = None,
        evolution_history: Optional[List[Dict]] = None,
        category: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M")
        self.keywords = keywords or []
        self.tags = tags or []
        self.context = context or "General"
        self.links = links or []
        self.embedding = embedding

        # Extended fields
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        self.last_accessed = last_accessed or datetime.now().strftime("%Y%m%d%H%M")
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"

    def record_access(self) -> None:
        """Record an access event, updating count and timestamp."""
        self.retrieval_count += 1
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M")

    def record_evolution(self, action: str, details: str = "") -> None:
        """Record an evolution event.

        Args:
            action: Evolution type (e.g. 'strengthen', 'update_neighbor').
            details: Evolution details.
        """
        self.evolution_history.append({
            "timestamp": datetime.now().strftime("%Y%m%d%H%M"),
            "action": action,
            "details": details,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory note to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "keywords": self.keywords,
            "tags": self.tags,
            "context": self.context,
            "links": self.links,
            "embedding": self.embedding,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "last_accessed": self.last_accessed,
            "evolution_history": self.evolution_history,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNote":
        """Deserialize a memory note from a dictionary."""
        return cls(
            id=data.get("id"),
            content=data["content"],
            timestamp=data.get("timestamp"),
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            context=data.get("context", ""),
            links=data.get("links", []),
            embedding=data.get("embedding"),
            importance_score=data.get("importance_score", 1.0),
            retrieval_count=data.get("retrieval_count", 0),
            last_accessed=data.get("last_accessed"),
            evolution_history=data.get("evolution_history", []),
            category=data.get("category", "Uncategorized"),
        )

    def get_embedding_text(self) -> str:
        """Get the combined text for embedding computation (Equation 3).

        e_i = f_enc[ concat(c_i, K_i, G_i, X_i) ]
        """
        parts = [self.content]
        if self.keywords:
            parts.append(", ".join(self.keywords))
        if self.tags:
            parts.append(", ".join(self.tags))
        if self.context:
            parts.append(self.context)
        return " , ".join(parts)

    def __repr__(self) -> str:
        return (
            f"MemoryNote(id={self.id[:8]}..., "
            f"content={self.content[:50]}..., "
            f"keywords={self.keywords}, "
            f"tags={self.tags})"
        )
