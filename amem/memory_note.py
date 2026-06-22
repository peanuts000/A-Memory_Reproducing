"""
MemoryNote: The atomic memory unit in A-Mem.

Each memory note follows the Zettelkasten principle of atomic note-taking,
containing structured attributes: content, timestamp, keywords, tags,
context, links, and embedding vector (Equation 1 in the paper).

m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any


class MemoryNote:
    """Basic memory unit with structured metadata.

    Attributes:
        id: Unique identifier for the memory note.
        content: Original interaction content (c_i).
        timestamp: Time of interaction (t_i).
        keywords: LLM-generated keywords capturing key concepts (K_i).
        tags: LLM-generated categorical tags (G_i).
        context: LLM-generated contextual description (X_i).
        links: Set of linked memory IDs sharing semantic relationships (L_i).
        embedding: Dense vector representation for similarity matching (e_i).
    """

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        links: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        timestamp: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M")
        self.keywords = keywords or []
        self.tags = tags or []
        self.context = context or ""
        self.links = links or []
        self.embedding = embedding

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
        )

    def get_embedding_text(self) -> str:
        """Get the combined text used for embedding computation (Equation 3).

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
