"""
MemoryNote: A-Mem 系统中的原子记忆单元。

每个记忆笔记遵循 Zettelkasten 的原子笔记原则，
包含结构化属性：内容、时间戳、关键词、标签、上下文、链接和嵌入向量（论文中的公式 1）。

m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any


class MemoryNote:
    """基础记忆单元，包含结构化元数据。

    属性:
        id: 记忆笔记的唯一标识符。
        content: 原始交互内容 (c_i)。
        timestamp: 交互时间 (t_i)。
        keywords: LLM 生成的关键词，捕获关键概念 (K_i)。
        tags: LLM 生成的分类标签 (G_i)。
        context: LLM 生成的上下文描述 (X_i)。
        links: 共享语义关系的链接记忆 ID 集合 (L_i)。
        embedding: 用于相似度匹配的稠密向量表示 (e_i)。
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
        """将记忆笔记序列化为字典。"""
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
        """从字典反序列化记忆笔记。"""
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
        """获取用于嵌入计算的组合文本（公式 3）。

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
