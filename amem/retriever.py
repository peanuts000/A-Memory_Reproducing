"""
SimpleEmbeddingRetriever: 基于余弦相似度的记忆检索系统。

使用 SentenceTransformer (all-MiniLM-L6-v2) 对文本进行编码，并计算
余弦相似度来检索相关记忆（论文中的公式 8-10）。

论文参考：
  e_q = f_enc(q)                          (公式 8)
  s_{q,i} = (e_q · e_i) / (|e_q| |e_i|)  (公式 9)
  M_retrieved = {m_i | rank(s_{q,i}) <= k} (公式 10)
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 为中国用户使用 Hugging Face 镜像源
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class SimpleEmbeddingRetriever:
    """基于嵌入的检索系统，使用余弦相似度。

    该检索器使用 SentenceTransformer 模型对文档进行编码，
    并基于查询与文档嵌入之间的余弦相似度检索 top-k 最相关文档。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """初始化检索器。

        参数:
            model_name: SentenceTransformer 模型名称。
                        默认使用论文指定的 'all-MiniLM-L6-v2'。
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}

    def add_documents(self, documents: List[str]) -> None:
        """向检索器索引中添加文档。

        对文档进行编码并追加到现有索引。
        如果尚无文档，则初始化索引。

        参数:
            documents: 要添加到索引的文本字符串列表。
        """
        if not documents:
            return

        new_embeddings = self.model.encode(documents, show_progress_bar=False)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.corpus = list(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.corpus.extend(documents)
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> List[int]:
        """搜索 top-k 最相似文档。

        参数:
            query: 查询文本。
            k: 返回的结果数量。

        返回:
            top-k 最相似文档的索引列表。
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """带相似度分数的搜索。

        参数:
            query: 查询文本。
            k: 返回的结果数量。

        返回:
            top-k 结果的 (索引, 相似度分数) 元组列表。
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    def save(self, cache_file: str, embeddings_file: str) -> None:
        """将检索器状态保存到磁盘。

        参数:
            cache_file: 保存语料库和 document_ids 的 pickle 文件路径。
            embeddings_file: 保存 numpy 嵌入数组的文件路径。
        """
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)

        state = {
            "corpus": self.corpus,
            "document_ids": self.document_ids,
            "model_name": self.model_name,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, cache_file: str, embeddings_file: str) -> "SimpleEmbeddingRetriever":
        """从磁盘加载检索器状态。

        参数:
            cache_file: 语料库 pickle 文件路径。
            embeddings_file: numpy 嵌入文件路径。

        返回:
            恢复的 SimpleEmbeddingRetriever 实例。
        """
        with open(cache_file, "rb") as f:
            state = pickle.load(f)

        retriever = cls(model_name=state.get("model_name", "all-MiniLM-L6-v2"))
        retriever.corpus = state["corpus"]
        retriever.document_ids = state["document_ids"]

        if os.path.exists(embeddings_file):
            retriever.embeddings = np.load(embeddings_file)

        return retriever

    @classmethod
    def from_memories(
        cls, memories: Dict, model_name: str = "all-MiniLM-L6-v2"
    ) -> "SimpleEmbeddingRetriever":
        """从现有记忆笔记构建检索器。

        为每个记忆创建包含内容、上下文、关键词和标签的文档字符串，
        然后对其进行索引。

        参数:
            memories: 记忆 ID 到 MemoryNote 对象的字典映射。
            model_name: SentenceTransformer 模型名称。

        返回:
            填充了记忆数据的新 SimpleEmbeddingRetriever 实例。
        """
        retriever = cls(model_name)
        docs = []
        for m in memories.values():
            metadata = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata}"
            docs.append(doc)
        if docs:
            retriever.add_documents(docs)
        return retriever

    def __len__(self) -> int:
        return len(self.corpus)
