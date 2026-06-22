"""
SimpleEmbeddingRetriever: Cosine similarity-based memory retrieval.

Uses SentenceTransformer (all-MiniLM-L6-v2) to encode text and compute
cosine similarity for retrieving relevant memories (Equations 8-10).

Paper reference:
  e_q = f_enc(q)                          (Equation 8)
  s_{q,i} = (e_q · e_i) / (|e_q| |e_i|)  (Equation 9)
  M_retrieved = {m_i | rank(s_{q,i}) <= k} (Equation 10)
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbeddingRetriever:
    """Embedding-based retrieval system using cosine similarity.

    This retriever encodes documents using a SentenceTransformer model
    and retrieves the top-k most relevant documents based on cosine
    similarity between query and document embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever with a SentenceTransformer model.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                        Default is 'all-MiniLM-L6-v2' as specified in the paper.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the retriever's index.

        Encodes the documents and appends them to the existing index.
        If no documents exist yet, initializes the index.

        Args:
            documents: List of text strings to add to the index.
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
        """Search for the top-k most similar documents.

        Args:
            query: The query text to search for.
            k: Number of top results to return.

        Returns:
            List of indices of the top-k most similar documents.
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Search with similarity scores.

        Args:
            query: The query text to search for.
            k: Number of top results to return.

        Returns:
            List of (index, similarity_score) tuples for the top-k results.
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    def save(self, cache_file: str, embeddings_file: str) -> None:
        """Save retriever state to disk.

        Args:
            cache_file: Path to save the corpus and document_ids pickle.
            embeddings_file: Path to save the numpy embeddings array.
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
        """Load retriever state from disk.

        Args:
            cache_file: Path to the corpus pickle file.
            embeddings_file: Path to the numpy embeddings file.

        Returns:
            A restored SimpleEmbeddingRetriever instance.
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
        """Build a retriever from existing memory notes.

        Creates document strings combining content, context, keywords,
        and tags for each memory, then indexes them.

        Args:
            memories: Dictionary mapping memory IDs to MemoryNote objects.
            model_name: SentenceTransformer model name.

        Returns:
            A new SimpleEmbeddingRetriever populated with the memories.
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
