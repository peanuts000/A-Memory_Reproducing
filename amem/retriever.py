"""
SimpleEmbeddingRetriever: Embedding-based memory retrieval system.

Uses SentenceTransformer (all-MiniLM-L6-v2) to encode text and compute
cosine similarity for retrieving relevant memories (Equations 8-10 in the paper).

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
from rank_bm25 import BM25Okapi

# Use Hugging Face mirror for Chinese users
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class SimpleEmbeddingRetriever:
    """Embedding-based retrieval system using cosine similarity.

    Encodes documents using a SentenceTransformer model and retrieves
    top-k most relevant documents based on cosine similarity between
    query and document embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the retriever.

        Args:
            model_name: SentenceTransformer model name.
                        Defaults to 'all-MiniLM-L6-v2' as specified in the paper.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the retriever index.

        Encodes documents and appends to existing index.
        Initializes the index if no documents exist yet.

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
        """Search for top-k most similar documents.

        Args:
            query: Query text.
            k: Number of results to return.

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
            query: Query text.
            k: Number of results to return.

        Returns:
            List of (index, similarity_score) tuples for top-k results.
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
            cache_file: Path to pickle file for corpus and document_ids.
            embeddings_file: Path to numpy file for embeddings.
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
            cache_file: Path to corpus pickle file.
            embeddings_file: Path to numpy embeddings file.

        Returns:
            Restored SimpleEmbeddingRetriever instance.
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
        """Build retriever from existing memory notes.

        Creates document strings combining content, context, keywords,
        and tags for each memory, then indexes them.

        Args:
            memories: Dictionary mapping memory IDs to MemoryNote objects.
            model_name: SentenceTransformer model name.

        Returns:
            New SimpleEmbeddingRetriever instance populated with memory data.
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


def simple_tokenize(text: str) -> List[str]:
    """Tokenization function for BM25. Supports unigram + bigram for better keyword matching."""
    tokens = text.lower().split()
    # Add bigrams for better phrase matching
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
    return tokens + bigrams


class HybridRetriever:
    """Hybrid retrieval system combining BM25 keyword matching and semantic embedding retrieval.

    Balances two retrieval methods via the alpha parameter:
      - alpha=0: Pure BM25
      - alpha=1: Pure semantic embedding
      - alpha=0.6: Semantic retrieval dominant (default)

    Args:
        model_name: SentenceTransformer model name.
        alpha: Semantic retrieval weight (0-1).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", alpha: float = 0.6):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.alpha = alpha
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to BM25 and semantic index.

        Args:
            documents: List of text strings to add.
        """
        if not documents:
            return

        if not self.corpus:
            self.corpus = list(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

        # Update BM25 index
        tokenized_corpus = [simple_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Update semantic embeddings
        new_embeddings = self.model.encode(documents, show_progress_bar=False)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, k: int = 5) -> List[int]:
        """Hybrid retrieval: BM25 + semantic embedding.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            List of indices of the top-k most similar documents.
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))

        # BM25 scores
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        # Normalize BM25 scores to [0, 1]
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max - bm25_min > 1e-6:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # Semantic similarity scores
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        # Hybrid scores
        hybrid_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores

        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Hybrid retrieval with scores.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            List of (index, hybrid_score) tuples for top-k results.
        """
        if not self.corpus or self.embeddings is None:
            return []

        k = min(k, len(self.corpus))

        # BM25 scores
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max - bm25_min > 1e-6:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # Semantic similarity scores
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        # Hybrid scores
        hybrid_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores

        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return [(int(idx), float(hybrid_scores[idx])) for idx in top_k_indices]

    def save(self, cache_file: str, embeddings_file: str) -> None:
        """Save retriever state to disk."""
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)

        state = {
            "corpus": self.corpus,
            "document_ids": self.document_ids,
            "model_name": self.model_name,
            "alpha": self.alpha,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, cache_file: str, embeddings_file: str) -> "HybridRetriever":
        """Load retriever state from disk."""
        with open(cache_file, "rb") as f:
            state = pickle.load(f)

        retriever = cls(
            model_name=state.get("model_name", "all-MiniLM-L6-v2"),
            alpha=state.get("alpha", 0.5),
        )
        retriever.corpus = state["corpus"]
        retriever.document_ids = state["document_ids"]

        if os.path.exists(embeddings_file):
            retriever.embeddings = np.load(embeddings_file)

        # Rebuild BM25 index
        if retriever.corpus:
            tokenized_corpus = [simple_tokenize(doc) for doc in retriever.corpus]
            retriever.bm25 = BM25Okapi(tokenized_corpus)

        return retriever

    @classmethod
    def from_memories(
        cls, memories: Dict, model_name: str = "all-MiniLM-L6-v2", alpha: float = 0.5
    ) -> "HybridRetriever":
        """Build hybrid retriever from existing memory notes."""
        retriever = cls(model_name, alpha)
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
