"""
Local search module for entity-based retrieval.

Retrieves relevant chunks based on entity matching and semantic similarity.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from ..cache.embedding_cache import EmbeddingCache
from ..cache.batch_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class LocalSearch:
    """Local search retriever using entities and semantic similarity."""

    def __init__(
        self,
        graph,
        embedding_function,
        top_k_entities: int = 10,
        top_k_chunks: int = 5,
        similarity_weight: float = 0.6,
        graph_weight: float = 0.4,
        show_progress: bool = True,
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 10
    ):
        self.graph = graph
        self.embedding_function = embedding_function
        self.top_k_entities = top_k_entities
        self.top_k_chunks = top_k_chunks
        self.similarity_weight = similarity_weight
        self.graph_weight = graph_weight
        self.show_progress = show_progress
        self.cache = cache if cache is not None else EmbeddingCache()

        self.batch_processor = BatchEmbeddingProcessor(
            embedding_function,
            batch_size=batch_size
        )

        # In-memory embedding stores
        self.chunk_embeddings: Dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------------
    # Embedding computation
    # ---------------------------------------------------------------------

    def compute_chunk_embeddings(self, chunks: List[Dict[str, Any]]):
        """Precompute embeddings for all chunks using batch processing."""
        logger.info(f"Computing embeddings for {len(chunks)} chunks")

        chunk_texts = {
            chunk["chunk_id"]: chunk.get("text", "")
            for chunk in chunks
        }

        embeddings = self.batch_processor.batch_embed_dict_texts(
            chunk_texts,
            cache=self.cache,
            desc="Embedding chunks"
        )

        for cid, emb in embeddings.items():
            emb = np.array(emb).flatten()
            self.chunk_embeddings[cid] = emb

            # Persist safely
            try:
                self.cache.set_chunk_embedding(str(cid), emb)
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Entity handling (SemRAG-style: graph-based, NOT vector-based)
    # ---------------------------------------------------------------------

    def find_relevant_entities(self, query: str) -> List[Tuple[str, float]]:
        """
        Find relevant entities based on textual match + graph frequency.
        NO cosine similarity on entities (SemRAG-correct).
        """
        query_lower = query.lower()
        entity_scores = []

        for node in self.graph.nodes():
            if not node.startswith("entity_"):
                continue

            data = self.graph.nodes[node]
            name = data.get("entity_name", "")
            if not name:
                continue

            score = 0.0

            # Simple lexical match
            if name.lower() in query_lower:
                score += 1.0

            # Graph frequency boost
            frequency = data.get("frequency", 1)
            score += np.log1p(frequency) * 0.1

            if score > 0:
                entity_scores.append((name, score))

        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:self.top_k_entities]

    def get_entity_chunks(self, entity_names: List[str]) -> List[int]:
        """Get chunk IDs connected to the given entities."""
        chunk_ids = set()

        for entity_name in entity_names:
            for node in self.graph.nodes():
                if (
                    node.startswith("entity_")
                    and entity_name.lower() in node.lower()
                ):
                    for neighbor in self.graph.neighbors(node):
                        if neighbor.startswith("chunk_"):
                            chunk_ids.add(int(neighbor.split("_")[1]))

        return list(chunk_ids)

    # ---------------------------------------------------------------------
    # Chunk ranking
    # ---------------------------------------------------------------------

    def rank_chunks(
        self,
        query: str,
        chunk_ids: List[int],
        chunks: List[Dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """Rank chunks using semantic similarity + graph signal."""

        # ✅ CORRECT query embedding usage
        query_embedding = self.embedding_function([query])[0]
        query_embedding = np.array(query_embedding).reshape(1, -1)

        ranked = []

        for chunk_id in chunk_ids:
            if chunk_id not in self.chunk_embeddings:
                continue

            chunk_embedding = self.chunk_embeddings[chunk_id].reshape(1, -1)

            # 🔒 Defensive dimension check
            if query_embedding.shape[1] != chunk_embedding.shape[1]:
                logger.warning(
                    f"Skipping chunk {chunk_id}: "
                    f"query dim {query_embedding.shape[1]} != "
                    f"chunk dim {chunk_embedding.shape[1]}"
                )
                continue

            similarity = cosine_similarity(
                query_embedding,
                chunk_embedding
            )[0][0]

            # Graph-based score
            chunk_node = f"chunk_{chunk_id}"
            if self.graph.has_node(chunk_node):
                num_entities = sum(
                    1 for n in self.graph.neighbors(chunk_node)
                    if n.startswith("entity_")
                )

                graph_score = min(num_entities / 10.0, 1.0)
            else:
                graph_score = 0.0

            final_score = (
                self.similarity_weight * similarity
                + self.graph_weight * graph_score
            )

            ranked.append((chunk_id, final_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ---------------------------------------------------------------------
    # Search API
    # ---------------------------------------------------------------------

    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform local SemRAG-style search."""
        logger.info(f"Performing local search for query: {query}")

        entities = self.find_relevant_entities(query)
        entity_names = [e[0] for e in entities]

        candidate_chunk_ids = self.get_entity_chunks(entity_names)

        ranked_chunks = self.rank_chunks(
            query,
            candidate_chunk_ids,
            chunks
        )

        top_chunk_ids = [
            cid for cid, _ in ranked_chunks[:self.top_k_chunks]
        ]

        chunk_map = {c["chunk_id"]: c for c in chunks}
        retrieved_chunks = [
            chunk_map[cid]["text"]
            for cid in top_chunk_ids
            if cid in chunk_map
        ]

        return {
            "chunks": retrieved_chunks,
            "chunk_ids": top_chunk_ids,
            "entities": entity_names[:5],
            "num_candidates": len(candidate_chunk_ids),
            "entity_scores": entities[:5],
        }
