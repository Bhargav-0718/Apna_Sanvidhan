"""Global search module for community-based retrieval.

Retrieves relevant community summaries for high-level queries.
"""

import numpy as np
import sys
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from ..cache.embedding_cache import EmbeddingCache
from ..cache.batch_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class GlobalSearch:
    """Global search retriever using community summaries."""
    
    def __init__(
        self,
        embedding_function,
        top_k_communities: int = 5,
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 10,
        vector_store = None
    ):
        """Initialize global search.
        
        Args:
            embedding_function: Function to get embeddings
            top_k_communities: Number of communities to retrieve
            cache: Optional EmbeddingCache instance
            batch_size: Batch size for batch processing
            vector_store: Optional VectorStore instance for fast retrieval
        """
        self.embedding_function = embedding_function
        self.top_k_communities = top_k_communities
        self.cache = cache if cache is not None else EmbeddingCache()
        self.vector_store = vector_store
        self.batch_processor = BatchEmbeddingProcessor(embedding_function, batch_size=batch_size)
        
        # Cache for community summary embeddings
        self.community_embeddings = {}
    
    def compute_community_embeddings(
        self, 
        community_summaries: Dict[int, str]
    ):
        """Precompute embeddings for community summaries using batch processing.
        
        Args:
            community_summaries: Dictionary mapping community_id to summary
        """
        logger.info(f"Computing embeddings for {len(community_summaries)} communities using batch processing")
        
        # Fast-path: load from vector store if already present
        if self.vector_store is not None:
            try:
                stats = self.vector_store.get_stats()
                expected_ids = set(community_summaries.keys())
                if stats.get("num_communities", 0) >= len(expected_ids) and expected_ids:
                    restored = self.vector_store.get_community_embeddings_dict()
                    missing = expected_ids - set(restored.keys())
                    if not missing:
                        self.community_embeddings = {int(cid): np.array(emb).flatten() for cid, emb in restored.items()}
                        logger.info(f"Loaded {len(self.community_embeddings)} community embeddings from vector store; skipping re-embedding")
                        # Add to vector store not needed; already there
                        return
                    else:
                        logger.info(f"Vector store missing {len(missing)} community embeddings; recomputing")
            except Exception as e:
                logger.warning(f"Falling back to re-embedding communities due to: {e}")

        # Batch process embeddings
        embeddings_dict = self.batch_processor.batch_embed_dict_texts(
            {str(cid): summary for cid, summary in community_summaries.items()},
            cache=self.cache,
            desc="Embedding communities"
        )
        
        # Convert string keys back to integers
        self.community_embeddings = {int(k): v for k, v in embeddings_dict.items()}
        
        # Add embeddings to vector store for fast retrieval
        if self.vector_store is not None:
            logger.info("Adding community embeddings to vector store")
            try:
                self.vector_store.add_communities(self.community_embeddings)
                logger.info(f"Successfully added {len(self.community_embeddings)} communities to vector store")
            except Exception as e:
                logger.error(f"Failed to add communities to vector store: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    def rank_communities(
        self, 
        query: str, 
        community_summaries: Dict[int, str]
    ) -> List[Tuple[int, float]]:
        """Rank communities by relevance to query.
        
        Args:
            query: User query
            community_summaries: Dictionary of community summaries
            
        Returns:
            List of (community_id, score) tuples sorted by score
        """
        query_embedding = self.embedding_function(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        community_scores = []

        # Normalize keys to integers (JSON loads keys as strings)
        summaries_int = {int(k): v for k, v in community_summaries.items()}

        for comm_id in summaries_int.keys():
            if comm_id in self.community_embeddings:
                comm_embedding = np.array(self.community_embeddings[comm_id]).reshape(1, -1)

                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, comm_embedding)[0][0]

                community_scores.append((comm_id, similarity))
        
        # Sort by score
        community_scores.sort(key=lambda x: x[1], reverse=True)
        
        return community_scores
    
    def search(
        self, 
        query: str, 
        community_summaries: Dict[int, str]
    ) -> Dict[str, Any]:
        """Perform global search to retrieve relevant community summaries.
        
        Args:
            query: User query
            community_summaries: Dictionary of community summaries
            
        Returns:
            Dictionary with retrieved community summaries
        """
        logger.info(f"Performing global search for query: {query}")
        
        # Normalize keys to integers for consistent matching
        summaries_int = {int(k): v for k, v in community_summaries.items()}

        # Rank communities
        ranked_communities = self.rank_communities(query, summaries_int)
        
        # Get top-k communities
        top_community_ids = [comm_id for comm_id, score in ranked_communities[:self.top_k_communities]]
        
        # Retrieve community summaries
        retrieved_summaries = [
            summaries_int[comm_id]
            for comm_id in top_community_ids
            if comm_id in summaries_int
        ]
        
        return {
            "community_summaries": retrieved_summaries,
            "community_ids": top_community_ids,
            "num_communities": len(summaries_int),
            "community_scores": ranked_communities[:self.top_k_communities]
        }
