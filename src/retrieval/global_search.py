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
        batch_size: int = 10
    ):
        """Initialize global search.
        
        Args:
            embedding_function: Function to get embeddings
            top_k_communities: Number of communities to retrieve
            cache: Optional EmbeddingCache instance
            batch_size: Batch size for batch processing
        """
        self.embedding_function = embedding_function
        self.top_k_communities = top_k_communities
        self.cache = cache if cache is not None else EmbeddingCache()
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
        
        # Batch process embeddings
        embeddings_dict = self.batch_processor.batch_embed_dict_texts(
            {str(cid): summary for cid, summary in community_summaries.items()},
            cache=self.cache,
            desc="Embedding communities"
        )
        
        # Convert string keys back to integers
        self.community_embeddings = {int(k): v for k, v in embeddings_dict.items()}
        # Persist per-community embeddings in cache as well
        try:
            for k, v in self.community_embeddings.items():
                self.cache.set_community_embedding(int(k), v)
        except Exception:
            # Non-fatal: sentence-level cache still persists embeddings
            pass
    
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
        
        for comm_id in community_summaries.keys():
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
        
        # Rank communities
        ranked_communities = self.rank_communities(query, community_summaries)
        
        # Get top-k communities
        top_community_ids = [comm_id for comm_id, score in ranked_communities[:self.top_k_communities]]
        
        # Retrieve community summaries
        retrieved_summaries = [
            community_summaries[comm_id] 
            for comm_id in top_community_ids 
            if comm_id in community_summaries
        ]
        
        return {
            "community_summaries": retrieved_summaries,
            "community_ids": top_community_ids,
            "num_communities": len(community_summaries),
            "community_scores": ranked_communities[:self.top_k_communities]
        }
