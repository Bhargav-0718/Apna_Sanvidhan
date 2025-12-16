"""Embedding cache module for storing and retrieving cached embeddings.

Uses pickle files to cache sentence, chunk, entity, and community embeddings to avoid
recomputing expensive LLM embeddings on subsequent runs.
"""

import pickle
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for storing and retrieving embeddings using pickle files."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.sentence_cache_file = self.cache_dir / "sentence_embeddings.pkl"
        self.chunk_cache_file = self.cache_dir / "chunk_embeddings.pkl"
        self.entity_cache_file = self.cache_dir / "entity_embeddings.pkl"
        self.community_cache_file = self.cache_dir / "community_embeddings.pkl"
        
        # In-memory caches
        self.sentence_embeddings: Dict[str, np.ndarray] = {}
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.community_embeddings: Dict[int, np.ndarray] = {}
        
        # Load existing caches
        self._load_caches()
    
    def _load_caches(self):
        """Load existing cache files from disk."""
        if self.sentence_cache_file.exists():
            try:
                with open(self.sentence_cache_file, 'rb') as f:
                    self.sentence_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.sentence_embeddings)} cached sentence embeddings")
            except Exception as e:
                logger.warning(f"Error loading sentence cache: {e}")
                self.sentence_embeddings = {}
        
        if self.chunk_cache_file.exists():
            try:
                with open(self.chunk_cache_file, 'rb') as f:
                    self.chunk_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.chunk_embeddings)} cached chunk embeddings")
            except Exception as e:
                logger.warning(f"Error loading chunk cache: {e}")
                self.chunk_embeddings = {}
        
        if self.entity_cache_file.exists():
            try:
                with open(self.entity_cache_file, 'rb') as f:
                    self.entity_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.entity_embeddings)} cached entity embeddings")
            except Exception as e:
                logger.warning(f"Error loading entity cache: {e}")
                self.entity_embeddings = {}
        
        if self.community_cache_file.exists():
            try:
                with open(self.community_cache_file, 'rb') as f:
                    self.community_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.community_embeddings)} cached community embeddings")
            except Exception as e:
                logger.warning(f"Error loading community cache: {e}")
                self.community_embeddings = {}
    
    def _text_to_hash(self, text: str) -> str:
        """Convert text to a hash key for caching.
        
        Args:
            text: Text to hash
            
        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_sentence_embedding(self, sentence: str) -> Optional[np.ndarray]:
        """Get cached sentence embedding.
        
        Args:
            sentence: Sentence text
            
        Returns:
            Cached embedding if available, None otherwise
        """
        key = self._text_to_hash(sentence)
        return self.sentence_embeddings.get(key)
    
    def set_sentence_embedding(self, sentence: str, embedding: np.ndarray):
        """Cache a sentence embedding.
        
        Args:
            sentence: Sentence text
            embedding: Embedding vector
        """
        key = self._text_to_hash(sentence)
        self.sentence_embeddings[key] = embedding
    
    def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get cached chunk embedding.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Cached embedding if available, None otherwise
        """
        return self.chunk_embeddings.get(chunk_id)
    
    def set_chunk_embedding(self, chunk_id: str, embedding: np.ndarray):
        """Cache a chunk embedding.
        
        Args:
            chunk_id: Chunk identifier
            embedding: Embedding vector
        """
        self.chunk_embeddings[chunk_id] = embedding
    
    def get_entity_embedding(self, entity_name: str) -> Optional[np.ndarray]:
        """Get cached entity embedding.
        
        Args:
            entity_name: Entity name/text
            
        Returns:
            Cached embedding if available, None otherwise
        """
        key = self._text_to_hash(entity_name)
        return self.entity_embeddings.get(key)
    
    def set_entity_embedding(self, entity_name: str, embedding: np.ndarray):
        """Cache an entity embedding.
        
        Args:
            entity_name: Entity name/text
            embedding: Embedding vector
        """
        key = self._text_to_hash(entity_name)
        self.entity_embeddings[key] = embedding
    
    def get_community_embedding(self, community_id: int) -> Optional[np.ndarray]:
        """Get cached community embedding.
        
        Args:
            community_id: Community identifier
            
        Returns:
            Cached embedding if available, None otherwise
        """
        return self.community_embeddings.get(community_id)
    
    def set_community_embedding(self, community_id: int, embedding: np.ndarray):
        """Cache a community embedding.
        
        Args:
            community_id: Community identifier
            embedding: Embedding vector
        """
        self.community_embeddings[community_id] = embedding
    
    def save_caches(self):
        """Save all caches to disk."""
        try:
            with open(self.sentence_cache_file, 'wb') as f:
                pickle.dump(self.sentence_embeddings, f)
            logger.info(f"Saved {len(self.sentence_embeddings)} sentence embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving sentence cache: {e}")
        
        try:
            with open(self.chunk_cache_file, 'wb') as f:
                pickle.dump(self.chunk_embeddings, f)
            logger.info(f"Saved {len(self.chunk_embeddings)} chunk embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving chunk cache: {e}")
        
        try:
            with open(self.entity_cache_file, 'wb') as f:
                pickle.dump(self.entity_embeddings, f)
            logger.info(f"Saved {len(self.entity_embeddings)} entity embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving entity cache: {e}")
        
        try:
            with open(self.community_cache_file, 'wb') as f:
                pickle.dump(self.community_embeddings, f)
            logger.info(f"Saved {len(self.community_embeddings)} community embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving community cache: {e}")
    
    def clear_caches(self):
        """Clear all in-memory caches and delete cache files."""
        self.sentence_embeddings.clear()
        self.chunk_embeddings.clear()
        self.entity_embeddings.clear()
        self.community_embeddings.clear()
        
        for cache_file in [self.sentence_cache_file, self.chunk_cache_file, self.entity_cache_file, self.community_cache_file]:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    logger.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached embeddings.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "sentence_embeddings": len(self.sentence_embeddings),
            "chunk_embeddings": len(self.chunk_embeddings),
            "entity_embeddings": len(self.entity_embeddings),
            "community_embeddings": len(self.community_embeddings),
            "total_embeddings": len(self.sentence_embeddings) + len(self.chunk_embeddings) + len(self.entity_embeddings) + len(self.community_embeddings)
        }
