"""Batch embedding processor for efficient API usage.

Provides batch processing of embeddings to reduce API calls and improve performance.
"""

import logging
from typing import List, Dict, Any, Callable, Tuple, Optional
import numpy as np
from tqdm import tqdm
import sys

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """Processes embeddings in batches for improved efficiency."""
    
    def __init__(self, embedding_function: Callable, batch_size: int = 10):
        """Initialize batch processor.
        
        Args:
            embedding_function: Function that takes list of texts and returns embeddings
            batch_size: Number of texts to process per batch
        """
        self.embedding_function = embedding_function
        self.batch_size = batch_size
    
    def batch_embed_texts(
        self,
        texts: List[str],
        cache: Optional[Any] = None,
        desc: str = "Processing embeddings"
    ) -> List[np.ndarray]:
        """Batch process texts for embeddings.
        
        Args:
            texts: List of texts to embed
            cache: Optional EmbeddingCache instance
            desc: Description for progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        cached_count = 0
        
        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc=desc,
            unit="batch",
            file=sys.stdout,
            disable=False,
            total=(len(texts) + self.batch_size - 1) // self.batch_size
        ):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                if cache is not None:
                    cached_emb = cache.get_sentence_embedding(text)
                    if cached_emb is not None:
                        batch_embeddings.append(cached_emb)
                        cached_count += 1
                    else:
                        texts_to_embed.append(text)
                        text_indices.append(j)
                else:
                    texts_to_embed.append(text)
                    text_indices.append(j)
            
            # Get embeddings for uncached texts (if any)
            if texts_to_embed:
                try:
                    new_embeddings = self.embedding_function(texts_to_embed)
                    
                    # Cache new embeddings
                    if cache is not None:
                        for text, emb in zip(texts_to_embed, new_embeddings):
                            cache.set_sentence_embedding(text, emb)
                    
                    # Insert new embeddings back in correct positions
                    result_idx = 0
                    for j in range(len(batch_texts)):
                        if j in text_indices:
                            batch_embeddings.insert(j, new_embeddings[result_idx])
                            result_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error getting embeddings for batch: {e}")
                    # Return zero embeddings as fallback
                    batch_embeddings = [np.zeros(1536) for _ in batch_texts]
            
            embeddings.extend(batch_embeddings)
        
        if cache is not None:
            logger.info(f"Batch embedding: {cached_count} from cache, {len(embeddings) - cached_count} newly computed")
        
        return embeddings
    
    def batch_embed_dict_texts(
        self,
        text_dict: Dict[str, str],
        cache: Optional[Any] = None,
        desc: str = "Processing embeddings"
    ) -> Dict[str, np.ndarray]:
        """Batch process dictionary of texts for embeddings.
        
        Args:
            text_dict: Dictionary mapping keys to texts
            cache: Optional EmbeddingCache instance
            desc: Description for progress bar
            
        Returns:
            Dictionary mapping keys to embedding vectors
        """
        keys = list(text_dict.keys())
        texts = [text_dict[k] for k in keys]
        embeddings = self.batch_embed_texts(texts, cache, desc)
        return {k: emb for k, emb in zip(keys, embeddings)}


class BatchEntityEmbeddingProcessor:
    """Specialized batch processor for entity embeddings."""
    
    def __init__(self, embedding_function: Callable, batch_size: int = 10):
        """Initialize entity batch processor.
        
        Args:
            embedding_function: Function that takes list of texts and returns embeddings
            batch_size: Number of entities to process per batch
        """
        self.embedding_function = embedding_function
        self.batch_size = batch_size
    
    def batch_embed_entities(
        self,
        entities: List[Dict[str, str]],
        cache: Optional[Any] = None,
        desc: str = "Embedding entities"
    ) -> Dict[str, np.ndarray]:
        """Batch process entities for embeddings.
        
        Args:
            entities: List of entity dicts with 'name' and 'type' keys
            cache: Optional EmbeddingCache instance
            desc: Description for progress bar
            
        Returns:
            Dictionary mapping entity names to embeddings
        """
        entity_embeddings = {}
        cached_count = 0
        
        # Extract unique entity names
        unique_names = list(set(e.get("name", "") for e in entities if e.get("name")))
        
        # Process in batches
        for i in tqdm(
            range(0, len(unique_names), self.batch_size),
            desc=desc,
            unit="batch",
            file=sys.stdout,
            disable=False,
            total=(len(unique_names) + self.batch_size - 1) // self.batch_size
        ):
            batch_names = unique_names[i:i + self.batch_size]
            batch_embeddings = {}
            names_to_embed = []
            
            # Check cache for each entity name
            for name in batch_names:
                if cache is not None:
                    cached_emb = cache.get_entity_embedding(name)
                    if cached_emb is not None:
                        batch_embeddings[name] = cached_emb
                        cached_count += 1
                    else:
                        names_to_embed.append(name)
                else:
                    names_to_embed.append(name)
            
            # Get embeddings for uncached entities
            if names_to_embed:
                try:
                    new_embeddings = self.embedding_function(names_to_embed)
                    
                    # Cache and store new embeddings (ensure flattened)
                    for name, emb in zip(names_to_embed, new_embeddings):
                        emb_flat = np.array(emb).flatten()
                        batch_embeddings[name] = emb_flat
                        if cache is not None:
                            cache.set_entity_embedding(name, emb_flat)
                    
                except Exception as e:
                    logger.error(f"Error getting embeddings for entity batch: {e}")
                    # Return zero embeddings as fallback
                    for name in names_to_embed:
                        batch_embeddings[name] = np.zeros(1536)
            
            entity_embeddings.update(batch_embeddings)
        
        if cache is not None:
            logger.info(f"Entity embedding: {cached_count} from cache, {len(entity_embeddings) - cached_count} newly computed")
        
        return entity_embeddings
