"""Vector store using FAISS for efficient embedding storage and retrieval."""

import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for chunk, entity, and community embeddings."""
    
    def __init__(self, embedding_dim: int = 1536, cache_dir: str = "./data/cache"):
        """Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default: OpenAI 1536)
            cache_dir: Directory to store vector indices
        """
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate indices for each embedding type
        self.chunk_index = faiss.IndexFlatL2(embedding_dim)
        self.entity_index = faiss.IndexFlatL2(embedding_dim)
        self.community_index = faiss.IndexFlatL2(embedding_dim)
        
        # Metadata storage (maps index position to actual ID/text)
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.entity_metadata: List[Dict[str, Any]] = []
        self.community_metadata: List[Dict[str, Any]] = []
        
        # Load existing indices if available
        self._load_indices()
    
    def _load_indices(self):
        """Load existing vector indices from disk."""
        try:
            chunk_index_path = self.cache_dir / "chunk_index.faiss"
            chunk_meta_path = self.cache_dir / "chunk_metadata.json"
            
            if chunk_index_path.exists():
                self.chunk_index = faiss.read_index(str(chunk_index_path))
                with open(chunk_meta_path, 'r') as f:
                    self.chunk_metadata = json.load(f)
                logger.info(f"Loaded {len(self.chunk_metadata)} chunk embeddings from vector store")
            
            entity_index_path = self.cache_dir / "entity_index.faiss"
            entity_meta_path = self.cache_dir / "entity_metadata.json"
            
            if entity_index_path.exists():
                self.entity_index = faiss.read_index(str(entity_index_path))
                with open(entity_meta_path, 'r') as f:
                    self.entity_metadata = json.load(f)
                logger.info(f"Loaded {len(self.entity_metadata)} entity embeddings from vector store")
            
            community_index_path = self.cache_dir / "community_index.faiss"
            community_meta_path = self.cache_dir / "community_metadata.json"
            
            if community_index_path.exists():
                self.community_index = faiss.read_index(str(community_index_path))
                with open(community_meta_path, 'r') as f:
                    self.community_metadata = json.load(f)
                logger.info(f"Loaded {len(self.community_metadata)} community embeddings from vector store")
        
        except Exception as e:
            logger.warning(f"Error loading vector indices: {e}. Starting with empty indices.")
    
    def add_chunks(self, embeddings_dict: Dict[int, np.ndarray]):
        """Add chunk embeddings to the vector store.
        
        Args:
            embeddings_dict: Dictionary mapping chunk_id to embedding vector
        """
        if not embeddings_dict:
            return
        
        logger.info(f"Adding {len(embeddings_dict)} chunk embeddings to vector store")
        
        # Check if we already have embeddings - if so, clear and rebuild
        if len(self.chunk_metadata) > 0:
            logger.info(f"Clearing existing {len(self.chunk_metadata)} chunks and rebuilding index")
            self.chunk_index.reset()
            self.chunk_metadata = []
        
        embeddings_array = []
        for chunk_id, emb in embeddings_dict.items():
            emb = np.array(emb).flatten().astype(np.float32)
            if emb.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb.shape[0]}")
            embeddings_array.append(emb)
            
            self.chunk_metadata.append({
                "chunk_id": int(chunk_id),
                "index_pos": len(self.chunk_metadata)
            })
        
        embeddings_array = np.array(embeddings_array)
        self.chunk_index.add(embeddings_array)
    
    def add_entities(self, embeddings_dict: Dict[str, np.ndarray]):
        """Add entity embeddings to the vector store.
        
        Args:
            embeddings_dict: Dictionary mapping entity_name to embedding vector
        """
        if not embeddings_dict:
            return
        
        logger.info(f"Adding {len(embeddings_dict)} entity embeddings to vector store")
        
        # Check if we already have embeddings - if so, clear and rebuild
        if len(self.entity_metadata) > 0:
            logger.info(f"Clearing existing {len(self.entity_metadata)} entities and rebuilding index")
            self.entity_index.reset()
            self.entity_metadata = []
        
        embeddings_array = []
        for entity_name, emb in embeddings_dict.items():
            emb = np.array(emb).flatten().astype(np.float32)
            if emb.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb.shape[0]}")
            embeddings_array.append(emb)
            
            self.entity_metadata.append({
                "entity_name": str(entity_name),
                "index_pos": len(self.entity_metadata)
            })
        
        embeddings_array = np.array(embeddings_array)
        self.entity_index.add(embeddings_array)
    
    def add_communities(self, embeddings_dict: Dict[int, np.ndarray]):
        """Add community embeddings to the vector store.
        
        Args:
            embeddings_dict: Dictionary mapping community_id to embedding vector
        """
        if not embeddings_dict:
            return
        
        logger.info(f"Adding {len(embeddings_dict)} community embeddings to vector store")
        
        # Check if we already have embeddings - if so, clear and rebuild
        if len(self.community_metadata) > 0:
            logger.info(f"Clearing existing {len(self.community_metadata)} communities and rebuilding index")
            self.community_index.reset()
            self.community_metadata = []
        
        embeddings_array = []
        for comm_id, emb in embeddings_dict.items():
            emb = np.array(emb).flatten().astype(np.float32)
            if emb.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb.shape[0]}")
            embeddings_array.append(emb)
            
            self.community_metadata.append({
                "community_id": int(comm_id),
                "index_pos": len(self.community_metadata)
            })
        
        embeddings_array = np.array(embeddings_array)
        self.community_index.add(embeddings_array)
    
    def search_chunks(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk_id, distance) tuples
        """
        query_emb = np.array(query_embedding).flatten().astype(np.float32).reshape(1, -1)
        distances, indices = self.chunk_index.search(query_emb, min(k, len(self.chunk_metadata)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.chunk_metadata):
                chunk_id = self.chunk_metadata[idx]["chunk_id"]
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + distance)
                results.append((chunk_id, similarity))
        
        return results
    
    def search_entities(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entities.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (entity_name, similarity_score) tuples
        """
        if len(self.entity_metadata) == 0:
            return []
        
        query_emb = np.array(query_embedding).flatten().astype(np.float32).reshape(1, -1)
        distances, indices = self.entity_index.search(query_emb, min(k, len(self.entity_metadata)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.entity_metadata):
                entity_name = self.entity_metadata[idx]["entity_name"]
                similarity = 1 / (1 + distance)
                results.append((entity_name, similarity))
        
        return results
    
    def search_communities(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar communities.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (community_id, similarity_score) tuples
        """
        if len(self.community_metadata) == 0:
            return []
        
        query_emb = np.array(query_embedding).flatten().astype(np.float32).reshape(1, -1)
        distances, indices = self.community_index.search(query_emb, min(k, len(self.community_metadata)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.community_metadata):
                comm_id = self.community_metadata[idx]["community_id"]
                similarity = 1 / (1 + distance)
                results.append((comm_id, similarity))
        
        return results
    
    def save(self):
        """Save vector indices and metadata to disk."""
        try:
            faiss.write_index(self.chunk_index, str(self.cache_dir / "chunk_index.faiss"))
            with open(self.cache_dir / "chunk_metadata.json", 'w') as f:
                json.dump(self.chunk_metadata, f)
            
            faiss.write_index(self.entity_index, str(self.cache_dir / "entity_index.faiss"))
            with open(self.cache_dir / "entity_metadata.json", 'w') as f:
                json.dump(self.entity_metadata, f)
            
            faiss.write_index(self.community_index, str(self.cache_dir / "community_index.faiss"))
            with open(self.cache_dir / "community_metadata.json", 'w') as f:
                json.dump(self.community_metadata, f)
            
            logger.info("Saved vector store indices to disk")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics."""
        # Provide both legacy keys and the num_* keys used by example.py output
        chunk_count = len(self.chunk_metadata)
        entity_count = len(self.entity_metadata)
        community_count = len(self.community_metadata)
        return {
            "chunk_embeddings": chunk_count,
            "entity_embeddings": entity_count,
            "community_embeddings": community_count,
            "total_embeddings": chunk_count + entity_count + community_count,
            "num_chunks": chunk_count,
            "num_entities": entity_count,
            "num_communities": community_count,
        }

    # ------------------------------------------------------------------
    # Reconstruct embeddings from FAISS (to avoid re-embedding)
    # ------------------------------------------------------------------
    def _reconstruct_embeddings(self, index: faiss.Index, metadata: List[Dict[str, Any]], key_field: str) -> Dict[Any, np.ndarray]:
        """Rebuild an embedding dict from a FAISS index and metadata."""
        embeddings: Dict[Any, np.ndarray] = {}
        for pos, meta in enumerate(metadata):
            try:
                emb = index.reconstruct(pos)
                embeddings[meta[key_field]] = np.array(emb, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to reconstruct embedding at position {pos}: {e}")
        return embeddings

    def get_chunk_embeddings_dict(self) -> Dict[int, np.ndarray]:
        """Return all chunk embeddings keyed by chunk_id from the FAISS index."""
        return self._reconstruct_embeddings(self.chunk_index, self.chunk_metadata, "chunk_id")

    def get_entity_embeddings_dict(self) -> Dict[str, np.ndarray]:
        """Return all entity embeddings keyed by entity_name from the FAISS index."""
        return self._reconstruct_embeddings(self.entity_index, self.entity_metadata, "entity_name")

    def get_community_embeddings_dict(self) -> Dict[int, np.ndarray]:
        """Return all community embeddings keyed by community_id from the FAISS index."""
        return self._reconstruct_embeddings(self.community_index, self.community_metadata, "community_id")
