"""Batch entity extraction with parallel processing optimizations.

Provides parallel batch processing for entity extraction to improve performance.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
import logging
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class BatchEntityExtractor:
    """Extract entities from multiple chunks using parallel batch processing."""
    
    def __init__(
        self,
        llm_client,
        entity_types: List[str] = None,
        batch_size: int = 5,
        cache: Optional[Any] = None,
        max_workers: int = 5
    ):
        """Initialize batch entity extractor.
        
        Args:
            llm_client: LLM client for entity extraction
            entity_types: List of entity types to extract
            batch_size: Number of chunks to process per batch
            cache: Optional embedding cache for entity deduplication
            max_workers: Maximum number of parallel workers for concurrent API calls
        """
        self.llm_client = llm_client
        self.entity_types = entity_types or [
            "PERSON", "ORGANIZATION", "LOCATION", 
            "EVENT", "CONCEPT", "DATE"
        ]
        self.batch_size = batch_size
        self.cache = cache
        self.max_workers = max_workers
        self.extracted_entities_cache = {}  # In-memory cache for this run
        self.cache_lock = threading.Lock()  # Thread-safe cache access
    
    def extract_from_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        use_cache: bool = True,
        skip_duplicates: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from multiple chunks using parallel batch processing.
        
        Args:
            chunks: List of chunk dictionaries
            use_cache: Whether to use in-memory cache for duplicate texts
            skip_duplicates: Whether to skip extracting from duplicate chunks
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        logger.info(f"Starting parallel entity extraction from {len(chunks)} chunks (batch_size={self.batch_size}, workers={self.max_workers})")
        
        all_entities = []
        all_relationships = []
        processed_texts = set() if use_cache else None
        
        # Filter out duplicate chunks upfront if needed
        chunks_to_process = []
        for chunk in chunks:
            chunk_text = chunk.get("text_with_buffer", chunk.get("text", ""))
            if use_cache and skip_duplicates:
                text_hash = hash(chunk_text)
                if text_hash in processed_texts:
                    continue
                processed_texts.add(text_hash)
            chunks_to_process.append(chunk)
        
        logger.info(f"Processing {len(chunks_to_process)} unique chunks (skipped {len(chunks) - len(chunks_to_process)} duplicates)")
        
        # Process chunks in parallel batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for parallel processing
            future_to_chunk = {
                executor.submit(self._extract_from_single_chunk, 
                              chunk.get("text_with_buffer", chunk.get("text", "")),
                              chunk["chunk_id"]): chunk
                for chunk in chunks_to_process
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(
                as_completed(future_to_chunk),
                total=len(future_to_chunk),
                desc="Extracting entities (parallel)",
                unit="chunk",
                file=sys.stdout,
                disable=False
            ):
                try:
                    result = future.result()
                    all_entities.extend(result["entities"])
                    all_relationships.extend(result["relationships"])
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Error extracting from chunk {chunk['chunk_id']}: {str(e)}")
        
        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)
        
        logger.info(f"Parallel extraction complete: {len(unique_entities)} unique entities, {len(all_relationships)} relationships")
        
        return unique_entities, all_relationships
    
    def _extract_from_single_chunk(
        self,
        chunk_text: str,
        chunk_id: int
    ) -> Dict[str, Any]:
        """Extract entities from a single chunk.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_id: Identifier for the chunk
            
        Returns:
            Dictionary with entities and relationships
        """
        from ..llm.prompt_templates import PromptTemplates
        
        # Check in-memory cache first (thread-safe)
        text_hash = hash(chunk_text)
        with self.cache_lock:
            if text_hash in self.extracted_entities_cache:
                cached_result = self.extracted_entities_cache[text_hash].copy()
                cached_result["chunk_id"] = chunk_id
                return cached_result
        
        prompt = PromptTemplates.format_entity_extraction(chunk_text)
        
        try:
            response = self.llm_client.generate_json(prompt, temperature=0.3)
            
            entities = response.get("entities", [])
            relationships = response.get("relationships", [])
            
            # Add chunk reference to each entity
            for entity in entities:
                entity["source_chunk"] = chunk_id
            
            # Add chunk reference to each relationship
            for rel in relationships:
                rel["source_chunk"] = chunk_id
            
            result = {
                "chunk_id": chunk_id,
                "entities": entities,
                "relationships": relationships
            }
            
            # Cache the result (thread-safe)
            with self.cache_lock:
                self.extracted_entities_cache[text_hash] = result
            
            logger.debug(f"Extracted {len(entities)} entities from chunk {chunk_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting entities from chunk {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "entities": [],
                "relationships": []
            }
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities, keeping track of all source chunks.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of deduplicated entities
        """
        entity_map = {}
        
        for entity in entities:
            name = entity["name"].lower().strip()
            entity_type = entity.get("type", "UNKNOWN")
            key = f"{name}_{entity_type}"
            
            if key not in entity_map:
                entity_map[key] = {
                    **entity,
                    "source_chunks": [entity["source_chunk"]],
                    "frequency": 1
                }
                # Remove the old source_chunk field
                entity_map[key].pop("source_chunk", None)
            else:
                # Update existing entity
                if entity["source_chunk"] not in entity_map[key]["source_chunks"]:
                    entity_map[key]["source_chunks"].append(entity["source_chunk"])
                entity_map[key]["frequency"] += 1
                # Keep more detailed description if available
                if len(entity.get("description", "")) > len(entity_map[key].get("description", "")):
                    entity_map[key]["description"] = entity.get("description", "")
        
        return list(entity_map.values())
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Get statistics about extracted entities.
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            "cached_extractions": len(self.extracted_entities_cache),
            "entity_cache_size": sum(
                len(r["entities"]) for r in self.extracted_entities_cache.values()
            )
        }
