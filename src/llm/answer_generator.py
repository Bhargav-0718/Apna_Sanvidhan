"""Answer generation module for SemRAG system.

Handles local, global, and hybrid search answer generation.
"""

from typing import Dict, Any, List, Optional
import logging
from .llm_client import LLMClient
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate answers using different search strategies."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize answer generator.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        self.prompts = PromptTemplates()
    
    def generate_local_answer(
        self, 
        question: str, 
        context_chunks: List[str],
        chunk_ids: List[int] = None,
        chunk_objects: List[Dict[str, Any]] = None,
        chunk_scores: List[float] = None,
        entities: List[str] = None,
        num_candidates: int = None
    ) -> Dict[str, Any]:
        """Generate answer using local search (entity-based) context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            chunk_ids: IDs of chunks used
            chunk_objects: Full chunk objects with metadata
            chunk_scores: Relevance scores for chunks
            entities: Relevant entities
            
        Returns:
            Dictionary with answer and metadata including chunks used
        """
        logger.info(f"Generating local search answer for: {question}")
        
        prompt = self.prompts.format_local_search(
            question=question,
            context=context_chunks,
            entities=entities or []
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        # Build chunks used metadata
        chunks_used = []
        if chunk_objects:
            for idx, chunk_obj in enumerate(chunk_objects):
                # Normalize score to a float (may be tuple (id, score))
                raw_score = chunk_scores[idx] if chunk_scores and idx < len(chunk_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                chunk_info = {
                    "chunk_id": chunk_obj.get("chunk_id", idx),
                    "text": chunk_obj.get("text", ""),
                    "score": norm_score,
                    "metadata": chunk_obj.get("metadata", {})
                }
                chunks_used.append(chunk_info)
        elif chunk_ids:
            for idx, chunk_id in enumerate(chunk_ids):
                raw_score = chunk_scores[idx] if chunk_scores and idx < len(chunk_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                chunk_info = {
                    "chunk_id": chunk_id,
                    "text": context_chunks[idx] if idx < len(context_chunks) else "",
                    "score": norm_score
                }
                chunks_used.append(chunk_info)

        # Extract article/clause information for UI citation block
        sources_cited = []
        for idx, chunk_text in enumerate(context_chunks):
            source_info = self._extract_source_info(chunk_text)
            raw_score = chunk_scores[idx] if chunk_scores and idx < len(chunk_scores) else None
            if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                norm_score = raw_score[1]
            else:
                norm_score = raw_score
            sources_cited.append({
                "chunk_id": chunk_ids[idx] if chunk_ids and idx < len(chunk_ids) else idx,
                "text": chunk_text[:500],
                "article": source_info.get("article"),
                "clause": source_info.get("clause"),
                "full_text": chunk_text,
                "score": norm_score
            })
        
        return {
            "answer": answer,
            "search_type": "local",
            "num_chunks": len(context_chunks),
            "num_candidates": num_candidates if num_candidates is not None else len(context_chunks),
            "chunk_ids": chunk_ids or list(range(len(context_chunks))),
            "chunks_used": chunks_used,
            "entities": entities or [],
            "context": context_chunks,
            "sources_cited": sources_cited
        }
    
    def generate_global_answer(
        self, 
        question: str, 
        community_summaries: List[str],
        community_ids: List[int] = None,
        community_scores: List[float] = None
    ) -> Dict[str, Any]:
        """Generate answer using global search (community-based) context.
        
        Args:
            question: User question
            community_summaries: Community summaries
            community_ids: IDs of communities used
            community_scores: Relevance scores for communities
            
        Returns:
            Dictionary with answer and metadata including communities used
        """
        logger.info(f"Generating global search answer for: {question}")
        
        prompt = self.prompts.format_global_search(
            question=question,
            community_summaries=community_summaries
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        # Build communities used metadata
        communities_used = []
        if community_ids:
            for idx, comm_id in enumerate(community_ids):
                raw_score = community_scores[idx] if community_scores and idx < len(community_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                comm_info = {
                    "community_id": comm_id,
                    "summary": community_summaries[idx] if idx < len(community_summaries) else "",
                    "score": norm_score
                }
                communities_used.append(comm_info)
        
        return {
            "answer": answer,
            "search_type": "global",
            "num_communities": len(community_summaries),
            "community_ids": community_ids or list(range(len(community_summaries))),
            "communities_used": communities_used,
            "context": community_summaries
        }
    
    def generate_hybrid_answer(
        self,
        question: str,
        local_context: List[str],
        global_context: List[str],
        entities: List[str],
        local_chunk_ids: List[int] = None,
        local_chunk_objects: List[Dict[str, Any]] = None,
        local_chunk_scores: List[float] = None,
        global_community_ids: List[int] = None,
        global_community_scores: List[float] = None,
        entity_scores: List[float] = None
    ) -> Dict[str, Any]:
        """Generate answer using hybrid search context.
        
        Args:
            question: User question
            local_context: Local search context chunks
            global_context: Global search community summaries
            entities: Relevant entities
            local_chunk_ids: IDs of local chunks used
            local_chunk_objects: Full chunk objects with metadata
            local_chunk_scores: Relevance scores for local chunks
            global_community_ids: IDs of communities used
            global_community_scores: Relevance scores for communities
            entity_scores: Relevance scores for entities
            
        Returns:
            Dictionary with answer and comprehensive metadata
        """
        logger.info(f"Generating hybrid search answer for: {question}")
        
        prompt = self.prompts.format_hybrid_search(
            question=question,
            local_context=local_context,
            global_context=global_context,
            entities=entities
        )
        
        answer = self.llm_client.generate(prompt, temperature=0.7)
        
        # Build local chunks used metadata
        local_chunks_used = []
        if local_chunk_objects:
            for idx, chunk_obj in enumerate(local_chunk_objects):
                raw_score = local_chunk_scores[idx] if local_chunk_scores and idx < len(local_chunk_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                chunk_info = {
                    "chunk_id": chunk_obj.get("chunk_id", idx),
                    "text": chunk_obj.get("text", ""),
                    "score": norm_score,
                    "metadata": chunk_obj.get("metadata", {})
                }
                local_chunks_used.append(chunk_info)
        elif local_chunk_ids:
            for idx, chunk_id in enumerate(local_chunk_ids):
                raw_score = local_chunk_scores[idx] if local_chunk_scores and idx < len(local_chunk_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                chunk_info = {
                    "chunk_id": chunk_id,
                    "text": local_context[idx] if idx < len(local_context) else "",
                    "score": norm_score
                }
                local_chunks_used.append(chunk_info)
        
        # Build global communities used metadata
        global_communities_used = []
        if global_community_ids:
            for idx, comm_id in enumerate(global_community_ids):
                raw_score = global_community_scores[idx] if global_community_scores and idx < len(global_community_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                comm_info = {
                    "community_id": comm_id,
                    "summary": global_context[idx] if idx < len(global_context) else "",
                    "score": norm_score
                }
                global_communities_used.append(comm_info)
        
        # Build entities used metadata
        entities_used = []
        if entities:
            for idx, entity in enumerate(entities):
                raw_score = entity_scores[idx] if entity_scores and idx < len(entity_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                entity_info = {
                    "entity": entity,
                    "score": norm_score
                }
                entities_used.append(entity_info)
        
        total_sources = len(local_context) + len(global_context)
        
        # Extract article/clause information from chunks
        sources_cited = []
        if local_context:
            for idx, chunk_text in enumerate(local_context):
                source_info = self._extract_source_info(chunk_text)
                raw_score = local_chunk_scores[idx] if local_chunk_scores and idx < len(local_chunk_scores) else None
                if isinstance(raw_score, tuple) and len(raw_score) >= 2:
                    norm_score = raw_score[1]
                else:
                    norm_score = raw_score
                sources_cited.append({
                    "chunk_id": local_chunk_ids[idx] if local_chunk_ids and idx < len(local_chunk_ids) else idx,
                    "text": chunk_text[:500],  # First 500 chars for preview
                    "article": source_info.get("article"),
                    "clause": source_info.get("clause"),
                    "full_text": chunk_text,
                    "score": norm_score
                })
        
        return {
            "answer": answer,
            "search_type": "hybrid",
            "num_local_chunks": len(local_context),
            "num_global_communities": len(global_context),
            "total_sources": total_sources,
            "local_chunk_ids": local_chunk_ids or list(range(len(local_context))),
            "local_chunks_used": local_chunks_used,
            "global_community_ids": global_community_ids or list(range(len(global_context))),
            "global_communities_used": global_communities_used,
            "entities": entities,
            "entities_used": entities_used,
            "local_context": local_context,
            "global_context": global_context,
            "sources_cited": sources_cited,
            "retrieval_metadata": {
                "chunks": local_chunks_used,
                "communities": global_communities_used,
                "entities": entities_used
            }
        }
    
    def _extract_source_info(self, chunk_text: str) -> Dict[str, Any]:
        """Extract Article and Clause information from chunk text.
        
        Args:
            chunk_text: The chunk text to parse
            
        Returns:
            Dictionary with extracted article and clause numbers
        """
        import re
        
        source_info = {"article": None, "clause": None}
        
        # Look for Article references (e.g., "Article 15", "Article 370")
        article_match = re.search(r'Article\s+(\d+[A-Z]?)', chunk_text, re.IGNORECASE)
        if article_match:
            source_info["article"] = article_match.group(1)
        
        # Look for clause references (e.g., "(1)", "(2)", "(3)")
        clause_matches = re.findall(r'\((\d+)\)', chunk_text)
        if clause_matches:
            source_info["clause"] = clause_matches[0]
        
        return source_info
    
    def generate_answer(
        self,
        question: str,
        retrieval_results: Dict[str, Any],
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """Generate answer based on retrieval results and search type.
        
        Args:
            question: User question
            retrieval_results: Results from retrieval system with chunk/community metadata
            search_type: Type of search (local, global, or hybrid)
            
        Returns:
            Dictionary with answer and comprehensive metadata
        """
        if search_type == "local":
            return self.generate_local_answer(
                question=question,
                context_chunks=retrieval_results.get("chunks", []),
                chunk_ids=retrieval_results.get("chunk_ids", []),
                chunk_objects=retrieval_results.get("chunk_objects", []),
                chunk_scores=retrieval_results.get("chunk_scores", []),
                entities=retrieval_results.get("entities", []),
                num_candidates=retrieval_results.get("num_candidates")
            )
        elif search_type == "global":
            return self.generate_global_answer(
                question=question,
                community_summaries=retrieval_results.get("community_summaries", []),
                community_ids=retrieval_results.get("community_ids", []),
                community_scores=retrieval_results.get("community_scores", [])
            )
        else:  # hybrid
            return self.generate_hybrid_answer(
                question=question,
                local_context=retrieval_results.get("local_chunks", []),
                global_context=retrieval_results.get("global_summaries", []),
                entities=retrieval_results.get("entities", []),
                local_chunk_ids=retrieval_results.get("local_chunk_ids", []),
                local_chunk_objects=retrieval_results.get("local_chunk_objects", []),
                local_chunk_scores=retrieval_results.get("local_chunk_scores", []),
                global_community_ids=retrieval_results.get("global_community_ids", []),
                global_community_scores=retrieval_results.get("global_community_scores", []),
                entity_scores=retrieval_results.get("entity_scores", [])
            )
