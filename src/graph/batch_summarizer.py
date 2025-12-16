"""Batch/parallel summarization for chunks and communities.

Uses concurrent workers to speed up LLM-based summarization without changing prompts.
"""

from typing import List, Dict, Any, Optional
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchSummarizer:
    """Generate summaries for chunks and communities with parallel workers."""

    def __init__(
        self,
        llm_client,
        show_progress: bool = True,
        max_workers: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.show_progress = show_progress
        self.max_workers = max_workers

    def _summarize_chunk_text(self, chunk_text: str) -> str:
        from ..llm.prompt_templates import PromptTemplates
        prompt = PromptTemplates.format_chunk_summary(chunk_text)
        try:
            summary = self.llm_client.generate(prompt, temperature=0.5, max_tokens=200)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating chunk summary: {e}")
            return (chunk_text or "")[:200] + "..."

    def summarize_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[int, str]:
        """Parallel chunk summarization. Returns mapping chunk_id -> summary."""
        logger.info(
            f"Generating summaries for {len(chunks)} chunks in parallel (workers={self.max_workers})"
        )

        summaries: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._summarize_chunk_text, chunk.get("text", "")): chunk["chunk_id"]
                for chunk in chunks
            }

            iterator = as_completed(futures)
            if self.show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc="Summarizing chunks (parallel)",
                    unit="chunk",
                    file=sys.stdout,
                    disable=False,
                )

            for future in iterator:
                chunk_id = futures[future]
                try:
                    summaries[chunk_id] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} summary failed: {e}")
                    summaries[chunk_id] = ""

        return summaries

    def _summarize_community(
        self,
        community_chunk_texts: List[str],
        community_entities: List[str],
        chunk_summaries: Optional[List[str]] = None,
    ) -> str:
        from ..llm.prompt_templates import PromptTemplates

        # Build summaries_list from provided summaries or generate ad-hoc
        if chunk_summaries is not None:
            summaries_list = chunk_summaries
        else:
            summaries_list = [self._summarize_chunk_text(txt) for txt in community_chunk_texts]

        prompt = PromptTemplates.format_community_summary(
            entities=community_entities, chunk_summaries=summaries_list
        )
        try:
            summary = self.llm_client.generate(prompt, temperature=0.6, max_tokens=300)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating community summary: {e}")
            return " ".join(summaries_list[:3])

    def summarize_communities(
        self,
        communities: Dict[int, List[int]],
        chunks: List[Dict[str, Any]],
        entities_by_community: Dict[int, List[str]],
        chunk_summaries: Optional[Dict[int, str]] = None,
    ) -> Dict[int, str]:
        """Parallel community summarization. Returns mapping community_id -> summary."""
        logger.info(
            f"Generating summaries for {len(communities)} communities in parallel (workers={self.max_workers})"
        )

        # Build quick lookups
        chunk_map = {c["chunk_id"]: c for c in chunks}

        def build_args(comm_id: int):
            # texts for this community
            texts = [chunk_map[cid]["text"] for cid in communities.get(comm_id, []) if cid in chunk_map]
            # entities
            ents = entities_by_community.get(comm_id, [])
            # summaries list aligned with texts if provided
            if chunk_summaries is not None:
                summaries_list = [chunk_summaries.get(cid, "") for cid in communities.get(comm_id, [])]
            else:
                summaries_list = None
            return texts, ents, summaries_list

        results: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._summarize_community, *build_args(comm_id)): comm_id
                for comm_id in communities.keys()
            }

            iterator = as_completed(futures)
            if self.show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc="Summarizing communities (parallel)",
                    unit="community",
                    file=sys.stdout,
                    disable=False,
                )

            for future in iterator:
                comm_id = futures[future]
                try:
                    results[comm_id] = future.result()
                except Exception as e:
                    logger.error(f"Community {comm_id} summary failed: {e}")
                    results[comm_id] = ""

        return results
