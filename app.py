"""
Streamlit UI for Apna Sanvidhan Question Answering System.
"""

import streamlit as st
import sys
from pathlib import Path
import logging
from typing import Dict, Any

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.apnasanvidhan import ApnaSanvidhan

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Apna Sanvidhan - Constitutional Q&A",
    page_icon="📜",
    layout="wide"
)

# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
/* Answer container adapts to theme */
.answer-card {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    padding: 2rem;
    border-radius: 12px;
    margin-top: 1rem;
    line-height: 1.7;
    border: 1px solid rgba(128,128,128,0.15);
}

/* Main section header */
.answer-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-color);
}

/* Subsections inside answer */
.answer-subtitle {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

/* Muted text that adapts */
.answer-muted {
    opacity: 0.85;
}

/* Source blocks */
.source-box {
    background-color: rgba(128,128,128,0.08);
    padding: 1rem;
    border-radius: 8px;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Pipeline loader (CRITICAL FIX HERE)
# ---------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    """
    Load and initialize the Apna Sanvidhan pipeline.
    This function is cached to avoid repeated heavy initialization.
    """
    with st.spinner("🔄 Initializing Apna Sanvidhan pipeline..."):
        try:
            config_path = ROOT_DIR / "config.yaml"
            pdf_path = ROOT_DIR / "data" / "Constitution_of_India.pdf"
            processed_dir = ROOT_DIR / "data" / "processed"

            pipeline = ApnaSanvidhan(config_path=str(config_path))

            # ---------------------------------------------------------
            # Correct lifecycle handling
            # ---------------------------------------------------------
            if processed_dir.exists() and any(processed_dir.iterdir()):
                st.info("📂 Loading processed constitutional data...")
                pipeline.load_processed_data()
            else:
                st.info(
                    "📄 Processing Constitution of India for the first time.\n\n"
                    "⏳ This may take several minutes. Please wait..."
                )
                pipeline.process_document(str(pdf_path))

            return pipeline

        except Exception as e:
            logger.error("Pipeline initialization failed", exc_info=True)
            st.error(f"❌ Failed to initialize pipeline: {e}")
            return None

# ---------------------------------------------------------------------
# Result formatter
# ---------------------------------------------------------------------
def format_answer(result):
    """Format and display the answer with all metadata."""
    
    # Main answer
    st.header("✏️ Answer")
    st.write(result.get("answer", ""))
    
    # Display source statistics
    if result.get("search_type") == "hybrid":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Local Chunks", result.get("num_local_chunks", 0))
        with col2:
            st.metric("Global Communities", result.get("num_global_communities", 0))
        with col3:
            st.metric("Total Sources", result.get("total_sources", 0))
    elif result.get("search_type") == "local":
        # Prefer chunks actually used; fall back to candidates
        used_count = len(result.get("chunks_used", []))
        st.metric("Constitutional Sections Used", used_count or result.get("num_candidates", 0))
    elif result.get("search_type") == "global":
        # Global answers return 'context' as community summaries
        st.metric("Community Summaries Used", len(result.get("context", [])))
    
    # Display references block based on search type
    st.markdown("---")
    if result.get("search_type") == "global":
        st.markdown("### 🧭 Community Summaries Referenced")
        communities_used = result.get("communities_used", [])
        if communities_used:
            for idx, comm in enumerate(communities_used, 1):
                with st.container():
                    st.markdown(f"**Community {comm.get('community_id', idx)}**")
                    st.write(comm.get("summary", ""))
                    score = comm.get("score")
                    if score is not None:
                        try:
                            st.caption(f"Relevance Score: {float(score):.2%}")
                        except (TypeError, ValueError):
                            st.caption(f"Relevance Score: {score}")
                    st.divider()
        else:
            # Fallback to raw context summaries
            summaries = result.get("context", [])
            if summaries:
                for idx, summary in enumerate(summaries, 1):
                    with st.container():
                        st.markdown(f"**Community {idx}**")
                        st.write(summary)
                        st.divider()
            else:
                st.info("No community summaries were directly referenced for this answer.")
    else:
        st.markdown("### 📜 Constitutional Sections Referenced")
        # For local/hybrid, 'sources_cited' contains parsed article/clause. Fallback to chunks_used
        if result.get("sources_cited"):
            for idx, source in enumerate(result["sources_cited"], 1):
                with st.container():
                    # Article and Clause header
                    header_parts = []
                    if source.get("article"):
                        header_parts.append(f"Article {source['article']}")
                    if source.get("clause"):
                        header_parts.append(f"Clause {source['clause']}")
                    
                    if header_parts:
                        st.markdown(f"**{' - '.join(header_parts)}**")
                    else:
                        st.markdown(f"**Section {idx}**")
                    
                    # Full text
                    st.write(source.get("full_text", source.get("text", "")))
                    
                    # Score if available
                    if source.get("score") is not None:
                        score_val = source["score"]
                        # Handle tuple scores (e.g., (id, score)) or plain numbers
                        if isinstance(score_val, tuple) and len(score_val) >= 2:
                            score_val = score_val[1]
                        try:
                            score_float = float(score_val)
                            st.caption(f"Relevance Score: {score_float:.2%}")
                        except (TypeError, ValueError):
                            st.caption(f"Relevance Score: {score_val}")
                    
                    st.divider()
        else:
            # Fallback for local search when sources_cited missing
            chunks_used = result.get("chunks_used", [])
            if chunks_used:
                for idx, source in enumerate(chunks_used, 1):
                    with st.container():
                        st.markdown(f"**Section {idx}**")
                        st.write(source.get("text", ""))
                        score = source.get("score")
                        if score is not None:
                            try:
                                st.caption(f"Relevance Score: {float(score):.2%}")
                            except (TypeError, ValueError):
                                st.caption(f"Relevance Score: {score}")
                        st.divider()
            else:
                st.info("No specific constitutional sections were directly referenced for this answer.")

# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    # Header
    st.title("📜 Apna Sanvidhan")
    st.subheader("Constitutional Question Answering System")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        search_type = st.selectbox(
            "Search Method",
            options=["hybrid", "local", "global"],
            help="Local: Entity-based | Global: Community-based | Hybrid: Combined",
        )

        st.markdown("---")
        st.markdown("### FAISS Index Stats")
        stats_placeholder = st.container()

        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown(
            """
            - What are the Fundamental Rights?
            - What does Article 15 say?
            - How is the President of India elected?
            - What are the Directive Principles of State Policy?
            """
        )

    # Load pipeline
    pipeline = load_pipeline()

    if pipeline is None:
        st.stop()

    st.success("✅ Pipeline ready")

    # Show vector store stats (num_* keys now exposed by vector store)
    try:
        stats = pipeline.vector_store.get_stats()
        with stats_placeholder:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", stats.get("num_chunks", 0))
            col_b.metric("Entities", stats.get("num_entities", 0))
            col_c.metric("Communities", stats.get("num_communities", 0))
    except Exception as e:
        logger.warning(f"Failed to fetch vector store stats: {e}")

    # Query input
    st.markdown("### 💬 Ask a Question")
    query = st.text_area(
        "Enter your question about the Indian Constitution:",
        height=100,
        placeholder="e.g., What does Article 15 tell?",
    )

    # Search button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True)

    # Handle query
    if search_button:
        if not query.strip():
            st.warning("⚠️ Please enter a valid question.")
        else:
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    result = pipeline.query(query, search_type=search_type)
                    st.markdown("---")
                    format_answer(result)
                except Exception as e:
                    logger.error("Query failed", exc_info=True)
                    st.error(f"❌ Error processing query: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#666;">'
        'Built with ❤️ using Streamlit · Powered by SemRAG'
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
