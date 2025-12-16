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
    # Main answer
    st.header("✏️ Answer")
    st.write(result.get("answer", ""))

    # Detailed context
    if result.get("context"):
        st.subheader("Key Changes Introduced")
        contexts = result["context"]
        if isinstance(contexts, list):
            for ctx in contexts:
                st.write("•", ctx)
        else:
            st.write(contexts)

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
