# Apna Sanvidhan: SemRAG Implementation for the Constitution of India

A Semantic Knowledge-Augmented RAG (Retrieval-Augmented Generation) system for answering questions about the Constitution of India, based on the [SemRAG research paper](https://arxiv.org/abs/2507.21110).

## Overview

Apna Sanvidhan implements the SemRAG architecture specifically tailored for the Indian Constitution, combining:
- **Semantic Chunking**: Intelligent text segmentation of constitutional provisions based on semantic similarity
- **Entity Extraction**: LLM-based extraction of constitutional entities (articles, rights, principles, institutions)
- **Knowledge Graph**: Construction of entity-chunk-community graph structure of constitutional elements
- **Community Detection**: Identification of thematic constitutional communities using Leiden/Louvain algorithms
- **Hierarchical Summarization**: Constitutional article-level and community-level summaries
- **Hybrid Retrieval**: Local (constitutional entity-based) and global (thematic community-based) search

## Features

✅ **Semantic Chunking with Buffer Merging**: Constitutional provision segmentation
✅ **Constitutional Entity Extraction**: Articles, rights, principles, institutions extraction
✅ **Multi-level Summarization**: Article and constitutional community summaries
✅ **Hybrid Search**: Combines local and global retrieval strategies
✅ **Configurable Pipeline**: YAML-based configuration
✅ **Persistent Storage**: Save and reload processed constitutional data
✅ **Embedding Caching**: Persistent pickle-based caching for sentences, chunks, entities, and communities
✅ **Batch Processing**: Optimized batch embedding generation with configurable batch sizes
✅ **Parallel Entity Extraction**: Concurrent LLM API calls with ThreadPoolExecutor for faster extraction

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM provider)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/apna_sanvidhan.git
cd apna_sanvidhan
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. **Set up environment variables**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
```

## Quick Start

### Basic Usage

```python
from src.pipeline.apnasanvidhan import ApnaSanvidhan

# Initialize system
rag_system = ApnaSanvidhan(config_path="config.yaml")

# Process the Constitution of India
rag_system.process_document(pdf_path="data/Constitution_of_India.pdf")

# Query the system
result = rag_system.query(
    "What are the Fundamental Rights in the Indian Constitution?",
    search_type="hybrid"  # Options: local, global, hybrid
)

print(result["answer"])
```

### Using Pre-processed Data

```python
# Load previously processed constitutional data
rag_system = ApnaSanvidhan()
rag_system.load_processed_data()

# Query immediately
result = rag_system.query("What is Article 15 of the Constitution?")
print(result["answer"])
```

## Configuration

Edit `config.yaml` to customize the system:

```yaml
# Chunking parameters
chunking:
  similarity_threshold: 0.7  # Semantic boundary detection
  buffer_size: 1  # Context sentences (0, 1, 3, 5)
  min_chunk_size: 100
  max_chunk_size: 1000
  batch_size: 50  # Batch size for embedding processing

# Entity extraction for Constitution
entity_extraction:
  method: "llm"  # Use LLM for entity extraction
  batch_size: 10  # Batch size for entity extraction
  max_workers: 5  # Number of parallel workers for concurrent API calls
  use_extraction_cache: true  # Cache entity extraction results
  skip_duplicate_chunks: true  # Skip extracting from duplicate chunk texts
  entity_types:
    - PERSON  # Constitutional framers, officials, judges
    - ORGANIZATION  # Government bodies, institutions, Parliament, Courts
    - LOCATION  # States, territories, geographical divisions
    - ARTICLE/SECTION  # Constitutional articles and sections
    - FUNDAMENTAL_RIGHT  # Rights guaranteed by Constitution
    - DIRECTIVE_PRINCIPLE  # Directive Principles of State Policy
    - DUTY  # Fundamental Duties
    - CONCEPT  # Constitutional principles (sovereignty, democracy, secularism)
    - DATE  # Important constitutional dates and amendments
  extract_relationships: true
  max_entities_per_chunk: 20

# Data storage and caching
data:
  cache_dir: "./data/cache"  # Directory for embedding cache files

# Community detection
community_detection:
  algorithm: leiden  # leiden or louvain
  resolution: 1.0
  min_community_size: 2

# Retrieval
retrieval:
  local_search:
    top_k_entities: 10
    top_k_chunks: 5
    batch_size: 50  # Batch size for chunk embedding processing
  global_search:
    top_k_communities: 5
    batch_size: 50  # Batch size for community embedding processing
  hybrid:
    local_weight: 0.5
    global_weight: 0.5
```

## Architecture

The SemRAG pipeline consists of:

### 1. Document Processing
- **PDF Loading**: Extract text from the Constitution of India PDF
- **Semantic Chunking**: Split constitutional text based on semantic similarity
- **Buffer Merging**: Add context from surrounding constitutional provisions

### 3. Summarization
- **Chunk Summaries**: Concise summaries of each constitutional provision
- **Community Summaries**: High-level thematic summaries of constitutional areas

### 4. Retrieval & Generation
- **Local Search**: Constitutional entity-based retrieval (specific articles)
- **Global Search**: Constitutional thematic community-based retrieval (broad principles)
- **Hybrid Search**: Combined local and global constitutional retrieval
- **Answer Generation**: LLM-based answer synthesis

## Search Types

### Local Search (Entity-based)
Best for: **Specific, detailed constitutional questions**
- Retrieves provisions mentioning relevant entities (articles, rights, principles)
- Provides detailed, context-specific answers
- Example: "What is Article 15 of the Constitution?"

### Global Search (Community-based)
Best for: **High-level, thematic constitutional questions**
- Retrieves community summaries from constitutional themes
- Provides broad, synthesized constitutional answers
- Example: "What are the Fundamental Rights in the Constitution?"

### Hybrid Search
Best for: **Complex constitutional questions requiring both detail and breadth**
- Combines local and global retrieval
- Balances specific articles with overall constitutional principles
- Example: "How does the Constitution protect minority rights?"

## Project Structure

```
apna_sanvidhan/
├── config.yaml                      # Configuration file
├── requirements.txt                 # Dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
├── data/
│   ├── Constitution_of_India.pdf   # Input document
│   └── processed/                  # Processed constitutional data
│       ├── chunks.json
│       ├── entities.json
│       ├── graph.json
│       ├── communities.json
│       └── summaries.json
├── src/
    ├── cache/                      # Embedding caching and batch processing
    │   ├── embedding_cache.py      # Pickle-based persistent embedding cache
    │   └── batch_processor.py      # Batch embedding processor with cache
    ├── chunking/                   # Semantic chunking modules
    │   ├── semantic_chunker.py
    │   └── buffer_merger.py
    ├── graph/                      # Graph construction
    │   ├── batch_entity_extractor.py  # Parallel entity extraction
    │   ├── batch_summarizer.py        # Parallel summarization
    │   ├── graph_builder.py
    │   └── community_detector.py
    ├── llm/                        # LLM interaction
    │   ├── llm_client.py
    │   ├── prompt_templates.py
    │   └── answer_generator.py
    ├── retrieval/                  # Retrieval modules
    │   ├── local_search.py
    │   ├── global_search.py
    │   └── ranker.py
    └── pipeline/                   # Main pipeline
        └── apnasanvidhan.py

```
## API Reference

### ApnaSanvidhan

Main pipeline class for Constitution of India queries.

#### Methods

- `process_document(pdf_path=None, text=None)`: Process the Constitution PDF through the pipeline
- `query(question, search_type='hybrid')`: Query the Constitution
- `save_processed_data()`: Save processed constitutional data to disk
- `load_processed_data()`: Load previously processed constitutional data

### Search Types

- `local`: Constitutional entity-based search for specific articles
- `global`: Thematic community-based search for constitutional principles
- `hybrid`: Combined local and global search (recommended)

## Performance Tips

### 1. Embedding Caching
- **First run**: Embeddings are computed and cached to `./data/cache/`
- **Subsequent runs**: Embeddings loaded from cache instantly (2-3 seconds total)
- **Cache files**: 
  - `sentence_embeddings.pkl` - Sentence embeddings (hash-keyed)
  - `chunk_embeddings.pkl` - Chunk embeddings (ID-keyed)
  - `entity_embeddings.pkl` - Entity embeddings (hash-keyed)
  - `community_embeddings.pkl` - Community embeddings (ID-keyed)

### 2. Batch Processing
- **Chunking embeddings**: Set `chunking.batch_size: 50` (default)
- **Retrieval embeddings**: Set `retrieval.local_search.batch_size: 50` (default)
- **Global search**: Set `retrieval.global_search.batch_size: 50` (default)
- Larger batches = faster (if API allows), smaller batches = safer (rate limits)

### 3. Parallel Entity Extraction
- **Max workers**: Set `entity_extraction.max_workers: 5` (default)
- **Higher workers** (5-10): Faster but higher API load
- **Lower workers** (1-3): Slower but safer for rate limits
- **Batch size**: Set `entity_extraction.batch_size: 10` (default)
- **Skip duplicates**: Enable `entity_extraction.skip_duplicate_chunks: true` to skip redundant text
- **Typical speedup**: 3-5x faster than sequential extraction

### 4. Buffer Size
- **0**: No context (fastest, less accurate)
- **1**: One sentence context (balanced)
- **3-5**: More context (slower, more accurate)

### 5. Community Detection
- Use Leiden for better quality (requires igraph)
- Use Louvain for faster processing

### 6. End-to-End Speed
- **Full processing** (first run): ~5-10 minutes (depending on API rate limits)
- **Cached loading** (subsequent runs): ~2-3 seconds
- **Query response**: ~2-5 seconds with hybrid search

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_chunking.py
```

## Research Paper

This implementation is based on:

**SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering**
- Authors: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen
- arXiv: [2507.21110](https://arxiv.org/abs/2507.21110)

Key contributions:
- Semantic chunking with buffer merging
- Entity-based knowledge graph construction
- Hierarchical community-based summarization
- Hybrid local-global retrieval strategy

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this implementation, please cite the original SemRAG paper:

```bibtex
@article{zhong2025semrag,
  title={SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering},
  author={Zhong, Kezhen and Suleiman, Basem and Erradi, Abdelkarim and Chen, Shijing},
  journal={arXiv preprint arXiv:2507.21110},
  year={2025}
}
```

And acknowledge the Constitution of India as the primary source:

```
Constitution of India - The fundamental law of India
Adopted by the Constituent Assembly of India on November 26, 1949
```

## Acknowledgments

- SemRAG research paper authors
- The Constitution of India and the Constituent Assembly
- OpenAI for GPT models and embeddings
- NetworkX, NLTK, and other open-source libraries

## Support

For issues, questions, or contributions:
- GitHub Issues: [issues](https://github.com/Bhargav-0718/Apna_Sanvidhan/issues)
- Email: bhargav.07.bidkar@gmail.com

---