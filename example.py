"""Example usage of Apna Sanvidhan system - Constitution of India SemRAG."""

import json
import os
from src.pipeline.apnasanvidhan import ApnaSanvidhan

def main():
    """Run example queries on the Apna Sanvidhan system."""
    
    # Initialize the system
    print("Initializing Apna Sanvidhan - Constitution of India Query System...")
    rag_system = ApnaSanvidhan(config_path="config.yaml")
    
    # Check if processed data exists and is usable
    chunks_path = "data/processed/chunks.json"
    processed_data_exists = os.path.exists(chunks_path) and os.path.getsize(chunks_path) > 0
    
    if not processed_data_exists:
        print("\nProcessing Constitution of India (this may take several minutes)...")
        print("Steps:")
        print("1. Loading PDF")
        print("2. Semantic chunking")
        print("3. Extracting constitutional entities (articles, rights, principles)")
        print("4. Building knowledge graph")
        print("5. Detecting constitutional thematic communities")
        print("6. Generating summaries")
        print("7. Initializing retrieval (FAISS vector indices, no PKL caches)")
        
        # Process the Constitution of India
        rag_system.process_document(pdf_path="data/Constitution_of_India.pdf")
        
        print("\n✓ Constitution processed successfully!")
        print("Processed data saved to data/processed/")
        # Show FAISS vector store stats
        try:
            stats = rag_system.vector_store.get_stats()
            print(f"FAISS indices → chunks: {stats.get('num_chunks', 0)}, entities: {stats.get('num_entities', 0)}, communities: {stats.get('num_communities', 0)}")
        except Exception:
            pass
    else:
        print("\nLoading previously processed constitutional data...")
        try:
            rag_system.load_processed_data()
            print("✓ Data loaded successfully!")
            # Show FAISS vector store stats
            try:
                stats = rag_system.vector_store.get_stats()
                print(f"FAISS indices → chunks: {stats.get('num_chunks', 0)}, entities: {stats.get('num_entities', 0)}, communities: {stats.get('num_communities', 0)}")
            except Exception:
                pass
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"⚠️  Processed data invalid or missing ({exc}); reprocessing document...\n")
            rag_system.process_document(pdf_path="data/Constitution_of_India.pdf")
            print("\n✓ Document reprocessed successfully!")
            print("Processed data saved to data/processed/")
            # Show FAISS vector store stats
            try:
                stats = rag_system.vector_store.get_stats()
                print(f"FAISS indices → chunks: {stats.get('num_chunks', 0)}, entities: {stats.get('num_entities', 0)}, communities: {stats.get('num_communities', 0)}")
            except Exception:
                pass
    
    # Example questions demonstrating different search types
    questions = [
        {
            "question": "What are the Fundamental Rights guaranteed by the Indian Constitution?",
            "search_type": "local",
            "description": "LOCAL SEARCH (Entity-based) - Best for specific constitutional articles"
        },
        {
            "question": "Explain the constitutional structure and organization of the Indian government.",
            "search_type": "global",
            "description": "GLOBAL SEARCH (Community-based) - Best for comprehensive constitutional topics"
        },
        {
            "question": "How does the Constitution protect minority rights and ensure secularism?",
            "search_type": "hybrid",
            "description": "HYBRID SEARCH (Combined) - Best for complex constitutional questions"
        }
    ]
    
    # Query the system
    print("\n" + "="*80)
    print("APNA SANVIDHAN - CONSTITUTION OF INDIA QUERY EXAMPLES")
    print("="*80)
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}: {q['question']}")
        print(f"Search Type: {q['description']}")
        print("─"*80)
        
        # Query the system
        result = rag_system.query(
            question=q["question"],
            search_type=q["search_type"]
        )
        
        # Display results
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {len(result.get('context', []))} constitutional sections")
        print(f"Constitutional entities referenced: {', '.join(result.get('entities', [])[:5])}")
        if len(result.get('entities', [])) > 5:
            print(f"  ... and {len(result['entities']) - 5} more")
    
    print("\n" + "="*80)
    print("Interactive Mode - Query the Constitution of India")
    print("="*80)
    print("You can now ask your own questions about the Constitution!")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        # Get user input
        user_question = input("Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using Apna Sanvidhan!")
            break
        
        if not user_question:
            continue
        
        # Ask for search type
        print("Search type: [1] Local (specific), [2] Global (broad), [3] Hybrid (default)")
        search_choice = input("Choose (1/2/3 or press Enter for hybrid): ").strip()
        
        search_type_map = {
            '1': 'local',
            '2': 'global',
            '3': 'hybrid',
            '': 'hybrid'
        }
        search_type = search_type_map.get(search_choice, 'hybrid')
        
        # Query the system
        print(f"\nSearching Constitution ({search_type} search)...")
        result = rag_system.query(
            question=user_question,
            search_type=search_type
        )
        
        # Display result
        print(f"\n{'-'*80}")
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: {len(result.get('context', []))} constitutional sections")
        if result.get('entities'):
            print(f"Key constitutional entities: {', '.join(result['entities'][:5])}")
        print('-'*80 + "\n")

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it before running (Windows PowerShell):")
        print("  $env:OPENAI_API_KEY = 'your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
