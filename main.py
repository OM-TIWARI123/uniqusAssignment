import os
import json
import argparse
import warnings
from dotenv import load_dotenv

from config.settings import DATA_FOLDER
from src.data.downloader import download_filings
from src.data.converter import convert_html_to_pdf
from src.data.processor import extract_text_from_filings, chunk_documents
from src.vectorstore.chroma_store import create_vector_store, get_vector_store
from src.utils.llm_factory import initialize_llm
from src.graph.workflow import run_query_engine

warnings.filterwarnings("ignore")
load_dotenv()


def setup_pipeline():
    """Setup the complete document processing pipeline"""
    # Step 1: Download filings if needed
    if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
        download_filings()
    else:
        print(f"\n📁 Filings directory exists: {DATA_FOLDER}")
    
    # Step 2: Convert HTML to PDF
    convert_html_to_pdf(DATA_FOLDER)
    
    # Step 3: Extract text from PDFs
    documents = extract_text_from_filings(DATA_FOLDER)
    
    # Step 4: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 5: Create vector store
    vector_store = create_vector_store(chunks)
    
    return vector_store


def interactive_mode(vector_store, structured_llm, decomposition_llm):
    """Run the application in interactive mode"""
    print("\n" + "="*60)
    print("🤖 Interactive Query Mode")
    print("="*60)
    print("Enter your queries below. Type 'quit' to exit.\n")
    
    while True:
        query = input("💬 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n🔍 Processing your query...\n")
        response = run_query_engine(vector_store, structured_llm, decomposition_llm, query)
        print(json.dumps(response, indent=2))
        print("\n" + "-"*60 + "\n")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="10-K Filings RAG Query Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What was Microsoft's revenue in 2023?"
  python main.py  # Interactive mode
        """
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (optional, omit for interactive mode)"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip document processing and use existing vector store"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📊 10-K Filings RAG Query Engine")
    print("="*60)
    
    # Initialize LLMs
    _, structured_llm, decomposition_llm = initialize_llm()
    
    # Setup pipeline (or skip if requested)
    if not args.skip_setup:
        vector_store = setup_pipeline()
    else:
        print("\n⚠️  Skipping setup, connecting to existing vector store...")
        vector_store = get_vector_store()
    
    # Run query or interactive mode
    if args.query:
        print(f"\n🔍 Running query: {args.query}\n")
        response = run_query_engine(vector_store, structured_llm, decomposition_llm, args.query)
        print(json.dumps(response, indent=2))
    else:
        interactive_mode(vector_store, structured_llm, decomposition_llm)


if __name__ == "__main__":
    main()