import os
import json
import re
from sec_edgar_downloader import Downloader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import argparse
import warnings
from langchain_chroma import Chroma
import chromadb
from chromadb import HttpClient
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Any
from entity import Source, RagResponse, QueryDecomposition
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = "AIzaSyDZb9IUA-oMq41E6KTMBXjd-2KJiEjDq0A")
from langgraph.graph import StateGraph, END
import argparse
from langchain_core.prompts import ChatPromptTemplate


structured_llm = llm.with_structured_output(schema=RagResponse)
decomposition_llm = llm.with_structured_output(schema=QueryDecomposition)




warnings.filterwarnings("ignore")
class State(TypedDict):
    question: str
    needs_decomposition: bool
    sub_queries: List[str]
    reasoning: str
    retrieved: List[Any]
    answer: str
    sources: List[dict] 


# Company information
COMPANIES = {
    "GOOGL": "0001652044",
    "MSFT": "0000789019",
    "NVDA": "0001045810"
}

YEARS = [2022, 2023, 2024]

def download_filings():
    """Download 10-K filings for specified companies and years"""
    print("Downloading 10-K filings...")
    
    # Create data folder
    data_folder = "Data_Scope"
    os.makedirs(data_folder, exist_ok=True)
    
    # Initialize downloader with a placeholder email
    dl = Downloader("Uniqus Assignment", "omt887@gmail.com", data_folder)
    
    # Download 10-K filings for each company and year
    total_filings = 0
    for ticker, cik in COMPANIES.items():
        print(f"Downloading 10-K filings for {ticker} (CIK: {cik})...")
        for year in YEARS:
            try:
                # Download 10-K filing for the specific year
                dl.get("10-K", cik, after=str(year)+"-01-01", before=str(year)+"-12-31", download_details=True)
                print(f"  Successfully downloaded {ticker} 10-K for {year}")
                total_filings += 1
            except Exception as e:
                print(f"  Failed to download {ticker} 10-K for {year}: {str(e)}")
    
    print(f"\nDownload complete! Total filings downloaded: {total_filings}")
    return data_folder

def extract_text_from_filings(data_folder):
    """Extract text from downloaded filings"""
    print("Extracting text from filings...")

    # Load PDF documents
    loader = DirectoryLoader(
        data_folder,
        glob="**/*.pdf",      # now looks for PDFs
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    return documents 

def chunk_documents(documents):
    """Split documents into chunks"""
    print("Chunking documents...")
    
    # Initialize text splitter with more reasonable parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks


def create_vector_store(chunks):
    """Create vector store using Chroma over HTTP"""
    print("Creating vector store connected to remote Chroma server...")

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create HttpClient instance
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    vector_store = Chroma(
        collection_name="sec_filings",
        embedding_function=embedding_model,
        client=chroma_client
    )

    # Add documents in batches
    batch_size = 500
    total_chunks = min(len(chunks), 5000)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Adding chunk batch {i//batch_size + 1}...")
        vector_store.add_documents(batch)

    print("Vector store created and populated!")
    return vector_store





def query_decomposition(query):
    """Use LLM to intelligently decompose complex queries into sub-queries"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent query decomposition assistant. Analyze user queries about financial data and break them down into specific sub-queries.

Guidelines:
- If the query asks "which company" or implies comparison, create sub-queries for MSFT, GOOGL, and NVDA
- If the query mentions specific metrics (revenue, margin, profit, earnings), include the metric name in each sub-query
- Include both company ticker and full name for better retrieval (e.g., "Microsoft MSFT")
- If the query asks for calculations (growth rates), include sub-queries for raw data needed
- For simple direct queries about one company, return just that query
- ALWAYS include the year in the sub-query"""),
        
        ("human", """Query: "{query}"

Examples:

Query: "What was Microsoft's total revenue in 2023?"
needs_decomposition: false
sub_queries: ["Microsoft MSFT total revenue 2023"]
reasoning: "Direct query for single company metric"

Query: "Which company had the highest operating margin in 2023?"
needs_decomposition: true
sub_queries: ["Microsoft MSFT operating margin 2023", "Google GOOGL operating margin 2023", "NVIDIA NVDA operating margin 2023"]
reasoning: "Comparison query requires retrieving operating margin for all three companies"

Query: "Compare AI investments across all three companies in 2024"
needs_decomposition: true
sub_queries: ["Microsoft MSFT AI investment spending 2024", "Google GOOGL AI investment spending 2024", "NVIDIA NVDA AI investment spending 2024"]
reasoning: "Multi-company comparison of AI investments"

Now decompose this query: "{query}"
""")
    ])
    
    chain = prompt | decomposition_llm
    
    try:
        response = chain.invoke({"query": query})
        
        return {
            "needs_decomposition": response.needs_decomposition,
            "sub_queries": response.sub_queries,
            "reasoning": response.reasoning
        }
    except Exception as e:
        # Fallback to original query if LLM fails
        print(f"Decomposition failed: {str(e)}")
        return {
            "needs_decomposition": False,
            "sub_queries": [query],
            "reasoning": f"Decomposition failed: {str(e)}"
        }


def decompose_node(state: State) -> State:
    result = query_decomposition(state["question"])
    return {
        "needs_decomposition": result["needs_decomposition"],
        "sub_queries": result["sub_queries"],
        "reasoning": result["reasoning"]
    }

def retrieve_node(state: State, vector_store) -> State:
    docs = []
    if state["needs_decomposition"]:
        for q in state["sub_queries"]:
            retrieved_docs = vector_store.similarity_search(q, k=3)
            docs.append({"query": q, "docs": retrieved_docs})
    else:
        retrieved_docs = vector_store.similarity_search(state["question"], k=3)
        docs.append({"query": state["question"], "docs": retrieved_docs})
    
    return {"retrieved": docs}
from langchain_core.prompts import ChatPromptTemplate

def synthesize_node(state: State) -> dict:
    # Collect context from retrieved documents
    context_texts = []
    sources = []
    
    for item in state["retrieved"]:
        docs = item["docs"]
        query = item["query"]
        
        joined_docs = "\n".join([doc.page_content for doc in docs])
        context_texts.append(f"Query: {query}\nDocs:\n{joined_docs}")
        
        # Build Source objects from doc metadata
        for doc in docs[:1]:  # take top 1 per query
            meta = doc.metadata or {}
            source_path = meta.get("source", "")
            
            # Try to parse company/year from filename
            company = "Unknown"
            if "1652044" in source_path:
                company = "GOOGL"
            elif "0789019" in source_path:
                company = "MSFT"
            elif "1045810" in source_path:
                company = "NVDA"
            
            year = None
            match = re.search(r"(\d{4})", source_path)
            if match:
                year = match.group(1)
            
            sources.append(Source(
                company=company,
                year=year or "Unknown",
                excerpt=doc.page_content[:200] + "...",
                page=meta.get("page")
            ))
    
    # Combine context
    context = "\n\n".join(context_texts)
    
    # Create a proper prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst. Provide structured answers based on the given context."),
        ("human", """Original Question: {question}

Context:
{context}

Sub-queries asked: {sub_queries}

Provide a comprehensive answer using the schema: RagResponse with query, answer, reasoning, sub_queries, and sources fields.""")
    ])
    
    # Create a chain with the prompt and structured LLM
    chain = prompt | structured_llm
    
    # Invoke with proper input
    response = chain.invoke({
        "question": state["question"],
        "context": context,
        "sub_queries": state.get("sub_queries", [])
    })
    
    # Convert response to dict and update sources
    result = response.model_dump()
    
    # Override sources with our manually parsed ones (better metadata)
    print("reesult", result)
    print("length of sources", len(result["sources"]))
    
    result["sources"] = [s.model_dump() for s in sources]
    
    return result


def build_graph(vector_store):
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, vector_store))
    workflow.add_node("synthesize", synthesize_node)

    # Edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()
def run_query_engine(vector_store, query: str):
    # Build graph
    graph = build_graph(vector_store)
    state = {"question": query}
    result = graph.invoke(state)
    final_response = {
        "query": result.get("question"),
        "answer": result.get("answer"),
        "reasoning": result.get("reasoning"),
        "sub_queries": result.get("sub_queries", []),
        "sources": result.get("sources", [])
    }
    
    return final_response


def main():
    """Main function to run the CLI application"""
    parser = argparse.ArgumentParser(description="10-K Filings RAG Query Engine")
    parser.add_argument("--query", type=str, help="Query to run against filings")
    args = parser.parse_args()

    print("10-K Filings RAG Query Engine")
    print("=" * 40)

    # Download filings if they don't exist
    data_folder = "Data_Scope"
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        data_folder = download_filings()
    else:
        print("Filings already downloaded. Skipping download step.")

    documents = extract_text_from_filings(data_folder)

    
    chunks = chunk_documents(documents)

    
    vector_store = create_vector_store(chunks)
    if args.query:
        response = run_query_engine(vector_store, args.query)
        print(json.dumps(response, indent=2))
    else:
        # Interactive mode
        print("\n" + "=" * 40)
        print("Interactive Query Mode")
        print("Enter 'quit' to exit")
        print("=" * 40)

        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'quit':
                break

            if query:
                response = run_query_engine(vector_store, query)
                print("\n" + json.dumps(response, indent=2))


if __name__ == "__main__":
    main()