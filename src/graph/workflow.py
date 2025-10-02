from langgraph.graph import StateGraph, END
from src.rag.decomposer import create_decompose_node
from src.rag.retriever import create_retrieve_node 
from src.rag.synthesizer import create_synthesize_node
from src.rag.decomposer import State


def build_graph(vector_store, structured_llm, decomposition_llm):
    """Build the LangGraph workflow"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("decompose", create_decompose_node(decomposition_llm))
    workflow.add_node("retrieve", create_retrieve_node(vector_store))
    workflow.add_node("synthesize", create_synthesize_node(structured_llm))
    
    # Define edges
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


def run_query_engine(vector_store, structured_llm, decomposition_llm, query: str):
    """Run a query through the RAG pipeline"""
    graph = build_graph(vector_store, structured_llm, decomposition_llm)
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