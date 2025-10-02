from .decomposer import State
from config.settings import RETRIEVAL_K


def create_retrieve_node(vector_store):
    """Factory function to create retrieve node with vector store"""
    def retrieve_node(state: State) -> State:
        docs = []
        if state["needs_decomposition"]:
            for q in state["sub_queries"]:
                retrieved_docs = vector_store.similarity_search(q, k=RETRIEVAL_K)
                docs.append({"query": q, "docs": retrieved_docs})
        else:
            retrieved_docs = vector_store.similarity_search(state["question"], k=RETRIEVAL_K)
            docs.append({"query": state["question"], "docs": retrieved_docs})
        
        return {"retrieved": docs}
    return retrieve_node