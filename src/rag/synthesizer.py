import re
from langchain_core.prompts import ChatPromptTemplate
from .decomposer import State
from src.models.entities import Source


def create_synthesize_node(structured_llm):
    """Factory function to create synthesize node with LLM"""
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
                
                # Parse company from filename
                company = "Unknown"
                if "1652044" in source_path:
                    company = "GOOGL"
                elif "0789019" in source_path:
                    company = "MSFT"
                elif "1045810" in source_path:
                    company = "NVDA"
                
                # Parse year from filename
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
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst. Provide structured answers based on the given context."),
            ("human", """Original Question: {question}

Context:
{context}

Sub-queries asked: {sub_queries}

Provide a comprehensive answer using the schema: RagResponse with query, answer, reasoning, sub_queries, and sources fields.""")
        ])
        
        # Create chain and invoke
        chain = prompt | structured_llm
        response = chain.invoke({
            "question": state["question"],
            "context": context,
            "sub_queries": state.get("sub_queries", [])
        })
        
        # Convert response to dict and update sources
        result = response.model_dump()
        result["sources"] = [s.model_dump() for s in sources]
        
        return result
    
    return synthesize_node