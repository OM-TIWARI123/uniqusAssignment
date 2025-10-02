from typing import TypedDict, List, Any
from langchain_core.prompts import ChatPromptTemplate


class State(TypedDict):
    question: str
    needs_decomposition: bool
    sub_queries: List[str]
    reasoning: str
    retrieved: List[Any]
    answer: str
    sources: List[dict]


def query_decomposition(query, decomposition_llm):
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
        print(f"⚠️  Decomposition failed: {str(e)}")
        return {
            "needs_decomposition": False,
            "sub_queries": [query],
            "reasoning": f"Decomposition failed: {str(e)}"
        }


def create_decompose_node(decomposition_llm):
    """Factory function to create decompose node with LLM"""
    def decompose_node(state: State) -> State:
        result = query_decomposition(state["question"], decomposition_llm)
        return {
            "needs_decomposition": result["needs_decomposition"],
            "sub_queries": result["sub_queries"],
            "reasoning": result["reasoning"]
        }
    return decompose_node