from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import LLM_MODEL, GOOGLE_API_KEY
from src.models.entities import RagResponse, QueryDecomposition


def initialize_llm():
    """Initialize the LLM and structured output versions"""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    
    structured_llm = llm.with_structured_output(schema=RagResponse)
    decomposition_llm = llm.with_structured_output(schema=QueryDecomposition)
    
    return llm, structured_llm, decomposition_llm