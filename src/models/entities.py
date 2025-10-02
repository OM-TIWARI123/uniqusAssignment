from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source document information"""
    company: str = Field(description="Company ticker symbol")
    year: str = Field(description="Filing year")
    excerpt: str = Field(description="Relevant excerpt from document")
    page: Optional[int] = Field(default=None, description="Page number")


class RagResponse(BaseModel):
    """Structured response from RAG system"""
    query: str = Field(description="Original user query")
    answer: str = Field(description="Generated answer")
    reasoning: str = Field(description="Reasoning behind the answer")
    sub_queries: List[str] = Field(default_factory=list, description="Sub-queries used")
    sources: List[Source] = Field(default_factory=list, description="Source documents")


class QueryDecomposition(BaseModel):
    """Query decomposition result"""
    needs_decomposition: bool = Field(description="Whether query needs decomposition")
    sub_queries: List[str] = Field(description="List of sub-queries")
    reasoning: str = Field(description="Reasoning for decomposition decision")