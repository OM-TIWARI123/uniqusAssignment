from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    company: str
    year: str
    excerpt: str
    page: Optional[int]

class RagResponse(BaseModel):
    query: str
    answer: str
    reasoning: Optional[str]
    sub_queries: List[str]
    sources: List[Source]
    

class QueryDecomposition(BaseModel):
    needs_decomposition: bool
    sub_queries: List[str]
    reasoning: str