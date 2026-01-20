from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RagState(BaseModel):
    question:str
    retrieved_docs:List[Document]=[]
    answer:str=""