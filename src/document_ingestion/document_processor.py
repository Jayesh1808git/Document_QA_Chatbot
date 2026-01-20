from typing import List , Union
from langchain_community.document_loaders import PyPDFLoader ,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path




class DocumentProcessor:
    """ Handles Document Loading and Splitting and processing."""
    def __init__(self,chunk_size:int=500,chunk_overlap:int=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "§", "\n", " "]

        )
    def load_from_pdf(self, file_path:Union[str,Path]) -> List[Document]:
        """ Load document from a single PDF file """
        loader=PyPDFLoader(str(file_path))
        return loader.load()
    def load_from_word(self, file_path:Union[str,Path]) -> List[Document]:
        """ Load document from a single PDF file """
        loader=Docx2txtLoader(str(file_path))
        return loader.load()
    def split_documents(self, documents:List[Document])->List[Document]:
        """ Split documents into smaller chunks """
        return self.text_splitter.split_documents(documents)
    def load_documents(self,src: Union[str, Path])->List[Document]:
        src=str(src)
        docs:List[Document] = []
        if src.lower().endswith(".pdf"):
            docs.extend(self.load_from_pdf(src))
        elif src.lower().endswith(".docx"):
            docs.extend(self.load_from_word(src))
        else:
            raise ValueError(f"Unsupported source type: {src}")
        return self.split_documents(docs)
    
    

            

