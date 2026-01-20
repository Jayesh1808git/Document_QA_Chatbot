from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from qdrant_client.http.models import Distance, VectorParams
import streamlit as st
load_dotenv()
# ":memory:" is the magic keyword for a temporary local database



class VectorStore:
    """ Manages a vector store for document embeddings. """
    def __init__(self):
        self.embeddings= FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_store=None
        self.retriever=None
        self.client= QdrantClient(
            url=os.getenv("Qdrant_url") or st.secrets.get("Qdrant_url"),
            api_key=os.getenv("Qdrant_db") or st.secrets.get("Qdrant_db"),
            prefer_grpc=True,
            timeout=100
        )

    # ... inside your VectorStore class ...

    def create_vectorstore(self, documents: List[Document]):
        collection_name = "rag_collection"
        
        # 1. Check if collection exists; if not, create it
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384, # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                ),
            )
            print(f"Created new collection: {collection_name}")

        # 2. Now initialize the VectorStore safely
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        
        # 3. Add documents and setup retriever
        self.vector_store.add_documents(documents)
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever
    def clear_vectorstore(self):
        """Clear the vector store and retriever"""
        self.vector_store = None
        self.retriever = None