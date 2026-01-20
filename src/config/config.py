"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
import streamlit as st
# Define a limiter: e.g., 2 requests per second

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if GROQ_API_KEY is None:
        raise ValueError("GROQ_API_KEY is not set in environment variables or Streamlit secrets.")
    rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5, 
    check_every_n_seconds=0.1, 
    max_bucket_size=10
    )
    
    # Model Configuration
    LLM_MODEL = "llama-3.1-8b-instant"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return ChatGroq(model_name=cls.LLM_MODEL, api_key=cls.GROQ_API_KEY,temperature=0,max_tokens=512,timeout=10,rate_limiter=cls.rate_limiter)