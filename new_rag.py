import streamlit as st
from pathlib import Path
import sys
import os 
import time
sys.path.append(str(Path(__file__).parent))
import re
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
import tempfile
from pathlib import Path


    

# pip install presidio-analyzer presidio-anonymizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_pii(text):
    results = analyzer.analyze(text=text, language='en')
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    return anonymized_result.text
def save_uploaded_file(uploaded_file) -> Path:
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            return Path(tmp.name)
# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
def cleanup_old_file():
    """Removes the physical file and resets session state if uploader is cleared"""
    # If the widget returns None, it means the user clicked the 'x'
            
            # Reset session states
    st.session_state.temp_file_path = None
    st.session_state.uploaded = False
    st.session_state.rag_system = None
    st.session_state.llm=None
    # Clear embeddings from vectorstore
    vector_store.clear_vectorstore()
    # Clear resource cache to allow re-initialization later
    st.cache_resource.clear()

def delete_document():
    """Manually delete the uploaded document"""
    if 'temp_file_path' in st.session_state and st.session_state.temp_file_path:
        path = Path(st.session_state.temp_file_path)
        
        # Reset session states
        st.session_state.temp_file_path = None
        st.session_state.uploaded = False
        st.session_state.rag_system = None
        st.session_state.llm=None
        # Clear embeddings from vectorstore
        vector_store.clear_vectorstore()
        # Clear resource cache to allow re-initialization later
        st.cache_resource.clear()
def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if "llm" not in st.session_state:
        st.session_state.llm = None


vector_store = VectorStore()
@st.cache_resource
def initialize_rag_and_build_graph(file_path):
    """Initialize RAG and build graph in one function (cached)"""
    try:
        # Initialize document processor
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Process documents
        if file_path:
            documents = doc_processor.load_documents(file_path)
        
            # Create vector store
            vector_store.create_vectorstore(documents)
            os.remove(file_path)
            
            # Initialize LLM
            llm = Config.get_llm()
            st.session_state.llm = llm
            
            # Build RAG system with graph
            graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(),
                llm=llm
            )
            graph_builder.build()
            
            return graph_builder
        else:
            st.session_state.uploaded = False
            st.info("Please upload a document")
            
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None
def run_rag_query(rag_system, question):
    """Run a query on the RAG system"""
    try:
        question = scrub_pii(question)
        if rag_system:
            result = rag_system.run(question)
            return result
    except Exception as e:
        st.error(f"Failed to get answer: {str(e)}")
        return None
def main():
    init_session_state()
    # Title
    st.title("🔍 Legal RAG Document Asistant")
    st.markdown("Ask questions about the loaded documents")
    
            
    uploaded_file = st.file_uploader(
    "Upload documents",
    on_change=cleanup_old_file
    )
  

    temp_path=None
    if uploaded_file:
        st.session_state.uploaded=True
        uploaded_file.seek(0)
        temp_path = save_uploaded_file(uploaded_file)
        
        # Initialize RAG and build graph only if not already done
        if st.session_state.rag_system is None:
            rag_system = initialize_rag_and_build_graph(temp_path)
            st.session_state.rag_system = rag_system
        
        st.markdown("---")
        
        # Search interface
        with st.form("search_form"):
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know?"
            )
            submit = st.form_submit_button("🔍 Search")

        
        # Process search
        if submit and question:
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    
                    # Run RAG query
                    result = run_rag_query(st.session_state.rag_system, question)
                    
                    elapsed_time = time.time() - start_time
                    
                    if result:
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.success(result['answer'])
                        
                        # Show retrieved docs in expander
                        with st.expander("📄 Source Documents"):
                            for i, doc in enumerate(result['retrieved_docs'], 1):
                                st.text_area(
                                    f"Document {i}",
                                    doc.page_content[:300] + "...",
                                    height=100,
                                    disabled=True
                                )
                        
                        st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    else:
        st.info("File not uploaded")

    

    

if __name__ == "__main__":
    main()
