import logging
import streamlit as st
import os
from dotenv import load_dotenv
import rag_engine

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Contextual RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism and modern layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom main app styling */
    .stApp {
        background: radial-gradient(circle at top right, #1E1B4B, #09090B);
    }
    
    /* Header card */
    .header-container {
        background: rgba(30, 27, 75, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2.5rem;
        border-radius: 1.25rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #A5B4FC, #6366F1, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0C0A1F !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    /* Status indicator */
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .status-ok {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-missing {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* File uploader hover animation */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        background-color: rgba(15, 23, 42, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366F1;
        background-color: rgba(15, 23, 42, 0.5);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
    }
    
    /* Metric Card styling */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #818CF8;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Chat bubbles custom tweaks */
    .stChatMessage {
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.25);
    }
    
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: rgba(15, 23, 42, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Expander styling */
    .stExpander {
        background-color: rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "doc_metrics" not in st.session_state:
    st.session_state.doc_metrics = {"files": 0, "pages": 0, "chunks": 0}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("System Controls")
    st.markdown("---")
    
    # API Key Configuration
    st.subheader("🔑 Credentials")
    
    # Try fetching default from .env
    default_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=default_key,
        help="Input your Google AI Studio API key. Get one from: https://aistudio.google.com/"
    )
    
    # Visual validation indicator
    if api_key_input:
        st.markdown(
            '<div class="api-status status-ok">✓ Key Loaded</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="api-status status-missing">✗ Key Missing</div>', 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # Model Configurations
    st.subheader("⚙️ LLM Configuration")
    model_choice = st.selectbox(
        "Embedding & Reasoner",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Gemini 1.5 Flash is recommended for fast response times. Gemini 1.5 Pro is best for complex queries."
    )
    
    st.subheader("📏 Chunk settings")
    chunk_size = st.slider(
        "Chunk Size (characters)", 
        min_value=300, 
        max_value=2000, 
        value=1000, 
        step=100
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)", 
        min_value=50, 
        max_value=500, 
        value=200, 
        step=50
    )
    
    st.markdown("---")
    
    # Reset Controls
    st.subheader("🧹 System Clean")
    if st.button("Clear Memory & Database", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.processed_files = set()
        st.session_state.doc_metrics = {"files": 0, "pages": 0, "chunks": 0}
        st.session_state.chat_history = []
        st.success("Session cleared successfully!")
        st.rerun()

# --- MAIN APP LAYOUT ---
# Header
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">Cognitive PDF Reader</h1>
        <div class="header-subtitle">Analyze, search, and question PDF documents instantly using highly optimized local context search and Gemini AI.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Workspace Layout
col_left, col_right = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: File ingestion & indexing ---
with col_left:
    st.header("📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type="pdf", 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Processing block
    if uploaded_files:
        # Filter files that haven't been processed yet
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if not api_key_input:
                st.error("⚠️ Please provide a valid Gemini API Key in the sidebar to process files.")
            else:
                with st.spinner("Analyzing document structure & indexing embeddings..."):
                    try:
                        # Instantiate VectorStore if not exists
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = rag_engine.VectorStore()
                            
                        total_pages = 0
                        all_chunks = []
                        successfully_read = []
                        
                        for file in new_files:
                            try:
                                file_bytes = file.read()
                            except Exception as exc:
                                logger.error("Failed to read uploaded file '%s': %s", file.name, exc)
                                st.warning(f"Could not read '{file.name}', skipping.")
                                continue

                            try:
                                pages = rag_engine.extract_text_from_pdf(file_bytes, file.name)
                            except ValueError as exc:
                                logger.error("PDF extraction failed for '%s': %s", file.name, exc)
                                st.warning(f"'{file.name}' could not be parsed: {exc}")
                                continue

                            total_pages += len(pages)

                            chunks = rag_engine.chunk_text(
                                pages,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            all_chunks.extend(chunks)
                            successfully_read.append(file.name)

                        if not all_chunks:
                            st.warning("No text could be extracted from the uploaded files.")
                        else:
                            # Add embeddings — only mark files as processed after success
                            st.session_state.vector_store.add_chunks(all_chunks, api_key_input)

                            for name in successfully_read:
                                st.session_state.processed_files.add(name)

                            st.session_state.doc_metrics["files"] += len(successfully_read)
                            st.session_state.doc_metrics["pages"] += total_pages
                            st.session_state.doc_metrics["chunks"] += len(all_chunks)

                            st.success(f"Successfully processed {len(successfully_read)} new file(s)!")

                    except Exception as e:
                        logger.exception("Unexpected error during file processing")
                        st.error(f"Failed to process files: {e}")
                        
        # Display Database metrics
        st.markdown("### 📊 Index Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{st.session_state.doc_metrics["files"]}</div><div class="metric-label">Files</div></div>',
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{st.session_state.doc_metrics["pages"]}</div><div class="metric-label">Pages</div></div>',
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{st.session_state.doc_metrics["chunks"]}</div><div class="metric-label">Chunks</div></div>',
                unsafe_allow_html=True
            )
            
        # List indexed files
        st.markdown("### 📎 Indexed Documents")
        for f in st.session_state.processed_files:
            st.markdown(f"- `{f}`")

# --- RIGHT COLUMN: Chat QA ---
with col_right:
    st.header("💬 Document QA Channel")
    
    # Check if vector store is initialized
    if not st.session_state.processed_files:
        st.info("💡 Upload some PDF documents in the left panel to begin your RAG chat session.")
    else:
        # Display Chat History
        chat_container = st.container(height=500)
        
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    # If assistant and has source context, display in expander
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("🔍 Verified Retreived Context & Reference Chunks"):
                            for i, src in enumerate(message["sources"]):
                                chunk = src["chunk"]
                                st.markdown(
                                    f"**Source #{i+1}**: `{chunk['source']}` | Page {chunk['page_num']} | Score: `{src['score']:.4f}`"
                                )
                                st.code(chunk["text"], language="text")
        
        # User input
        if query := st.chat_input("Ask a question about your documents..."):
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(query)
            
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Query the RAG engine
            if not api_key_input:
                st.error("Please add your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Retrieving document contexts and generating answer..."):
                    try:
                        results = st.session_state.vector_store.search(
                            query,
                            api_key_input,
                            k=5
                        )
                    except Exception as e:
                        logger.exception("Document search failed")
                        st.error(f"Document search failed: {e}")
                        results = None

                    if results is not None:
                        try:
                            answer = rag_engine.generate_answer(
                                query,
                                results,
                                api_key_input,
                                model_name=model_choice
                            )
                        except Exception as e:
                            logger.exception("Answer generation failed")
                            st.error(f"Answer generation failed: {e}")
                            answer = None

                        if answer is not None:
                            with chat_container:
                                with st.chat_message("assistant"):
                                    st.write(answer)
                                    with st.expander("🔍 Verified Retreived Context & Reference Chunks"):
                                        for i, src in enumerate(results):
                                            chunk = src["chunk"]
                                            st.markdown(
                                                f"**Source #{i+1}**: `{chunk['source']}` | Page {chunk['page_num']} | Score: `{src['score']:.4f}`"
                                            )
                                            st.code(chunk["text"], language="text")

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": results
                            })

                            st.rerun()
