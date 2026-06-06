import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
import io


def _configure_and_embed(api_key: str, content, task_type: str) -> list:
    """Configure the Gemini API and return embeddings for the given content."""
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=content,
        task_type=task_type,
    )
    return response["embedding"]

def extract_text_from_pdf(pdf_file_bytes, filename: str) -> list[dict]:
    """
    Extracts text page-by-page from an uploaded PDF.
    Returns a list of dicts containing page content and metadata.
    """
    reader = PdfReader(io.BytesIO(pdf_file_bytes))
    pages_data = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "text": text.strip(),
                "page_num": i + 1,
                "source": filename
            })
    return pages_data

def chunk_text(pages_data: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits page-by-page text into overlapping chunks, maintaining metadata.
    """
    chunks = []
    for page in pages_data:
        text = page["text"]
        page_num = page["page_num"]
        source = page["source"]
        
        # Simple character-based splitting with overlap
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Adjust end to avoid splitting words in half if possible
            if end < len(text):
                # Look for last space within the next 20 characters
                space_idx = text.rfind(' ', start, end)
                if space_idx != -1 and space_idx > start:
                    end = space_idx
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "text": chunk_content,
                    "page_num": page_num,
                    "source": source
                })
            
            start = end - chunk_overlap
            # If chunk overlap makes start move backward or get stuck, break or force forward
            if start >= end:
                start = end
            if start < 0:
                start = 0
            # If we reached the end of the text, break
            if end >= len(text):
                break
                
    return chunks

class VectorStore:
    """
    Lightweight, in-memory vector index that uses NumPy for cosine similarity.
    Does not require external services or C++ binary dependencies (FAISS, Chroma).
    """
    def __init__(self):
        self.embeddings = []
        self.chunks = []
        
    def add_chunks(self, chunks: list[dict], api_key: str):
        """
        Generates embeddings for chunks in batches and stores them.
        """
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        
        # Batch requests to avoid API payload limit and optimize performance
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = _configure_and_embed(api_key, batch_texts, "retrieval_document")
            
            for j, emb in enumerate(batch_embeddings):
                self.embeddings.append(np.array(emb, dtype=np.float32))
                self.chunks.append(chunks[i + j])
                
    def search(self, query: str, api_key: str, k: int = 5) -> list[dict]:
        """
        Embeds the query and performs cosine similarity search.
        """
        if not self.embeddings:
            return []
            
        raw_embedding = _configure_and_embed(api_key, query, "retrieval_query")
        query_emb = np.array(raw_embedding, dtype=np.float32)
        
        similarities = []
        for emb in self.embeddings:
            norm_prod = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            if norm_prod == 0:
                sim = 0.0
            else:
                sim = np.dot(query_emb, emb) / norm_prod
            similarities.append(sim)
            
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results

def generate_answer(query: str, search_results: list[dict], api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Generates an answer using the retrieved context from search_results.
    """
    # Format context
    context_parts = []
    for i, res in enumerate(search_results):
        chunk = res["chunk"]
        context_parts.append(
            f"Document: {chunk['source']}\n"
            f"Page: {chunk['page_num']}\n"
            f"Content: {chunk['text']}\n"
            f"Similarity Score: {res['score']:.4f}\n"
            f"--------------------------------------------------"
        )
    context = "\n".join(context_parts)
    
    system_instruction = (
        "You are an expert Q&A AI assistant designed to answer questions based solely on the provided document contexts.\n"
        "Instructions:\n"
        "1. Provide a comprehensive, accurate, and detailed answer using ONLY the context given below.\n"
        "2. If the context does not contain the answer, say: 'Based on the uploaded documents, I couldn't find the answer to your question.' Do not attempt to make up or hallucinate any facts.\n"
        "3. Cite your sources directly in the answer using footnotes or inline markers [DocumentName, Page X] to point to where the information is located.\n"
        "4. Remain objective, clear, and professional."
    )
    
    prompt = (
        f"CONTEXT DETAILS:\n"
        f"{context}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"DETAILED ANSWER:"
    )
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    
    response = model.generate_content(prompt)
    return response.text
