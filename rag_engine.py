import logging
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
import io

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file_bytes, filename: str) -> list[dict]:
    """
    Extracts text page-by-page from an uploaded PDF.
    Returns a list of dicts containing page content and metadata.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_file_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to read '{filename}': the file may be corrupted or not a valid PDF") from exc

    pages_data = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            logger.warning("Could not extract text from page %d of '%s', skipping", i + 1, filename)
            continue
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
        if not api_key:
            raise ValueError("API key is required to generate embeddings")

        genai.configure(api_key=api_key)
        texts = [c["text"] for c in chunks]

        batch_size = 100
        new_embeddings = []
        new_chunks = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch_texts,
                    task_type="retrieval_document"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Embedding API request failed on batch starting at index {i}"
                ) from exc

            embeddings_data = response.get("embedding")
            if embeddings_data is None:
                raise RuntimeError(
                    f"Embedding API returned an unexpected response (missing 'embedding' key) "
                    f"on batch starting at index {i}"
                )

            for j, emb in enumerate(embeddings_data):
                new_embeddings.append(np.array(emb, dtype=np.float32))
                new_chunks.append(chunks[i + j])

        # Only commit to state after all batches succeed
        self.embeddings.extend(new_embeddings)
        self.chunks.extend(new_chunks)
                
    def search(self, query: str, api_key: str, k: int = 5) -> list[dict]:
        """
        Embeds the query and performs cosine similarity search.
        """
        if not self.embeddings:
            return []
        if not api_key:
            raise ValueError("API key is required to search")

        genai.configure(api_key=api_key)
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
        except Exception as exc:
            raise RuntimeError("Failed to generate embedding for the query") from exc

        embedding_data = response.get("embedding")
        if embedding_data is None:
            raise RuntimeError(
                "Embedding API returned an unexpected response (missing 'embedding' key)"
            )
        query_emb = np.array(embedding_data, dtype=np.float32)
        
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
    genai.configure(api_key=api_key)
    
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
    
    if not api_key:
        raise ValueError("API key is required to generate an answer")

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        raise RuntimeError(
            f"Gemini content generation failed (model={model_name})"
        ) from exc

    # response.text raises ValueError when content is blocked by safety filters
    try:
        return response.text
    except ValueError:
        # Provide feedback from the safety ratings when the response is blocked
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"The response was blocked by safety filters. Feedback: {feedback}"
        )
