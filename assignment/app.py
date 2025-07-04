import streamlit as st
import PyPDF2
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import os

# --- Configuration ---
PDF_PATH = "AI Training Document.pdf"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 50
TOP_K_CHUNKS = 1
LLM_MODEL_NAME = "llama3"
USE_OLLAMA_LLM = False

class DocumentProcessor:
    def __init__(self, chunk_size_words: int, chunk_overlap_words: int):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.chunk_size_chars = self.chunk_size_words * 6
        self.chunk_overlap_chars = self.chunk_overlap_words * 6

    def extract_text_from_pdf(self, pdf_path: str) -> str | None:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted_page_text = page.extract_text()
                    if extracted_page_text:
                        text += extracted_page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path}: {e}")
            return None

        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size_chars,
            chunk_overlap=self.chunk_overlap_chars,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        return text_splitter.split_text(text)

    def process_document(self, pdf_path: str) -> tuple[list[str], str | None]:
        raw_text = self.extract_text_from_pdf(pdf_path)
        if raw_text:
            chunks = self.chunk_text(raw_text)
            return chunks, raw_text
        return [], None

class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @st.cache_resource
    def load_model(_self):
        if _self.model is None:
            _self.model = SentenceTransformer(_self.model_name)
        return _self.model

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        self.model = self.load_model()
        return self.model.encode(texts, show_progress_bar=True)

class VectorDBManager:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadatas = []

    def build_index(self, embeddings: np.ndarray, texts: list[str], metadatas: list[dict]):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.texts = texts
        self.metadatas = metadatas

    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple[list[str], list[dict], list[float]]:
        if self.index is None:
            return [], [], []
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        texts = [self.texts[i] for i in indices[0]]
        metadatas = [self.metadatas[i] for i in indices[0]]
        return texts, metadatas, distances[0].tolist()

    def get_num_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

class LLMInterface:
    def __init__(self, model_name: str, use_ollama: bool):
        self.model_name = model_name
        self.use_ollama = use_ollama
        if self.use_ollama:
            try:
                from ollama import Client
                self.ollama_client = Client(host='http://localhost:11434')
            except:
                self.use_ollama = False

    def get_streaming_response(self, prompt: str):
        if self.use_ollama:
            try:
                response_stream = self.ollama_client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True
                )
                for chunk in response_stream:
                    yield chunk['message']['content']
            except:
                yield from self._simulate_response(prompt)
        else:
            yield from self._simulate_response(prompt)

    def _simulate_response(self, prompt: str):
        simulated = "I am a helpful AI assistant. I cannot find this information in the provided document."
        for word in simulated.split(" "):
            yield word + " "
            time.sleep(0.05)

class RAGPipeline:
    def __init__(self, pdf_path, embedding_model_name, chunk_size_words, chunk_overlap_words, top_k_chunks, llm_model_name, use_ollama_llm):
        self.pdf_path = pdf_path
        self.top_k_chunks = top_k_chunks
        self.doc_processor = DocumentProcessor(chunk_size_words, chunk_overlap_words)
        self.embed_generator = EmbeddingGenerator(embedding_model_name)
        self.vector_db = VectorDBManager()
        self.llm_interface = LLMInterface(llm_model_name, use_ollama_llm)
        self.document_chunks = []
        self.document_metadatas = []

    def process_and_index_document(self) -> bool:
        chunks, raw_text = self.doc_processor.process_document(self.pdf_path)
        if not chunks:
            return False
        self.document_chunks = chunks
        self.document_metadatas = [{"source": self.pdf_path, "chunk_id": i} for i in range(len(chunks))]
        embeddings = self.embed_generator.generate_embeddings(chunks)
        self.vector_db.build_index(embeddings, chunks, self.document_metadatas)
        return True

    def query_pipeline(self, user_query: str):
        if not self.vector_db.index:
            yield "Error: Document not processed."
            return
        query_embedding = self.embed_generator.generate_embeddings([user_query])
        chunks, metas, distances = self.vector_db.search(query_embedding, k=self.top_k_chunks)
        if not chunks or distances[0] > 1.0:
            yield "I am a bot, and I don't have enough information in the provided document to answer that question."
            return

        # Custom formatted response
        yield "**Here's a concise summary based on the document:**\n"
        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?]) +', chunk.strip())
            for s in sentences:
                yield f"\n- {s.strip()}"

    def get_indexed_chunk_count(self) -> int:
        return len(self.document_chunks)

@st.cache_resource
def get_rag_pipeline_instance():
    return RAGPipeline(
        pdf_path=PDF_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        chunk_size_words=CHUNK_SIZE_WORDS,
        chunk_overlap_words=CHUNK_OVERLAP_WORDS,
        top_k_chunks=TOP_K_CHUNKS,
        llm_model_name=LLM_MODEL_NAME,
        use_ollama_llm=USE_OLLAMA_LLM
    )

rag_pipeline = get_rag_pipeline_instance()

def main():
    st.set_page_config(page_title="Amlgo Labs RAG Chatbot", layout="wide")
    st.title("AI-Powered Document Chatbot")

    with st.sidebar:
        st.header("Chatbot Information")
        st.write(f"**LLM in use:** `{LLM_MODEL_NAME}` (Ollama: {USE_OLLAMA_LLM})")
        st.write(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
        st.write(f"**Indexed Chunks:** {rag_pipeline.get_indexed_chunk_count()}")
        st.write(f"**Chunk Size (words):** {CHUNK_SIZE_WORDS}")
        st.write(f"**Chunk Overlap (words):** {CHUNK_OVERLAP_WORDS}")
        st.write(f"**Top K Chunks for Retrieval:** {TOP_K_CHUNKS}")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if not st.session_state.get("document_processed", False):
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    if rag_pipeline.process_and_index_document():
                        st.session_state.document_processed = True
                        st.rerun()
                    else:
                        st.session_state.document_processed = False
                        st.error("Failed to process document.")
        else:
            st.success("Document already processed!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.document_processed and rag_pipeline.get_indexed_chunk_count() > 0:
        if prompt := st.chat_input("Ask me about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                full_response = ""
                placeholder = st.empty()
                for chunk in rag_pipeline.query_pipeline(prompt):
                    full_response += chunk + "\n"
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Please process the document first using the sidebar.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        st.error(f"Error: PDF file not found at '{PDF_PATH}'.")
        st.stop()
    main()
