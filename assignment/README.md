#  RAG Chatbot using LangChain, SentenceTransformers, and FAISS

A Retrieval-Augmented Generation (RAG) chatbot that extracts knowledge from a PDF document and answers queries using semantic search + an LLM (e.g., LLaMA 3). The project is built using Streamlit and runs locally in your browser.


# Project Architecture and Flow

PDF → Extract Text → Chunk Text → Generate Embeddings → Store in FAISS → Query → Retrieve Top Chunks → LLM → Response

Step-by-Step Flow -> 

PDF Upload: The user uploads a PDF (e.g., AI Training Document.pdf).
Text Extraction: The text is extracted from the PDF using PyPDF2.
Chunking: Text is split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
Embeddings: Each chunk is embedded into a vector using SentenceTransformer (e.g., all-MiniLM-L6-v2).
FAISS Indexing: All embeddings are stored in a FAISS vector database.
User Query: A user inputs a question into the chatbot.
Retrieval: FAISS retrieves the most relevant chunk(s).
LLM Response: A local or cloud-based LLM (like llama3) generates a response using the retrieved context.
Streaming Output: The answer is streamed live into the chatbot UI using Streamlit.


Model and Embedding Choices


Embedding Model	-> all-MiniLM-L6-v2	-> Fast, accurate 384-dim vectors, great for semantic similarity

Vector DB->	FAISS->	Fast similarity search on millions of vectors
Chunking ->	RecursiveCharacterTextSplitter ->	Retains sentence boundaries and context overlap
LLM	llama3 -> (via Ollama or cloud API)->	    Open-source, accurate, compatible with local setup
UI Framework ->	Streamlit      ->             	Quick web UI with file upload and streaming support

Run the Chatbot with Streaming Enabled
Open a terminal and run: -> streamlit run app.py
Upload your PDF via the web UI.

Ask questions in the input box.

Get real-time answers streamed from the LLM!