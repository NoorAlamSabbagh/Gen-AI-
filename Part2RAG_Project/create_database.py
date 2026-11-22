# Load PDF 
# Split into chunks 
# Create embeddings 
# Store into Chroma 
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

# Base directory for Part2RAG_Project
base_dir = Path(__file__).resolve().parent

# Load .env if present
load_dotenv(base_dir / ".env")

# 1. Load PDF from document loaders
pdf_path = base_dir / "document loaders" / "deeplearning.pdf"
print(f"Loading {pdf_path.name}...")
data = PyPDFLoader(str(pdf_path))
docs = data.load()
print(f"Loaded {len(docs)} pages.")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# 3. Embedding Model (free local via Ollama)
embedding_model = OpenAIEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    check_embedding_ctx_length=False,  # Ollama requires raw text strings, not token arrays
    chunk_size=64                      # Process in safe batches for local GPU/CPU
)

# 4. Store into Chroma inside Part2RAG_Project
chroma_dir = base_dir / "chroma_db"
print(f"Storing vector database in: {chroma_dir}")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(chroma_dir)
)
print("Chroma DB successfully created!")
