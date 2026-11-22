# from pathlib import Path
# from dotenv import load_dotenv 
# from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import HumanMessage, SystemMessage

# # Load .env file from the project directory
# env_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=env_path)  


# model = ChatMistralAI(model = "mistral-small-2506")
# result = model.invoke("Hello")

# print(result.content)  


# ////
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# # Connect to local Ollama (100% free, runs offline on your PC)
# model = ChatOpenAI(
#     model="llama3.2",                      # Or the model you downloaded (e.g. "qwen2.5:3b")
#     base_url="http://localhost:11434/v1",  # Local Ollama endpoint
#     api_key="ollama"                       # Dummy string (Ollama doesn't require a paid key)
# )

# result = model.invoke("Explain RAG in simple terms.")
# print(result.content)


# //
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Path to deeplearning.pdf (anchored to this script's directory)
pdf_path = Path(__file__).parent / "document loaders" / "deeplearning.pdf"

print(f"Loading {pdf_path.name}...")
data = PyPDFLoader(str(pdf_path))
docs = data.load()
print(f"Total pages loaded: {len(docs)}")

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100  # Recommended overlap for better context
)

chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# 3. Print the first chunk
print("\n--- First Chunk Content ---")
print(chunks[0].page_content)
