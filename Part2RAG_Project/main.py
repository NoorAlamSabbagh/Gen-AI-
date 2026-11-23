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
# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # 1. Path to deeplearning.pdf (anchored to this script's directory)
# pdf_path = Path(__file__).parent / "document loaders" / "deeplearning.pdf"

# print(f"Loading {pdf_path.name}...")
# data = PyPDFLoader(str(pdf_path))
# docs = data.load()
# print(f"Total pages loaded: {len(docs)}")

# # 2. Split the document into chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100  # Recommended overlap for better context
# )

# chunks = splitter.split_documents(docs)
# print(f"Total chunks created: {len(chunks)}")

# # 3. Print the first chunk
# print("\n--- First Chunk Content ---")
# print(chunks[0].page_content)


# //
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")

# Embedding model must match create_database.py
embedding_model = OpenAIEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    check_embedding_ctx_length=False
)

# Open the persisted Chroma database
chroma_dir = base_dir / "chroma_db"
vectorstore = Chroma(
    persist_directory=str(chroma_dir),
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    