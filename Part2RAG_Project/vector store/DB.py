from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. Documents
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

# 2. Free local embedding model via Ollama
embedding_model = OpenAIEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    check_embedding_ctx_length=False
)

# 3. Target directory: Part2RAG_Project / chroma_db
chroma_dir = Path(__file__).resolve().parent.parent / "chroma_db"

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=str(chroma_dir)
)

# 4. Similarity Search
print("--- Similarity Search Results ---")
result = vectorstore.similarity_search("what is used for data analysis?", k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)

# 5. Retriever
print("\n--- Retriever Results ---")
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.invoke("Explain deep learning")
for d in retrieved_docs:
    print(d.page_content)