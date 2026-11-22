# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader

# file_path = Path(__file__).parent / "GRU.pdf"
# data = PyPDFLoader(str(file_path))
# docs = data.load()
# # print(docs[0])
# print(docs[14])
# # print(len(docs))

# //
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)