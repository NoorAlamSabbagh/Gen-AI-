from pathlib import Path
from langchain_community.document_loaders import TextLoader

file_path = Path(__file__).parent / "notes.txt"
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()
# print(docs)
print(docs[0])
print(len(docs))



# from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(
#     separator= "",
#     chunk_size = 1000,
#     chunk_overlap=1
# )

# data = TextLoader("document loaders/notes.txt")

# docs = data.load()

# chunks = splitter.split_documents(docs)

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()
#     print()

