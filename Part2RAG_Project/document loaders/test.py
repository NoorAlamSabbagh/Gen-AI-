# from pathlib import Path
# from langchain_community.document_loaders import TextLoader

# file_path = Path(__file__).parent / "notes.txt"
# loader = TextLoader(file_path, encoding="utf-8")
# docs = loader.load()
# # print(docs)
# print(docs[0])
# print(len(docs))


# //
# from pathlib import Path
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI

# # 1. Load document using absolute path
# file_path = Path(__file__).parent / "notes.txt"
# loader = TextLoader(file_path, encoding="utf-8")
# docs = loader.load()

# print(f"Loaded {len(docs)} document(s)")
# print("Content:", docs[0].page_content)
# print("-" * 50)

# # 2. Initialize local Ollama model
# model = ChatOpenAI(
#     model="llama3.2",
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )

# # 3. Create prompt template
# template = ChatPromptTemplate.from_messages([
#     ("system", "hIII You are a helpful assistant. Summarize or respond to the user's notes."),
#     ("user", "{data}"),
# ])

# prompt = template.format_messages(data=docs[0].page_content)
# result = model.invoke(prompt)
# print(result.content)




# //
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap=1
)

file_path = Path(__file__).parent / "notes.txt"
data = TextLoader(file_path, encoding="utf-8")

docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

