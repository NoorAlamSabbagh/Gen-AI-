from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    dimensions=64
)

texts = [
    "Hello this is Noor Alam",
    "Hello your name is YouTube",
    "And you all are very beautifull"
]

vector = embeddings.embed_documents(texts)

print(vector)