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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Connect to local Ollama (100% free, runs offline on your PC)
model = ChatOpenAI(
    model="llama3.2",                      # Or the model you downloaded (e.g. "qwen2.5:3b")
    base_url="http://localhost:11434/v1",  # Local Ollama endpoint
    api_key="ollama"                       # Dummy string (Ollama doesn't require a paid key)
)

result = model.invoke("Explain RAG in simple terms.")
print(result.content)
