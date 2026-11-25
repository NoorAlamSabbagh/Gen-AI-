from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=5)

llm = ChatOllama(model="llama3.2", base_url="http://127.0.0.1:11434")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | llm | StrOutputParser()

print("Searching for latest AI news...")
news_result = search_tool.run("Latest AI news of 2026 ")

print("Summarizing with Ollama...\n")
for chunk in chain.stream({"news": news_result}):
    print(chunk, end="", flush=True)
print("\n")


print(search_tool.description)
print(search_tool.name)
print(search_tool.args)
