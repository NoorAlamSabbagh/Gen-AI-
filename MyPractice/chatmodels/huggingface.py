# import os
# from pathlib import Path

# from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if not api_token:
#     raise RuntimeError(
#         "HUGGINGFACEHUB_API_TOKEN is missing. Add your Hugging Face token to MyPractice/.env."
#     )

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     huggingfacehub_api_token=api_token,
# )

# model = ChatHuggingFace(llm=llm)
# response = model.invoke("who are you ? ")
# print(response.content)


# //2
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 128, "do_sample": False},
)

model = ChatHuggingFace(llm=llm)
response = model.invoke("Who are you?")
print(response.content)


# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1"
# )

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("who are you ? ")

# print(response.content)