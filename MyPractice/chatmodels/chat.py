# (1)
# from dotenv import load_dotenv

# load_dotenv()

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gpt-4.1")
# print(model)



# //(2)
# from dotenv import load_dotenv

# load_dotenv()

# from langchain.chat_models import init_chat_model

# # model = init_chat_model("gpt-4.1")
# model = init_chat_model("gpt-5")
# response = model.invoke("what is cricket ?")
# # print(response)
# print(response.content)


#(3)
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")
# response = model.invoke("what is cricket ?")
# print(response.content)


# //(4)
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")
# response = model.invoke("what is cricket ?")
# print(response.content)

# //(5)
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq

# load_dotenv()

# model = ChatGroq(model="openai/gpt-oss-120b")
# response = model.invoke("Give me a paragraph on Machine Learning.")
# print(response.content)



# //
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506",temperature=0.9)

response = model.invoke("write a poem on AI")

print(response.content)