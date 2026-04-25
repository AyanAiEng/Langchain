# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# from dotenv import load_dotenv
# import os

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
# )
# model= ChatHuggingFace(llm  = llm)

# chat_history = []

# while True:
#     user_input = str(input("You: "))
#     chat_history.append(user_input)
#     if user_input == "exit":
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(result.content)
#     print("AI: ",result.content)
# print(chat_history)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)
p