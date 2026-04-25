from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)



template1 = PromptTemplate(
    template="wrie a detailed report on {topic}",
    input_variables=["topicu"]
)


template2 = PromptTemplate(
    template="write a 5 lines summary on the following. \n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke(
    {'topic':'black hole'}
)

result1 = model.invoke(prompt1)

print("the detailed report")

print(result1.content)


prompt2 = template2.invoke(
    {'text':result1.content}
)
print("the summary")
result2 = model.invoke(prompt2)

print(result2.content)