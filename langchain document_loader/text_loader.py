from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
) 

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)

print(len(docs))

print(docs[0].page_content)

print("*************************")
print("*************************")
print("*************************")
print("*************************")

print(docs[0].metadata)

chain = prompt| model | parser

print(chain.invoke({"poem":docs[0].page_content}))
