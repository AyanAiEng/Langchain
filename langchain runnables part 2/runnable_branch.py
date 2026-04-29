from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableBranch,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# the branch chain work on if else logic

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen = prompt1 | model | parser

""" it is basically working as if else if condition in python that if the len of the text that is taken from the report gem is greateer then 500 so summaries it and if you wantto learn about the runnabe RunnablePassthrough so you can check he RunnablePassthrough.py it will help you
 """

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500, prompt2 |model| parser),
    RunnablePassthrough()
)

final_chain = report_gen | branch_chain

print(final_chain.invoke({"topic":"Israel vs Iran"}))