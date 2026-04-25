from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model1 = ChatHuggingFace(llm = llm)

model2 = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes":prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser 
})

merged_chain = prompt3 | model1 | parser
chain = parallel_chain | merged_chain

with open("text.txt","r") as f:
    text  = f.read()

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()
