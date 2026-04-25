from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel , Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model1 = ChatHuggingFace(llm = llm)
model2 = ChatHuggingFace(llm = llm)

class sentiment(BaseModel):
    sentiment:Literal["positive","negative"] = Field(description='Give the sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=sentiment)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | model1 | parser
positive_chain = prompt2 | model1 
negative_chain = prompt3 | model1  

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "negative",negative_chain),
    (lambda x:x.sentiment == "positive",positive_chain),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback":"This is a sexy phone"}))

chain.get_graph().print_ascii()


