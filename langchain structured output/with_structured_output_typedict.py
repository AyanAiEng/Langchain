from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

class hello(TypedDict):
    summary:str
    sentiment:str


model = ChatHuggingFace(
    llm = llm
)

structured_model = model.with_structured_output(hello)

message = """
The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""

result = structured_model.invoke(message)

print(result)
print(result["summary"])
print(result["sentiment"])

