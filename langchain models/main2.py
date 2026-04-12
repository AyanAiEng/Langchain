from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Create HuggingFace endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)

# Wrap inside ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Invoke
result = model.invoke("Explain Artificial Intelligence in simple words.")

print(result.content)