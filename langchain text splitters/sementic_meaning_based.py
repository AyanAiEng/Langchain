from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)


text = """
Artificial intelligence is transforming the world.

Machine learning models learn patterns from data.

Python is a popular programming language.

It is widely used in data science and AI.
"""

docs = text_splitter.create_documents([text])

print(docs)