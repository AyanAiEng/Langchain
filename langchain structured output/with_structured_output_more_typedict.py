from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict,Annotated,Literal,Optional
import os

load_dotenv()

class Review(TypedDict):
    name: Annotated[list[str],"Write all the key themes discussed in the review in a list"]
    summary:Annotated[str,"A brie summary of review"]
    sentiment:Annotated[Literal["pos","neg"],"Return sentiment of the review either negative,positive or neutral"]
    pros:Annotated[Optional[list[str]],"Write down all the prons inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside a list"]
    namee :Annotated[Optional[str],"Write the name of reviewer"]    

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm = llm)

structured_model = model.with_structured_output(Review)


message = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
I

"""

result = structured_model.invoke(message)

print(result)
print(result["name"])