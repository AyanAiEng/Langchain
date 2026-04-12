from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create chat prompt template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name='chat_history'),  # <-- comma added
    ("human", "{query}")  # <-- use variable placeholder for user query
])

# Load previous chat history from file
chat_history = []
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())  # <-- added parentheses

# Example user query
user_query = "Where is my refund?"

# Invoke the template with chat history and query
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': user_query
})

print(prompt)