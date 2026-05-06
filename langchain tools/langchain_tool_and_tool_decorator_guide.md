
# 🔧 LangChain @tool & Tools Explained (Basics → Advanced)

## 🧠 What is a Tool in LangChain?

A **tool** is a function that is converted into an AI-usable component so that an LLM (like GPT) can:
- Understand it
- Decide when to use it
- Pass structured input
- Get structured output

---

## ❓ What is @tool?

👉 `@tool` is a Python decorator used in LangChain.

It converts a normal Python function into a **LangChain Tool**.

Example:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str):
    """Get weather of a city"""
    return f"Weather in {city}"
```

---

## 🧠 Why do we use @tool?

Without @tool:
- Function is only Python code
- LLM cannot understand structure

With @tool:
- Adds metadata (name, description)
- Adds input schema
- Makes it callable by AI agents

---

## ⚡ Does tool just wrap a function?

YES — but more advanced:

A tool:
- Wraps function
- Adds schema
- Adds description
- Makes it AI-readable
- Enables agent calling

So:

👉 Tool = Function + Metadata + AI interface

---

## 🔥 What exactly does a tool do?

A tool enables:
- LLM decides WHEN to call function
- Automatically passes arguments
- Executes function
- Returns result to LLM

Flow:
User → LLM → Tool selection → Tool execution → Result

---

## 🚀 Advanced Understanding

Internally a tool becomes:

Tool(
  name="get_weather",
  description="Get weather of a city",
  args_schema={"city": "string"},
  func=your_function
)

---

## 🧠 Key Idea

👉 Tool = Bridge between Python and AI reasoning

---

## ⚡ One-line summary

A LangChain tool is a decorated function that turns normal Python code into an AI-callable action.
