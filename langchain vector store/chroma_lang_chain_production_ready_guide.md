# 🚀 Chroma + LangChain Production-Ready Guide (2026)

This file contains **industry-used essential functions** for working with entity["software","Chroma","vector database"] using entity["software","LangChain","LLM application framework"].

It includes:
- Setup
- Embeddings
- Core CRUD operations
- Search functions
- Advanced utilities
- Production-ready pattern

---

# 📦 1. Installation
```bash
pip install -U langchain langchain-chroma langchain-huggingface chromadb
```

---

# 🧠 2. Setup (Best Practice)
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Embeddings model (fast + reliable)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB setup
vector_store = Chroma(
    collection_name="production_db",
    embedding_function=embeddings,
    persist_directory="chroma_storage"
)
```

---

# 📥 3. ADD DATA (Most Important)

## ✔ add_documents (recommended)
```python
docs = [
    Document(page_content="LangChain is used for LLM apps", metadata={"id": 1}),
    Document(page_content="Chroma stores embeddings efficiently", metadata={"id": 2})
]

vector_store.add_documents(docs)
```

## ✔ add_texts (quick method)
```python
vector_store.add_texts([
    "AI is transforming software",
    "Vector databases store embeddings"
])
```

---

# 🔍 4. SEARCH / RETRIEVAL (Core Feature)

## ✔ similarity_search
```python
results = vector_store.similarity_search("What is Chroma?")

for r in results:
    print(r.page_content)
```

---

## ✔ similarity_search_with_score (important in production)
```python
results = vector_store.similarity_search_with_score("LangChain usage")

for doc, score in results:
    print(doc.page_content, score)
```

👉 Lower score = better match

---

# 🆔 5. GET BY IDS (Advanced)
```python
vector_store.get(ids=["id1", "id2"])
```

✔ Used when:
- You store custom IDs
- You want exact retrieval

---

# ❌ 6. DELETE DATA (Critical in production)
```python
vector_store.delete(ids=["id1"])
```

✔ Use cases:
- Removing outdated documents
- GDPR / compliance cleanup

---

# 🔄 7. UPDATE DATA (Workaround pattern)
Chroma does NOT fully “update”, so you do:

```python
vector_store.delete(ids=["id1"])

vector_store.add_documents([
    Document(page_content="Updated content", metadata={"id": "id1"})
])
```

---

# 📊 8. COLLECTION INFO (Monitoring)

## Count documents
```python
vector_store._collection.count()
```

## Peek data (debugging)
```python
vector_store.get()
```

---

# 💾 9. PERSISTENCE (VERY IMPORTANT)
```python
vector_store.persist()
```

⚠️ Note:
- In latest versions, persistence is handled automatically if `persist_directory` is set

---

# ⚡ 10. INDUSTRY BEST PRACTICE PIPELINE

```python
# 1. Create documents
from langchain_core.documents import Document

docs = [Document(page_content="AI is powerful")]

# 2. Add to vector DB
vector_store.add_documents(docs)

# 3. Query
results = vector_store.similarity_search("AI")

# 4. Use results in LLM (RAG)
for r in results:
    print(r.page_content)
```

---

# 🚀 11. PRODUCTION-READY TEMPLATE

```python
class VectorDBService:
    def __init__(self, vector_store):
        self.vs = vector_store

    def add_docs(self, texts):
        docs = [Document(page_content=t) for t in texts]
        self.vs.add_documents(docs)

    def search(self, query, k=5):
        return self.vs.similarity_search(query, k=k)

    def search_with_score(self, query, k=5):
        return self.vs.similarity_search_with_score(query, k=k)

    def delete_docs(self, ids):
        self.vs.delete(ids=ids)

    def get_all(self):
        return self.vs.get()
```

---

# 🧠 SUMMARY (WHAT INDUSTRY USES MOST)

### ⭐ TOP 5 MOST USED FUNCTIONS:
1. `add_documents()`
2. `add_texts()`
3. `similarity_search()`
4. `similarity_search_with_score()`
5. `delete()`

---

# 🔥 FINAL MINDSET

👉 Chroma = memory storage
👉 Embeddings = understanding text
👉 Search = semantic intelligence

Together = RAG systems (modern AI apps)

