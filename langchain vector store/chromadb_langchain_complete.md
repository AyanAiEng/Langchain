# ChromaDB × LangChain — Complete Production Reference

> **Stack:** Python 3.11+ · LangChain 0.3+ · ChromaDB 0.5+ · OpenAI / HuggingFace embeddings  
> All code is production-ready, typed, async-capable, and follows current best practices.

---

## Table of Contents

1. [Installation & Environment](#1-installation--environment)
2. [Client Setup — Local & Remote](#2-client-setup--local--remote)
3. [Embedding Models](#3-embedding-models)
4. [Collection Management](#4-collection-management)
5. [Adding Documents](#5-adding-documents)
6. [Querying & Retrieval](#6-querying--retrieval)
7. [Updating & Deleting Documents](#7-updating--deleting-documents)
8. [Metadata Filtering](#8-metadata-filtering)
9. [Retrievers (LangChain Integration)](#9-retrievers-langchain-integration)
10. [RAG Pipeline — Full Chain](#10-rag-pipeline--full-chain)
11. [MultiQueryRetriever](#11-multiqueryretriever)
12. [Contextual Compression Retriever](#12-contextual-compression-retriever)
13. [Self-Query Retriever](#13-self-query-retriever)
14. [Ensemble / Hybrid Retriever](#14-ensemble--hybrid-retriever)
15. [Document Loaders & Splitters](#15-document-loaders--splitters)
16. [Persistence & Backups](#16-persistence--backups)
17. [Async Operations](#17-async-operations)
18. [Production Utilities](#18-production-utilities)
19. [Error Handling & Retry Logic](#19-error-handling--retry-logic)
20. [Complete End-to-End Example](#20-complete-end-to-end-example)

---

## 1. Installation & Environment

```bash
# Core dependencies
pip install chromadb langchain langchain-community langchain-openai langchain-chroma

# Embedding options
pip install sentence-transformers          # local HuggingFace embeddings
pip install langchain-huggingface          # HF via LangChain wrapper

# Document loaders
pip install pypdf unstructured             # PDF / unstructured data
pip install beautifulsoup4 lxml            # web scraping loaders

# Production extras
pip install tenacity python-dotenv pydantic loguru
```

```python
# .env
OPENAI_API_KEY=sk-...
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_AUTH_TOKEN=your-token            # if using auth
```

---

## 2. Client Setup — Local & Remote

```python
import chromadb
from chromadb.config import Settings

# ── Local persistent client (most common for dev) ──────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")

# ── In-memory client (testing / CI) ────────────────────────────────────────
client = chromadb.EphemeralClient()

# ── Remote HTTP client (production server) ─────────────────────────────────
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials="your-token",
    ),
)

# ── Async HTTP client (FastAPI / async services) ────────────────────────────
async_client = await chromadb.AsyncHttpClient(host="localhost", port=8000)

# Health check
print(client.heartbeat())               # returns server timestamp
print(client.get_version())             # ChromaDB version string
```

---

## 3. Embedding Models

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import chromadb.utils.embedding_functions as ef

# ── OpenAI (best quality, paid) ────────────────────────────────────────────
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",     # 3072 dims — highest accuracy
    # model="text-embedding-3-small",   # 1536 dims — faster / cheaper
    dimensions=1536,                    # optional: reduce via Matryoshka
)

# ── HuggingFace local (free, privacy-safe) ─────────────────────────────────
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",   # top-ranked open model
    model_kwargs={"device": "cpu"},         # or "cuda" / "mps"
    encode_kwargs={"normalize_embeddings": True},
)

# ── Ollama (local LLM server) ───────────────────────────────────────────────
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ── Native ChromaDB embedding functions (used without LangChain wrapper) ───
chroma_oai_ef = ef.OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-large",
)
chroma_hf_ef = ef.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)
```

---

## 4. Collection Management

```python
from chromadb import Collection
from chromadb.api.models.Collection import Collection

# ── Create or get (idempotent — safe to call multiple times) ────────────────
collection: Collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=chroma_hf_ef,        # only needed for native client
    metadata={
        "hnsw:space": "cosine",             # cosine | l2 | ip
        "hnsw:construction_ef": 200,        # build quality (default 100)
        "hnsw:search_ef": 100,              # query quality (default 10)
        "hnsw:M": 16,                       # graph connections (default 16)
    },
)

# ── Get existing (raises if not found) ─────────────────────────────────────
collection = client.get_collection("my_documents")

# ── List all collections ────────────────────────────────────────────────────
collections = client.list_collections()
for col in collections:
    print(col.name, col.count())

# ── Delete collection ───────────────────────────────────────────────────────
client.delete_collection("my_documents")

# ── Collection stats ────────────────────────────────────────────────────────
print(f"Total documents: {collection.count()}")
```

---

## 5. Adding Documents

### 5a. Native ChromaDB API

```python
import uuid
from typing import Optional

def generate_ids(docs: list[str]) -> list[str]:
    """Deterministic IDs based on content hash."""
    import hashlib
    return [hashlib.md5(d.encode()).hexdigest() for d in docs]

# ── add() — insert new docs (raises on duplicate ID) ───────────────────────
collection.add(
    documents=["The sky is blue.", "Grass is green.", "Ocean is vast."],
    metadatas=[
        {"source": "nature", "category": "sky",   "year": 2024},
        {"source": "nature", "category": "plant",  "year": 2024},
        {"source": "nature", "category": "water",  "year": 2023},
    ],
    ids=["doc1", "doc2", "doc3"],
)

# ── upsert() — insert or overwrite (idempotent, preferred in production) ────
collection.upsert(
    documents=["Updated content here."],
    metadatas=[{"source": "web", "updated": True}],
    ids=["doc1"],
)

# ── Batch upsert large datasets ─────────────────────────────────────────────
def batch_upsert(
    collection: Collection,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
    batch_size: int = 500,
) -> None:
    """Upsert documents in safe-sized batches to avoid memory spikes."""
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print(f"Upserted batch {i // batch_size + 1}")
```

### 5b. LangChain `Chroma` Wrapper

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── From texts ──────────────────────────────────────────────────────────────
vectorstore = Chroma.from_texts(
    texts=["First document.", "Second document."],
    embedding=openai_embeddings,
    metadatas=[{"source": "a"}, {"source": "b"}],
    collection_name="langchain_docs",
    persist_directory="./chroma_db",
)

# ── From Document objects ───────────────────────────────────────────────────
docs = [
    Document(page_content="LangChain is a framework.", metadata={"topic": "AI"}),
    Document(page_content="ChromaDB stores vectors.", metadata={"topic": "DB"}),
]
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=openai_embeddings,
    collection_name="langchain_docs",
    persist_directory="./chroma_db",
)

# ── Load existing vectorstore ───────────────────────────────────────────────
vectorstore = Chroma(
    collection_name="langchain_docs",
    embedding_function=openai_embeddings,
    persist_directory="./chroma_db",
)

# ── Add documents to existing store ────────────────────────────────────────
ids = vectorstore.add_documents(docs)               # returns list of IDs
ids = vectorstore.add_texts(["More text."], metadatas=[{"source": "new"}])
```

---

## 6. Querying & Retrieval

### 6a. Native ChromaDB Queries

```python
# ── Similarity search ───────────────────────────────────────────────────────
results = collection.query(
    query_texts=["What is the color of the sky?"],
    n_results=5,
    include=["documents", "metadatas", "distances", "embeddings"],
)
# results keys: ids, documents, metadatas, distances, embeddings

# ── Query by raw embedding vector ───────────────────────────────────────────
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],    # pre-computed vector
    n_results=3,
)

# ── Get documents by ID ─────────────────────────────────────────────────────
result = collection.get(
    ids=["doc1", "doc2"],
    include=["documents", "metadatas"],
)

# ── Get ALL documents (paginated) ───────────────────────────────────────────
def get_all_documents(collection: Collection, batch_size: int = 1000) -> list[dict]:
    total = collection.count()
    all_docs = []
    for offset in range(0, total, batch_size):
        batch = collection.get(limit=batch_size, offset=offset)
        all_docs.extend(
            {"id": i, "doc": d, "meta": m}
            for i, d, m in zip(batch["ids"], batch["documents"], batch["metadatas"])
        )
    return all_docs
```

### 6b. LangChain Similarity Search

```python
# ── Basic similarity search ─────────────────────────────────────────────────
docs = vectorstore.similarity_search("What is LangChain?", k=4)

# ── With relevance scores (lower distance = more similar) ──────────────────
docs_with_scores = vectorstore.similarity_search_with_score("LangChain framework", k=4)
for doc, score in docs_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}")

# ── Max Marginal Relevance — reduces redundancy ─────────────────────────────
docs = vectorstore.max_marginal_relevance_search(
    query="AI frameworks",
    k=5,            # number of results to return
    fetch_k=20,     # candidates to fetch before MMR re-ranking
    lambda_mult=0.5,  # 0=max diversity, 1=max relevance
)

# ── Similarity search by raw vector ────────────────────────────────────────
embedding = openai_embeddings.embed_query("my query")
docs = vectorstore.similarity_search_by_vector(embedding, k=4)
```

---

## 7. Updating & Deleting Documents

```python
# ── Update documents (native) ────────────────────────────────────────────────
collection.update(
    ids=["doc1"],
    documents=["Updated sky description."],
    metadatas=[{"source": "nature", "updated": True}],
)

# ── Delete by ID ─────────────────────────────────────────────────────────────
collection.delete(ids=["doc1", "doc2"])

# ── Delete by metadata filter ────────────────────────────────────────────────
collection.delete(where={"source": "nature"})
collection.delete(where={"year": {"$lt": 2023}})

# ── LangChain: delete by IDs ─────────────────────────────────────────────────
vectorstore.delete(ids=["doc1", "doc2"])

# ── Check if document exists ─────────────────────────────────────────────────
def document_exists(collection: Collection, doc_id: str) -> bool:
    result = collection.get(ids=[doc_id])
    return len(result["ids"]) > 0
```

---

## 8. Metadata Filtering

ChromaDB supports rich `where` filters for both `query()` and `get()`.

```python
# ── Single condition ─────────────────────────────────────────────────────────
results = collection.query(
    query_texts=["nature"],
    n_results=5,
    where={"source": "nature"},             # exact match
)

# ── Comparison operators ─────────────────────────────────────────────────────
where={"year": {"$gte": 2023}}              # >=
where={"year": {"$lte": 2024}}              # <=
where={"year": {"$gt": 2022}}               # >
where={"year": {"$lt": 2025}}               # <
where={"year": {"$eq": 2024}}               # ==
where={"year": {"$ne": 2023}}               # !=
where={"category": {"$in": ["sky", "water"]}}   # IN list
where={"category": {"$nin": ["plant"]}}         # NOT IN list

# ── Logical operators ─────────────────────────────────────────────────────────
where={
    "$and": [
        {"source": {"$eq": "nature"}},
        {"year": {"$gte": 2023}},
    ]
}
where={
    "$or": [
        {"category": "sky"},
        {"category": "water"},
    ]
}

# ── Document content filter (substring match) ────────────────────────────────
results = collection.query(
    query_texts=["color"],
    n_results=5,
    where_document={"$contains": "blue"},
    where_document={"$not_contains": "red"},
)

# ── LangChain filter syntax ───────────────────────────────────────────────────
docs = vectorstore.similarity_search(
    "AI frameworks",
    k=5,
    filter={"topic": "AI"},                  # LangChain passes to where
)
```

---

## 9. Retrievers (LangChain Integration)

```python
from langchain_core.vectorstores import VectorStoreRetriever

# ── Basic retriever ───────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",               # similarity | mmr | similarity_score_threshold
    search_kwargs={"k": 5},
)

# ── MMR retriever ─────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
)

# ── Score threshold retriever ─────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75, "k": 10},
)

# ── With metadata filter ──────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"topic": "AI"}},
)

# ── Invoke retriever ──────────────────────────────────────────────────────────
relevant_docs = retriever.invoke("What is LangChain?")
# async
relevant_docs = await retriever.ainvoke("What is LangChain?")
```

---

## 10. RAG Pipeline — Full Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If you cannot answer from the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in docs
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ── Basic RAG chain ───────────────────────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is ChromaDB?")

# ── RAG chain that also returns source documents ──────────────────────────────
rag_chain_with_sources = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
).assign(answer=RAG_PROMPT | llm | StrOutputParser())

result = rag_chain_with_sources.invoke("What is ChromaDB?")
print(result["answer"])

# ── Streaming ─────────────────────────────────────────────────────────────────
for chunk in rag_chain.stream("Explain vector databases"):
    print(chunk, end="", flush=True)

# ── Async streaming ───────────────────────────────────────────────────────────
async for chunk in rag_chain.astream("Explain vector databases"):
    print(chunk, end="", flush=True)
```

---

## 11. MultiQueryRetriever

Generates multiple query variants to improve recall.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
    include_original=True,          # also run the original query
)

unique_docs = multi_retriever.invoke("AI vector databases")
```

---

## 12. Contextual Compression Retriever

Extracts only the relevant passage from each retrieved document.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ── Extractor: pulls only relevant sentence(s) ────────────────────────────────
extractor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor, base_retriever=base_retriever
)

# ── Filter: drops irrelevant docs entirely ────────────────────────────────────
llm_filter = LLMChainFilter.from_llm(llm)

# ── Embedding-based filter (faster, no LLM calls) ────────────────────────────
embeddings_filter = EmbeddingsFilter(
    embeddings=openai_embeddings, similarity_threshold=0.76
)

# ── Pipeline: remove redundant → filter by embedding ─────────────────────────
pipeline = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsRedundantFilter(embeddings=openai_embeddings),
        embeddings_filter,
    ]
)
pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=base_retriever
)

compressed_docs = pipeline_retriever.invoke("vector similarity search")
```

---

## 13. Self-Query Retriever

Translates natural language queries with filters into structured retrieval.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="source",   description="Document source name",     type="string"),
    AttributeInfo(name="topic",    description="Main topic of the document", type="string"),
    AttributeInfo(name="year",     description="Publication year",          type="integer"),
    AttributeInfo(name="language", description="Document language (en/fr)", type="string"),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation on AI frameworks",
    metadata_field_info=metadata_field_info,
    verbose=True,
    enable_limit=True,          # allow LLM to specify k in the query
)

# LLM parses "from 2024" and builds the filter automatically
docs = self_query_retriever.invoke("AI frameworks published in 2024")
```

---

## 14. Ensemble / Hybrid Retriever

Combines dense vector search with sparse BM25 keyword search.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Build BM25 from the same corpus
corpus_docs = [
    Document(page_content="LangChain orchestrates LLMs."),
    Document(page_content="ChromaDB is an open-source vector store."),
    Document(page_content="Embeddings represent text as dense vectors."),
]

bm25_retriever = BM25Retriever.from_documents(corpus_docs)
bm25_retriever.k = 5

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Reciprocal Rank Fusion with configurable weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.4, 0.6],     # BM25 40%, dense 60%
)

docs = ensemble_retriever.invoke("vector database embeddings")
```

---

## 15. Document Loaders & Splitters

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    JSONLoader,
    CSVLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    SemanticChunker,
)

# ── PDF loader ────────────────────────────────────────────────────────────────
pdf_loader = PyPDFLoader("document.pdf")
pages = pdf_loader.load()

# ── Web loader ────────────────────────────────────────────────────────────────
web_loader = WebBaseLoader(["https://docs.langchain.com"])
web_docs = web_loader.load()

# ── Directory loader (all .txt files) ────────────────────────────────────────
dir_loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
all_docs = dir_loader.load()

# ── JSON loader with jq schema ────────────────────────────────────────────────
json_loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",
    content_key="text",
    metadata_func=lambda record, meta: {**meta, "source": record.get("id")},
)

# ── Best-practice splitter (preserves paragraph/sentence boundaries) ──────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    add_start_index=True,       # tracks char position in metadata
)
chunks = splitter.split_documents(pages)

# ── Token-aware splitter (essential for tight context windows) ────────────────
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# ── Semantic chunker (groups by meaning, not size) ────────────────────────────
semantic_splitter = SemanticChunker(
    embeddings=openai_embeddings,
    breakpoint_threshold_type="gradient",   # percentile | standard_deviation | gradient
)
semantic_chunks = semantic_splitter.split_documents(pages)

# ── Markdown-aware splitter ────────────────────────────────────────────────────
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
md_chunks = md_splitter.split_text(markdown_text)

# ── Full pipeline: load → split → embed → store ───────────────────────────────
def ingest_documents(paths: list[str], vectorstore: Chroma) -> int:
    all_chunks = []
    for path in paths:
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        # Attach source filename to each chunk
        for chunk in chunks:
            chunk.metadata["source_file"] = path
        all_chunks.extend(chunks)
    ids = vectorstore.add_documents(all_chunks)
    return len(ids)
```

---

## 16. Persistence & Backups

```python
import shutil
from datetime import datetime
from pathlib import Path

# ── PersistentClient auto-saves on every write ────────────────────────────────
# No manual persist() call needed in chromadb >= 0.4.0

# ── Backup: copy the directory ────────────────────────────────────────────────
def backup_chroma(source_dir: str = "./chroma_db") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"./backups/chroma_{timestamp}"
    shutil.copytree(source_dir, backup_path)
    print(f"Backup saved: {backup_path}")
    return backup_path

# ── Export collection to JSON ─────────────────────────────────────────────────
import json

def export_collection(collection: Collection, output_path: str) -> None:
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data['ids'])} documents to {output_path}")

# ── Import collection from JSON ────────────────────────────────────────────────
def import_collection(collection: Collection, input_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)
    collection.upsert(
        ids=data["ids"],
        documents=data["documents"],
        metadatas=data["metadatas"],
        embeddings=data["embeddings"],
    )
    print(f"Imported {len(data['ids'])} documents")
```

---

## 17. Async Operations

```python
import asyncio
from langchain_chroma import Chroma

# ── Async similarity search ───────────────────────────────────────────────────
async def async_search(query: str) -> list:
    docs = await vectorstore.asimilarity_search(query, k=5)
    return docs

# ── Async MMR search ──────────────────────────────────────────────────────────
docs = await vectorstore.amax_marginal_relevance_search(query, k=5, fetch_k=20)

# ── Async add documents ───────────────────────────────────────────────────────
ids = await vectorstore.aadd_documents(docs)

# ── Concurrent multi-query retrieval ─────────────────────────────────────────
async def parallel_queries(queries: list[str]) -> list[list]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    tasks = [retriever.ainvoke(q) for q in queries]
    return await asyncio.gather(*tasks)

results = asyncio.run(parallel_queries(["query A", "query B", "query C"]))

# ── Async RAG chain invocation ────────────────────────────────────────────────
answer = await rag_chain.ainvoke("What is ChromaDB?")
```

---

## 18. Production Utilities

```python
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional, Any

# ── Typed response model ──────────────────────────────────────────────────────
class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    num_docs_retrieved: int = 0

# ── Deduplicate retrieved docs ────────────────────────────────────────────────
def deduplicate_docs(docs: list) -> list:
    seen: set[str] = set()
    unique = []
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    return unique

# ── Rerank by relevance score ─────────────────────────────────────────────────
def filter_by_score(docs_with_scores: list[tuple], threshold: float = 0.7) -> list:
    """Filter out documents below similarity threshold."""
    return [doc for doc, score in docs_with_scores if score >= threshold]

# ── Collection health check ────────────────────────────────────────────────────
def health_check(client, collection_name: str) -> dict:
    try:
        col = client.get_collection(collection_name)
        count = col.count()
        return {"status": "healthy", "collection": collection_name, "documents": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ── Logging wrapper ────────────────────────────────────────────────────────────
def logged_query(vectorstore: Chroma, query: str, k: int = 5) -> list:
    logger.info(f"Query: {query!r} | k={k}")
    docs = vectorstore.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(docs)} documents")
    return docs

# ── Cost estimator for OpenAI embeddings ──────────────────────────────────────
def estimate_embedding_cost(texts: list[str], price_per_1k_tokens: float = 0.00013) -> float:
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars / 4          # ~4 chars per token
    cost = (estimated_tokens / 1000) * price_per_1k_tokens
    print(f"Estimated tokens: {estimated_tokens:,.0f} | Estimated cost: ${cost:.4f}")
    return cost
```

---

## 19. Error Handling & Retry Logic

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import openai

@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def embed_with_retry(texts: list[str]) -> list[list[float]]:
    return openai_embeddings.embed_documents(texts)

# ── Safe upsert with error handling ──────────────────────────────────────────
def safe_upsert(collection: Collection, documents: list[str],
                metadatas: list[dict], ids: list[str]) -> bool:
    try:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.success(f"Upserted {len(ids)} documents")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

# ── Graceful query fallback ────────────────────────────────────────────────────
def safe_query(vectorstore: Chroma, query: str, k: int = 5) -> list:
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.warning(f"Primary search failed ({e}), retrying with k=1")
        try:
            return vectorstore.similarity_search(query, k=1)
        except Exception as fallback_err:
            logger.error(f"Fallback search also failed: {fallback_err}")
            return []
```

---

## 20. Complete End-to-End Example

A production-ready RAG service with all patterns combined.

```python
"""
rag_service.py — Production RAG service using ChromaDB + LangChain
"""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
PERSIST_DIR   = "./chroma_db"
COLLECTION    = "production_docs"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 5
SCORE_THRESH  = 0.70


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    num_docs: int


class RAGService:
    """Production-ready RAG service with ChromaDB + LangChain."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._chain = self._build_chain()
        logger.info(f"RAGService ready | docs={self.vectorstore._collection.count()}")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str, glob: str = "**/*.pdf") -> int:
        loader = DirectoryLoader(directory, glob=glob, loader_cls=PyPDFLoader)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        ids = self.vectorstore.add_documents(chunks)
        logger.success(f"Ingested {len(ids)} chunks from {directory}")
        return len(ids)

    def ingest_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        metadatas = metadatas or [{}] * len(texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        chunks = self.splitter.split_documents(docs)
        return self.vectorstore.add_documents(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _build_retriever(self):
        # Dense retriever
        dense = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3, "lambda_mult": 0.6},
        )
        # Embedding-based compression
        compressor = EmbeddingsFilter(
            embeddings=self.embeddings, similarity_threshold=SCORE_THRESH
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=dense
        )

    def _build_chain(self):
        retriever = self._build_retriever()
        prompt = ChatPromptTemplate.from_template(
            "Answer using ONLY the context below. If insufficient, say so.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        format_docs = lambda docs: "\n\n---\n".join(
            f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
        )
        return (
            RunnableParallel(
                context=retriever | format_docs,
                question=RunnablePassthrough(),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        retriever = self._build_retriever()
        docs = retriever.invoke(question)
        answer = self._chain.invoke(question)
        return RAGResponse(
            question=question,
            answer=answer,
            sources=list({d.metadata.get("source", "unknown") for d in docs}),
            num_docs=len(docs),
        )

    async def aquery(self, question: str) -> RAGResponse:
        retriever = self._build_retriever()
        docs, answer = await asyncio.gather(
            retriever.ainvoke(question),
            self._chain.ainvoke(question),
        )
        return RAGResponse(
            question=question,
            answer=answer,
            sources=list({d.metadata.get("source", "unknown") for d in docs}),
            num_docs=len(docs),
        )

    def stream(self, question: str):
        for chunk in self._chain.stream(question):
            yield chunk

    def health(self) -> dict:
        return {
            "status": "ok",
            "collection": COLLECTION,
            "total_documents": self.vectorstore._collection.count(),
        }

    def delete_by_source(self, source: str) -> None:
        self.vectorstore._collection.delete(where={"source": source})
        logger.info(f"Deleted documents with source={source!r}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    svc = RAGService()

    # Ingest sample documents
    svc.ingest_texts(
        texts=[
            "ChromaDB is an open-source AI-native vector database.",
            "LangChain is a framework for building LLM-powered applications.",
            "RAG combines retrieval with generation for grounded answers.",
        ],
        metadatas=[
            {"source": "chromadb-docs", "topic": "database"},
            {"source": "langchain-docs", "topic": "framework"},
            {"source": "rag-paper",     "topic": "technique"},
        ],
    )

    # Sync query
    response = svc.query("What is ChromaDB used for?")
    print(f"\nQ: {response.question}")
    print(f"A: {response.answer}")
    print(f"Sources: {response.sources}")

    # Streaming
    print("\nStreaming answer:")
    for chunk in svc.stream("Explain RAG in one sentence"):
        print(chunk, end="", flush=True)
    print()

    # Health
    print(f"\nHealth: {svc.health()}")
```

---

## Quick-Reference Cheat Sheet

| Task | Method |
|---|---|
| Create / get collection | `client.get_or_create_collection(name, ...)` |
| Add new docs | `collection.add(documents, metadatas, ids)` |
| Insert or overwrite | `collection.upsert(...)` |
| Query by text | `collection.query(query_texts, n_results)` |
| Query by vector | `collection.query(query_embeddings, n_results)` |
| Get by ID | `collection.get(ids=[...])` |
| Update doc | `collection.update(ids, documents, metadatas)` |
| Delete by ID | `collection.delete(ids=[...])` |
| Delete by filter | `collection.delete(where={...})` |
| Count docs | `collection.count()` |
| LangChain add | `vectorstore.add_documents(docs)` |
| LangChain search | `vectorstore.similarity_search(query, k)` |
| LangChain MMR | `vectorstore.max_marginal_relevance_search(query, k, fetch_k)` |
| LangChain retriever | `vectorstore.as_retriever(search_type, search_kwargs)` |
| Async search | `await vectorstore.asimilarity_search(query, k)` |

---

*Last updated: 2025 · ChromaDB 0.5+ · LangChain 0.3+ · langchain-chroma 0.1+*
