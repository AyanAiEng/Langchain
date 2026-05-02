# ContextualCompressionRetriever — Complete Guide

> **The memory trick, the architecture, the industry use cases, and how to build real products with it.**

---

## 🧠 THE MEMORY TRICK — Never Forget ContextualCompressionRetriever

Imagine you ask a librarian: **"What page talks about ibuprofen side effects?"**

The librarian goes to a book, pulls out the **entire 400-page medical encyclopedia**, drops it on your desk and says: *"The answer is somewhere in there."*

That's what standard RAG does. 😤

Now imagine a **smarter librarian**:

1. Finds the right book ✅
2. Opens it to the right chapter ✅
3. **Highlights only the 3 sentences that actually answer your question** ✅
4. Hands you JUST those 3 sentences ✅

```
📚 DUMB LIBRARIAN (Standard RAG):
   Query → Retrieve full chunks → Pass 2000 tokens to LLM
   LLM gets: "...ibuprofen history, manufacturing, dosage, side effects, interactions, storage..."
   (90% noise, 10% signal)

🧠 SMART LIBRARIAN (ContextualCompressionRetriever):
   Query → Retrieve full chunks → COMPRESS to relevant parts → Pass 200 tokens to LLM
   LLM gets: "Ibuprofen may cause nausea, stomach bleeding, and kidney issues."
   (100% signal, 0% noise)
```

> **The Golden Rule of Contextual Compression:**
> *"Don't give the LLM a haystack. Give it only the needles."*

---

## Table of Contents

1. [What is ContextualCompressionRetriever?](#1-what-is-it)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How It Works — Architecture & Flow](#3-how-it-works)
4. [The Three Compression Strategies](#4-compression-strategies)
5. [Code Examples — From Basic to Advanced](#5-code-examples)
6. [Industry Use Cases](#6-industry-use-cases)
7. [Where It Is Most Useful](#7-where-it-is-most-useful)
8. [Building Real Products](#8-building-real-products)
9. [Combining with Other Retrievers](#9-combining-with-other-retrievers)
10. [Comparison Table](#10-comparison-table)
11. [Limitations & When NOT to Use It](#11-limitations)
12. [Best Practices & Tuning](#12-best-practices)
13. [Summary — One-Page Cheatsheet](#13-summary)

---

## 1. What is ContextualCompressionRetriever?

`ContextualCompressionRetriever` is a LangChain retriever that wraps any existing retriever and adds a **post-retrieval compression step** — it takes the full documents retrieved by the base retriever and **filters, trims, or summarizes them** to keep only the parts relevant to the user's query.

It separates retrieval into two distinct phases:

```
Phase 1 — RETRIEVAL:   Find candidate documents (any retriever)
Phase 2 — COMPRESSION: Trim those documents to only what matters
```

### One-Line Definition

> ContextualCompressionRetriever = **"Retrieve first, then surgically remove everything irrelevant before passing to the LLM."**

### Where It Lives in LangChain

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
```

---

## 2. The Problem It Solves

### The Chunking Dilemma

Every RAG system must chunk documents before storing them. This creates an unsolvable tradeoff:

```
SMALL CHUNKS (200 tokens):
  ✅ Precise, targeted
  ❌ Lose context — sentences get cut mid-thought
  ❌ Miss answers that span chunk boundaries

LARGE CHUNKS (1000 tokens):
  ✅ Rich context, fewer boundary issues
  ❌ Carry massive amounts of irrelevant content
  ❌ Waste LLM context window
  ❌ Dilute the signal with noise
```

**There is no perfect chunk size.** ContextualCompressionRetriever solves this by letting you use large chunks for retrieval (good recall) but compressing them for generation (good precision).

### Problem 1: Context Window Pollution

Your LLM has a limited context window. If each retrieved chunk is 800 tokens and you retrieve 5 chunks, that's 4,000 tokens of context — most of which is irrelevant filler. The LLM has to "read" all of it and pick out the relevant parts, increasing cost, latency, and hallucination risk.

### Problem 2: The "Buried Answer" Problem

A document chunk might be 95% irrelevant but contains 1 golden sentence that answers the query. Standard retrieval passes the entire chunk. ContextualCompression extracts just that one sentence.

### Real Example

**Query:** *"What is the cancellation policy for annual plans?"*

**Retrieved chunk (800 tokens):**
```
Section 4: Subscription Plans and Billing

Our platform offers three tiers of subscription: Basic ($9/month),
Professional ($29/month), and Enterprise (custom pricing). All plans
include access to our core features including unlimited projects,
real-time collaboration, and 24/7 support.

Payment is processed on the first of each month for monthly plans,
and annually for yearly subscriptions. We accept all major credit
cards, PayPal, and wire transfer for Enterprise clients.

Annual plan subscribers receive a 20% discount compared to monthly
billing. This discount is applied automatically at checkout.

Cancellation Policy: Annual plan subscribers may cancel at any time,
but refunds are only issued within the first 30 days of the billing
period. After 30 days, the subscription continues until the end of
the annual term with no partial refunds.

For monthly plan subscribers, cancellation takes effect at the end
of the current billing month. No refund is issued for the current month.

Enterprise clients should contact their account manager for custom
cancellation terms outlined in their service agreement.
```

**What the LLM receives WITHOUT compression:** All 800 tokens above.

**What the LLM receives WITH compression:**
```
Annual plan subscribers may cancel at any time, but refunds are only
issued within the first 30 days of the billing period. After 30 days,
the subscription continues until the end of the annual term with no
partial refunds.
```

**Result:** 90% token reduction. 100% relevant. Cheaper, faster, more accurate.

---

## 3. How It Works — Architecture & Flow

### The Two-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│              "What is the cancellation policy?"                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: BASE RETRIEVER                      │
│  (Any retriever: VectorStore, MMR, BM25, MultiQuery, etc.)      │
│                                                                 │
│  Returns: 4 full document chunks (each ~800 tokens)             │
│  Total context: ~3200 tokens                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │  4 full chunks
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: COMPRESSOR                           │
│                                                                 │
│  For EACH chunk, the compressor asks:                           │
│  "What part of this chunk is relevant to the query?"            │
│                                                                 │
│  Chunk 1 (800 tokens) → Compressed to 2 relevant sentences      │
│  Chunk 2 (800 tokens) → Completely irrelevant → DROPPED         │
│  Chunk 3 (800 tokens) → Compressed to 1 relevant paragraph      │
│  Chunk 4 (800 tokens) → Compressed to 3 relevant sentences      │
│                                                                 │
│  Output: 3 compressed excerpts (~200 tokens total)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │  3 compressed, relevant excerpts
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANSWER GENERATION LLM                        │
│  Input: ~200 tokens of pure signal                              │
│  Output: Accurate, focused answer                               │
└─────────────────────────────────────────────────────────────────┘
```

### What Makes It "Contextual"

The word **"contextual"** is key — the compression is not generic summarization. The compressor knows the **user's query** and extracts only what is relevant to **that specific query**. The same document compressed for two different queries produces two completely different extracts.

---

## 4. The Three Compression Strategies

LangChain provides three built-in compressors, each with different tradeoffs:

### Strategy 1: LLMChainExtractor — Extract Relevant Passages

Uses an LLM to read each retrieved document and extract only the sentences/paragraphs that are relevant to the query.

```
Input:  Full 800-token chunk
Output: 1–3 extracted sentences that answer the query
Method: LLM reads and extracts
Cost:   Medium (LLM call per document)
Best for: When you need precise sentence-level extraction
```

### Strategy 2: LLMChainFilter — Filter Entire Documents

Uses an LLM to classify each retrieved document as relevant or irrelevant to the query. Keeps the entire document or drops it entirely — no partial extraction.

```
Input:  Full 800-token chunk
Output: Either the FULL chunk (relevant) or NOTHING (irrelevant)
Method: LLM classifies yes/no
Cost:   Lower (binary classification, not extraction)
Best for: When documents are already well-chunked and you just want to drop irrelevant ones
```

### Strategy 3: EmbeddingsFilter — Semantic Similarity Filter

Uses embedding similarity (no LLM call!) to drop documents that are not semantically similar enough to the query.

```
Input:  Full 800-token chunk
Output: Either the FULL chunk (above similarity threshold) or NOTHING
Method: Cosine similarity between query and doc embeddings
Cost:   Lowest (no LLM call, just math)
Best for: High-volume, low-latency pipelines where LLM cost matters
```

### Strategy Comparison

| Strategy | Method | Cost | Precision | Drops Irrelevant | Trims Within Doc |
|---|---|---|---|---|---|
| `LLMChainExtractor` | LLM extracts passages | High | Highest | ✅ | ✅ |
| `LLMChainFilter` | LLM classifies yes/no | Medium | High | ✅ | ❌ |
| `EmbeddingsFilter` | Cosine similarity | Lowest | Medium | ✅ | ❌ |
| `DocumentCompressorPipeline` | Combines the above | Varies | Best | ✅ | ✅ |

---

## 5. Code Examples — From Basic to Advanced

### Installation

```bash
pip install langchain langchain-openai langchain-community chromadb
```

### Strategy 1: LLMChainExtractor (Most Powerful)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Setup
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

docs = [
    Document(page_content="""
        Section 4: Subscription Plans and Billing
        Our platform offers Basic ($9/month), Professional ($29/month),
        and Enterprise plans. Annual plan subscribers receive a 20% discount.
        
        Cancellation Policy: Annual plan subscribers may cancel at any time,
        but refunds are only issued within the first 30 days. After 30 days,
        no partial refunds are issued.
        
        For technical support, visit our help center at support.example.com.
        Our team is available 24/7 via live chat or email.
    """),
    Document(page_content="""
        Section 7: Data Privacy and Security
        We take data privacy seriously. All user data is encrypted at rest
        using AES-256 encryption and in transit using TLS 1.3.
        
        We never sell user data to third parties. Users may request data
        deletion at any time by contacting privacy@example.com.
        
        Our systems comply with GDPR, CCPA, and SOC 2 Type II standards.
    """),
]

vectorstore = Chroma.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Create LLMChainExtractor compressor
compressor = LLMChainExtractor.from_llm(llm)

# Wrap base retriever with compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Query
query = "What is the cancellation policy for annual plans?"
compressed_docs = compression_retriever.get_relevant_documents(query)

for doc in compressed_docs:
    print(doc.page_content)
    print("---")

# Output: Only the cancellation-relevant sentences, nothing else
```

### Strategy 2: LLMChainFilter (Faster, Drops Whole Docs)

```python
from langchain.retrievers.document_compressors import LLMChainFilter

# Uses LLM as a YES/NO classifier per document
filter_compressor = LLMChainFilter.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=filter_compressor,
    base_retriever=base_retriever
)

docs = compression_retriever.get_relevant_documents(
    "What is the cancellation policy for annual plans?"
)
# Returns full relevant docs, drops the privacy/security doc entirely
```

### Strategy 3: EmbeddingsFilter (Cheapest — No LLM Calls)

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Pure math — no LLM cost
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76  # Drop docs below this cosine similarity
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)

docs = compression_retriever.get_relevant_documents(
    "What is the cancellation policy for annual plans?"
)
```

### Strategy 4: Pipeline Compressor (Best of All Worlds)

Chain multiple compressors together:

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Split large docs into smaller pieces
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separator=". "
)

# Step 2: Filter irrelevant pieces by embedding similarity
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.75
)

# Combine: Split → Filter → Result
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, embeddings_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)
```

### Full Production RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
compressor = LLMChainExtractor.from_llm(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Cheap model for compression
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 6})
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is the refund policy?"})
print(result["result"])
```

### Async Support

```python
import asyncio

async def async_retrieve(query: str):
    docs = await compression_retriever.aget_relevant_documents(query)
    return docs

results = asyncio.run(async_retrieve("What is the cancellation policy?"))
```

---

## 6. Industry Use Cases

### 6.1 Legal & Contract Analysis

**Problem:** Legal documents are notoriously long. A 200-page contract retrieved for the query "What are the termination clauses?" returns huge chunks stuffed with definitions, boilerplate, recitals, and schedules — all irrelevant.

**ContextualCompression Solution:** Retrieves contract sections, then extracts ONLY the termination-related sentences from each section.

**Products Built:**
- Contract review assistants (NDA, MSA, SLA analysis)
- Regulatory compliance checkers (GDPR article-level extraction)
- Legal due diligence tools for M&A
- Clause extraction and comparison engines
- eDiscovery platforms for litigation support

**Token Savings:** Legal chunks often compress 10:1 — from 1000 tokens down to 100.

### 6.2 Healthcare & Medical Records

**Problem:** A patient's medical record has 50 pages of history, labs, prescriptions, and notes. A query about "current medications" should return a clean list — not the entire record.

**ContextualCompression Solution:** Retrieves relevant sections of the record, then extracts only the medication-related lines.

**Products Built:**
- Clinical decision support tools
- Patient record Q&A systems (for doctors)
- Insurance pre-authorization assistants
- Medical coding and billing extraction tools
- Drug interaction checkers from patient histories

**Critical Benefit:** In healthcare, irrelevant context isn't just wasteful — it can lead to wrong diagnoses. Compression improves safety.

### 6.3 Financial Reports & Earnings Analysis

**Problem:** A 10-K filing or earnings report is 80–150 pages. An analyst asking "What did management say about supply chain risks?" needs 3 sentences, not 80 pages.

**ContextualCompression Solution:** Retrieves the MD&A (Management Discussion & Analysis) section and compresses it to risk-relevant statements.

**Products Built:**
- Earnings call transcript analyzers
- SEC filing Q&A tools
- Investment research assistants
- ESG risk report generators
- Portfolio monitoring dashboards

### 6.4 Enterprise Customer Support

**Problem:** Support documentation has long troubleshooting articles. A user asking "How do I reset my 2FA?" shouldn't get a 1200-word article about account security — just the 3 steps that answer the question.

**ContextualCompression Solution:** Retrieves support articles, compresses to the specific steps or answer relevant to the user's issue.

**Products Built:**
- AI-powered helpdesk agents (Zendesk, Freshdesk replacements)
- In-app contextual help systems
- Guided troubleshooting wizards
- Product documentation chatbots
- IT service desk automation

**Impact:** Average handle time reduced by 40–60% when agents get compressed, precise answers instead of article walls.

### 6.5 Academic Research & Literature Review

**Problem:** A researcher asks "What methods were used in studies about mRNA vaccine efficacy?" Research papers are 10–30 pages each. Standard RAG returns full paper chunks with abstracts, introductions, related work — mostly irrelevant to the methods question.

**ContextualCompression Solution:** Extracts only the methodology sections relevant to the query.

**Products Built:**
- Literature review assistants (for academics)
- Systematic review automation tools
- Patent analysis systems
- Grant writing assistants
- Research summarization tools

### 6.6 E-Commerce & Product Information

**Problem:** Product pages and catalogues have long descriptions, specs, reviews, and FAQs mixed together. A query "Is this laptop good for video editing?" needs performance specs, GPU details, and RAM — not the shipping policy or color options.

**ContextualCompression Solution:** Retrieves product data chunks and extracts only the specs relevant to the "video editing" use case.

**Products Built:**
- Conversational product advisors
- Specification comparison engines
- Review summarizers ("What do customers say about battery life?")
- B2B procurement assistants

### 6.7 Government & Public Services

**Problem:** Government regulations and policy documents are dense and long. Citizens asking "Am I eligible for housing assistance?" need specific eligibility criteria extracted from 50-page policy documents.

**ContextualCompression Solution:** Retrieves policy sections and extracts eligibility-related passages only.

**Products Built:**
- Benefits eligibility advisors
- Tax filing assistants
- Permit and licensing chatbots
- Court document Q&A systems
- Immigration guidance tools

### 6.8 Software Engineering & DevOps

**Problem:** Technical documentation, Stack Overflow answers, and runbooks are long. A developer asking "How do I configure Redis connection pooling in Python?" doesn't need the full Redis documentation page.

**ContextualCompression Solution:** Retrieves the Redis docs page, extracts only the connection pooling configuration section.

**Products Built:**
- AI coding assistants (IDE plugins)
- Internal runbook chatbots
- CI/CD pipeline troubleshooting bots
- API documentation assistants
- Security compliance checkers

---

## 7. Where It Is Most Useful

| Condition | Why ContextualCompression Helps |
|---|---|
| **Long documents / large chunks** | Removes the 90% irrelevant content from each chunk |
| **Limited LLM context window** | Compresses 3000 tokens of context to 300 |
| **Cost-sensitive applications** | Fewer tokens = lower API costs |
| **Precision-critical use cases** | LLM only sees signal, not noise |
| **Documents with mixed topics** | A single chunk may cover 5 topics; extract only the one asked about |
| **High-accuracy requirements** | Reduces hallucination risk from irrelevant context |

### The ContextualCompression Sweet Spot

```
HIGH VALUE:                           LOW VALUE:
──────────────────────────────        ──────────────────────────────
✅ Long legal/financial documents      ❌ Short, pre-chunked FAQ articles
✅ Medical records                     ❌ Knowledge base with tiny, precise chunks
✅ Annual reports, 10-K filings        ❌ Single-sentence documents
✅ Technical manuals                   ❌ When latency is more important than accuracy
✅ Multi-topic documents               ❌ When cost of LLM compression > value gained
✅ Customer support with long articles ❌ Simple exact-match lookups
```

---

## 8. Building Real Products

### Product Architecture: Enterprise Document Intelligence

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  PDFs, Word Docs, Web Pages, Databases                           │
│  → Ingested, chunked (large chunks: 800–1200 tokens), embedded   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                               │
│  Base Retriever (MMR or Standard) → Top 8 candidate chunks       │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  8 full chunks (~6400 tokens)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  COMPRESSION LAYER                               │
│  LLMChainExtractor (cheap model: gpt-3.5-turbo)                  │
│  → Extracts relevant passages from each chunk                    │
│  → Drops completely irrelevant chunks                            │
│  Output: 3–4 compressed excerpts (~400 tokens total)             │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  ~400 tokens of pure signal
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  GENERATION LAYER                                │
│  Powerful LLM (gpt-4) with minimal, high-quality context         │
│  → Accurate, focused, cost-efficient answer                      │
└──────────────────────────────────────────────────────────────────┘
```

### Production Code: Document Intelligence System

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader


class DocumentIntelligenceSystem:
    """
    Production document Q&A system using ContextualCompressionRetriever
    for precise, cost-efficient answers from large document collections.
    """

    def __init__(self, openai_api_key: str, compression_strategy: str = "extract"):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Use cheap model for compression, powerful model for answering
        self.compression_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.answer_llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vectorstore = None
        self.compression_strategy = compression_strategy

    def ingest(self, pdf_paths: list[str]):
        """Load PDFs and store with LARGE chunks (better recall)."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,   # Large chunks for retrieval recall
            chunk_overlap=100
        )
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load_and_split(splitter)
            all_docs.extend(docs)

        self.vectorstore = Chroma.from_documents(all_docs, self.embeddings)
        print(f"Indexed {len(all_docs)} chunks.")

    def _build_compressor(self):
        """Build the compression strategy."""
        if self.compression_strategy == "extract":
            # Most precise: LLM extracts relevant passages
            return LLMChainExtractor.from_llm(self.compression_llm)

        elif self.compression_strategy == "filter":
            # Faster: LLM filters whole docs
            from langchain.retrievers.document_compressors import LLMChainFilter
            return LLMChainFilter.from_llm(self.compression_llm)

        elif self.compression_strategy == "embeddings":
            # Cheapest: No LLM cost
            return EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.76
            )

        elif self.compression_strategy == "pipeline":
            # Best quality: Split → Filter pipeline
            splitter = CharacterTextSplitter(
                chunk_size=300, chunk_overlap=0, separator=". "
            )
            emb_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.75
            )
            return DocumentCompressorPipeline(
                transformers=[splitter, emb_filter]
            )

    def build_retriever(self, k: int = 6):
        """Build the compression retriever."""
        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # MMR for diverse candidate set
            search_kwargs={"k": k, "fetch_k": k * 4}
        )
        compressor = self._build_compressor()
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def ask(self, question: str) -> dict:
        """Ask a question with full compression pipeline."""
        retriever = self.build_retriever(k=6)

        qa = RetrievalQA.from_chain_type(
            llm=self.answer_llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa.invoke({"query": question})

        # Calculate token savings (approximate)
        full_tokens = sum(
            len(doc.page_content.split()) * 1.3
            for doc in result["source_documents"]
        )

        return {
            "answer": result["result"],
            "compressed_context_approx_tokens": int(full_tokens),
            "source_count": len(result["source_documents"]),
            "sources": [
                doc.metadata.get("source", "Unknown")
                for doc in result["source_documents"]
            ]
        }


# Usage
system = DocumentIntelligenceSystem(
    openai_api_key="your-key",
    compression_strategy="extract"   # or "filter", "embeddings", "pipeline"
)
system.ingest(["annual_report.pdf", "policy_manual.pdf", "contracts.pdf"])

result = system.ask("What were the main revenue drivers in Q3?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

---

## 9. Combining with Other Retrievers

ContextualCompressionRetriever is a **wrapper** — it enhances ANY base retriever. Here are the power combinations:

### Combo 1: Compression + MMR (Diversity + Precision)

```python
# MMR ensures diversity → Compression ensures precision
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.5}
)
compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=mmr_retriever   # MMR inside Compression
)
```

### Combo 2: Compression + MultiQueryRetriever (Recall + Precision)

```python
# MultiQuery ensures recall → Compression ensures precision
from langchain.retrievers.multi_query import MultiQueryRetriever

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm
)
compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=mq_retriever   # MultiQuery inside Compression
)
```

### Combo 3: The Ultimate Pipeline (MMR + MultiQuery + Compression)

```python
# Step 1: MultiQuery for maximum recall
mq_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 16, "lambda_mult": 0.5}
    ),
    llm=query_gen_llm  # Cheap LLM for query generation
)

# Step 2: Compression for maximum precision
compressor = LLMChainExtractor.from_llm(compression_llm)

ultimate_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=mq_retriever
)
```

### The Full Retrieval Stack Visualized

```
USER QUERY
    │
    ▼
MultiQueryRetriever
(Generates 4 query variants → 4×MMR searches → 16 diverse candidate docs)
    │
    ▼
ContextualCompressionRetriever
(Processes each of the 16 docs → Extracts only relevant passages → Drops irrelevant)
    │
    ▼
Final Context: 4–6 compressed excerpts, ~500 tokens of pure signal
    │
    ▼
LLM Answer Generation
```

---

## 10. Comparison Table

| Retriever | What It Optimizes | Problem It Solves |
|---|---|---|
| **Standard VectorStore** | Speed | Baseline retrieval |
| **MMR** | Diversity | Redundant results |
| **MultiQueryRetriever** | Recall | Vocabulary/phrasing gaps |
| **ContextualCompression** | Precision | Noisy, irrelevant context |
| **MMR + Compression** | Diversity + Precision | Redundancy + Noise |
| **MultiQuery + Compression** | Recall + Precision | Vocabulary gaps + Noise |
| **All Three Combined** | Recall + Diversity + Precision | Production-grade RAG |

---

## 11. Limitations & When NOT to Use It

### Limitations

**1. Adds Latency**
Each compressed document requires an extra LLM call (for LLMChainExtractor/Filter). For 6 retrieved docs, that's 6 additional LLM calls before the final answer. On average, this adds 2–5 seconds.

**2. Adds Cost**
LLMChainExtractor uses tokens for compression. At scale, this can be significant. Mitigation: use a cheap model (gpt-3.5-turbo) for compression, a powerful model for answering.

**3. Risk of Over-Compression**
The compressor might extract too aggressively, removing context that provides necessary background for understanding the extracted sentence.

**4. Compressor Can Be Wrong**
If the LLM compressor misidentifies what's relevant (especially with ambiguous queries), you lose information before the final LLM even sees it. This is rarer but possible.

**5. Not Useful for Short, Precise Chunks**
If your knowledge base is already well-chunked into 100-token precise sentences, compression adds cost with no benefit.

### When NOT to Use ContextualCompression

- Knowledge base documents are already short and precise
- Ultra-low latency is required (< 500ms)
- Very high volume with tight cost constraints → use EmbeddingsFilter instead
- Simple yes/no or exact-match queries

---

## 12. Best Practices & Tuning

### ✅ Use a Cheap LLM for Compression

Compression is classification/extraction — it doesn't need GPT-4:

```python
# GOOD — cheap for compression, powerful for answering
compression_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
answer_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

compressor = LLMChainExtractor.from_llm(compression_llm)  # Cheap
qa = RetrievalQA(llm=answer_llm, retriever=compression_retriever)  # Powerful
```

### ✅ Use Large Base Chunks + Compression

Counter-intuitively, ContextualCompression works BETTER with larger chunks:

```python
# GOOD — large chunks (more context for compressor to work with)
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

# BAD — small chunks already precise, compression adds no value
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
```

### ✅ Choose Compressor by Budget

```python
# High budget, highest quality
compressor = LLMChainExtractor.from_llm(llm)

# Medium budget, good quality
compressor = LLMChainFilter.from_llm(llm)

# Low budget / high volume
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# Best balance (no LLM cost + smart splitting)
pipeline = DocumentCompressorPipeline(transformers=[splitter, embeddings_filter])
```

### ✅ Tune EmbeddingsFilter Threshold

```python
# Too high (0.90): Drops too many docs, misses relevant content
# Too low (0.50): Keeps too many docs, no real filtering
# Sweet spot: 0.72 – 0.80 for most use cases

EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
```

### ✅ Always Retrieve More Than You Need at Base Level

```python
# Retrieve 8–10 docs at base level → Compression will reduce to 3–5
# Don't retrieve only 3 at base level — compression might drop 2, leaving you with 1

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Generous
```

---

## 13. Summary — One-Page Cheatsheet

### What ContextualCompressionRetriever Does

```
Standard RAG:
  Query → Retrieve full chunks (lots of noise) → LLM answers (diluted context)

ContextualCompression RAG:
  Query → Retrieve full chunks → COMPRESS each chunk → LLM answers (pure signal)
```

### The Memory Trick

> A standard librarian hands you a 400-page encyclopedia.
> **ContextualCompressionRetriever is the smart librarian** who highlights only the 3 sentences that answer your question.

### Three Compressor Strategies

```
LLMChainExtractor  → Best quality,  highest cost  (extracts relevant sentences)
LLMChainFilter     → Good quality,  medium cost   (keeps or drops whole doc)
EmbeddingsFilter   → Good quality,  zero LLM cost (cosine similarity filter)
Pipeline           → Best quality,  medium cost   (split + filter combo)
```

### When to Use It

✅ Large documents with mixed content  
✅ Limited LLM context window  
✅ Cost-sensitive pipelines (compress = fewer final tokens)  
✅ Precision-critical applications (legal, medical, financial)  
✅ Any RAG system where retrieved chunks are noisy  

### Quick Start Code

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 6})
)
```

### The Power Stack

```
MultiQueryRetriever  →  finds everything relevant  (RECALL)
       +
MMR Retriever        →  removes duplicates          (DIVERSITY)
       +
ContextualCompression → removes noise               (PRECISION)
       =
Production-Grade RAG 🚀
```

---

*LangChain Docs: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression*  
*Original concept: RAG with selective context — Lewis et al. (2020), extended with compression strategies*
