# MMR — Maximal Marginal Relevance: The Complete Guide

> **The memory trick, the math, the industry use cases, and how to build real products with it.**

---

## 🧠 THE MEMORY TRICK — Never Forget MMR Again

Imagine you're building a **fruit basket** as a gift.

You go to the store and the shopkeeper ranks fruits by how "good" they are:

```
1st: Red Apple    ← most popular, take it ✅
2nd: Green Apple  ← also popular... but wait
3rd: Banana       ← different, interesting
4th: Red Apple    ← SAME as #1, skip it ❌
5th: Orange       ← different, take it ✅
```

A **dumb picker** takes the top 5 by score → you get: 🍎 🍏 🍎 🍎 🍎
A **smart picker (MMR)** balances quality AND variety → you get: 🍎 🍌 🍊 🍇 🍓

**MMR is the smart fruit basket picker.**

It always asks TWO questions at once:
1. ✅ *Is this item relevant to what I need?*
2. ✅ *Is this item DIFFERENT from what I already have?*

> **The Golden Rule of MMR:**
> "Give me the most relevant thing I haven't effectively seen yet."

---

## Table of Contents

1. [What is MMR?](#1-what-is-mmr)
2. [The Problem MMR Solves](#2-the-problem-mmr-solves)
3. [How MMR Works — The Math Made Simple](#3-how-mmr-works)
4. [MMR vs Standard Similarity Search](#4-mmr-vs-standard-similarity-search)
5. [Code Examples — From Basic to Advanced](#5-code-examples)
6. [Industry Use Cases](#6-industry-use-cases)
7. [Where MMR is Most Useful](#7-where-mmr-is-most-useful)
8. [Building Real Products with MMR](#8-building-real-products)
9. [Tuning the Lambda Parameter](#9-tuning-lambda)
10. [MMR + MultiQueryRetriever Together](#10-mmr--multiqueryretriever-together)
11. [Limitations & When NOT to Use MMR](#11-limitations)
12. [Best Practices](#12-best-practices)
13. [Summary — The One-Page Cheatsheet](#13-summary)

---

## 1. What is MMR?

**Maximal Marginal Relevance (MMR)** is a selection algorithm that retrieves documents (or any items) that are simultaneously:

- **Relevant** — close to the user's query in meaning
- **Diverse** — different from each other, avoiding redundant results

It was originally proposed in a 1998 paper by Carbonell & Goldstein for **search result diversification**, and has since become a core strategy in modern RAG (Retrieval-Augmented Generation) pipelines.

### One-Line Definition

> MMR = **"Pick the next document that is most relevant to the query AND most different from what you've already picked."**

### Where It Lives in LangChain

```python
# As a retrieval search type on any vector store
vectorstore.as_retriever(search_type="mmr")

# Or directly
vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.5)
```

---

## 2. The Problem MMR Solves

### The Redundancy Problem in Vector Search

Standard vector similarity search ranks documents purely by how close they are to your query embedding. This causes a critical flaw:

**If 10 documents in your knowledge base all say roughly the same thing, standard search returns ALL 10 of them.**

#### Real Example — Without MMR

Query: *"What are the benefits of exercise?"*

Standard top-5 results:
```
Doc 1: "Exercise improves cardiovascular health and reduces heart disease risk."
Doc 2: "Regular exercise strengthens your heart and lowers heart disease chances."
Doc 3: "Working out is great for your heart health and cardiovascular system."
Doc 4: "Exercise boosts cardiovascular fitness and protects against heart problems."
Doc 5: "Physical activity promotes heart health and reduces cardiovascular risk."
```

**All 5 documents say the same thing.** Your LLM gets zero variety. It can't answer about mental health benefits, weight management, bone density, sleep quality — all of which exist in your knowledge base but weren't retrieved.

#### Real Example — With MMR

Query: *"What are the benefits of exercise?"*

MMR top-5 results:
```
Doc 1: "Exercise improves cardiovascular health and reduces heart disease risk."
Doc 2: "Regular physical activity is linked to reduced anxiety and depression."
Doc 3: "Weight-bearing exercise increases bone density and prevents osteoporosis."
Doc 4: "Exercise improves sleep quality and reduces insomnia symptoms."
Doc 5: "Physical activity boosts metabolism and helps with weight management."
```

**5 different dimensions of benefits.** The LLM can now give a rich, comprehensive answer.

### The Core Problem in One Sentence

> Standard search optimizes for **relevance only**. MMR optimizes for **relevance + diversity**, eliminating redundant results and maximizing information coverage.

---

## 3. How MMR Works — The Math Made Simple

### The MMR Formula

```
MMR = argmax [ λ · Sim(doc, query) − (1 − λ) · max Sim(doc, selected_docs) ]
```

Let's break this down in plain English:

```
MMR score = (How relevant is this doc to the query?)
           MINUS
           (How similar is this doc to docs I've already selected?)
```

The **λ (lambda)** parameter controls the balance:
- `λ = 1.0` → Pure relevance (identical to standard search)
- `λ = 0.0` → Pure diversity (ignores relevance entirely)
- `λ = 0.5` → Equal balance (most common default)

### Step-by-Step Algorithm

**Setup:** You want to select K documents. MMR fetches a larger pool of `fetch_k` candidates first, then selects K from them.

```
Step 1: Embed the query
Step 2: Fetch top fetch_k=20 documents by raw similarity score
Step 3: Select the FIRST document (highest relevance score)

Step 4: For each remaining document, compute:
        MMR_score = λ × sim(doc, query) − (1−λ) × max_sim(doc, already_selected)

Step 5: Select the document with the HIGHEST MMR score
Step 6: Repeat Step 4–5 until K documents are selected
```

### Visual Walk-Through

```
Query: "How to improve team productivity?"
fetch_k = 6 candidates, we want k = 3

Candidate pool (by relevance to query):
  A: "Daily standups improve team productivity"          sim=0.95
  B: "Morning meetings boost team output"                sim=0.92  ← very similar to A
  C: "Async communication reduces meeting fatigue"       sim=0.85
  D: "Clear goal-setting improves team performance"      sim=0.82
  E: "Office plants make people more creative"           sim=0.40
  F: "Team bonding events improve morale and output"     sim=0.75

ROUND 1: Select A (highest relevance) → Selected: {A}

ROUND 2: Compute MMR for remaining docs:
  B: 0.5 × 0.92 − 0.5 × sim(B,A) = 0.46 − 0.5×0.90 = 0.46 − 0.45 = 0.01  ← penalized (too similar to A)
  C: 0.5 × 0.85 − 0.5 × sim(C,A) = 0.42 − 0.5×0.20 = 0.42 − 0.10 = 0.32  ← wins! different from A
  D: 0.5 × 0.82 − 0.5 × sim(D,A) = 0.41 − 0.5×0.30 = 0.41 − 0.15 = 0.26
  F: 0.5 × 0.75 − 0.5 × sim(F,A) = 0.37 − 0.5×0.40 = 0.37 − 0.20 = 0.17

  → Select C → Selected: {A, C}

ROUND 3: Compute MMR for remaining docs:
  B: 0.5 × 0.92 − 0.5 × max(sim(B,A), sim(B,C)) = 0.46 − 0.45 = 0.01
  D: 0.5 × 0.82 − 0.5 × max(sim(D,A), sim(D,C)) = 0.41 − 0.15 = 0.26  ← wins!
  F: 0.5 × 0.75 − 0.5 × max(sim(F,A), sim(F,C)) = 0.37 − 0.20 = 0.17

  → Select D → Selected: {A, C, D}

FINAL RESULT:
  A: "Daily standups improve team productivity"
  C: "Async communication reduces meeting fatigue"
  D: "Clear goal-setting improves team performance"
```

**Note how B was almost completely excluded** despite being the 2nd most relevant document — because it added zero new information.

---

## 4. MMR vs Standard Similarity Search

| Aspect | Standard Similarity Search | MMR Search |
|---|---|---|
| **Selection criterion** | Highest similarity to query | Highest relevance + lowest overlap with selected |
| **Result diversity** | Low — similar docs cluster together | High — each result adds new information |
| **Risk** | Redundant, repetitive context | Slightly less precision on top result |
| **LLM context quality** | Often 5 docs saying the same thing | 5 docs covering different facets |
| **Best for** | Finding THE most similar doc | Finding K maximally informative docs |
| **Speed** | Faster | Slightly slower (iterative selection) |
| **Key parameter** | `k` (number of results) | `k`, `fetch_k`, `lambda_mult` |

---

## 5. Code Examples — From Basic to Advanced

### Installation

```bash
pip install langchain langchain-openai langchain-community chromadb
```

### Basic MMR Search

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Create vector store
embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content="Exercise improves cardiovascular health."),
    Document(page_content="Regular workouts strengthen your heart."),
    Document(page_content="Physical activity reduces heart disease risk."),
    Document(page_content="Exercise reduces anxiety and depression symptoms."),
    Document(page_content="Working out boosts mental health significantly."),
    Document(page_content="Exercise improves bone density and strength."),
    Document(page_content="Physical activity helps with weight management."),
]
vectorstore = Chroma.from_documents(docs, embeddings)

# ❌ Standard search — likely returns 3 heart-related docs
standard_results = vectorstore.similarity_search(
    "benefits of exercise", k=3
)

# ✅ MMR search — returns 3 DIVERSE docs
mmr_results = vectorstore.max_marginal_relevance_search(
    "benefits of exercise",
    k=3,            # Number of final results to return
    fetch_k=10,     # Candidate pool size (always > k)
    lambda_mult=0.5 # Balance: 0=pure diversity, 1=pure relevance
)

print("Standard results:")
for doc in standard_results:
    print(f"  - {doc.page_content}")

print("\nMMR results:")
for doc in mmr_results:
    print(f"  - {doc.page_content}")
```

### MMR as a LangChain Retriever

```python
# Use search_type="mmr" to get an MMR-powered retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

# Works with any LangChain chain
docs = mmr_retriever.get_relevant_documents("What are the benefits of exercise?")
```

### Full RAG Pipeline with MMR

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True
)

response = qa_chain.invoke({"query": "What are the benefits of regular exercise?"})
print(response["result"])
```

### With FAISS

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)

mmr_results = vectorstore.max_marginal_relevance_search(
    query="benefits of exercise",
    k=5,
    fetch_k=25,
    lambda_mult=0.6
)
```

### With Pinecone (Production)

```python
from langchain_community.vectorstores import Pinecone

vectorstore = Pinecone.from_existing_index(
    index_name="my-index",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5}
)
```

### Comparing Standard vs MMR Side-by-Side

```python
def compare_retrieval(query: str, vectorstore, k: int = 4):
    """Show the difference between standard and MMR retrieval."""
    
    standard = vectorstore.similarity_search(query, k=k)
    mmr = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=k*4, lambda_mult=0.5
    )
    
    print(f"Query: '{query}'\n")
    
    print("=" * 50)
    print("STANDARD SEARCH (Relevance Only)")
    print("=" * 50)
    for i, doc in enumerate(standard, 1):
        print(f"{i}. {doc.page_content[:100]}")
    
    print("\n" + "=" * 50)
    print("MMR SEARCH (Relevance + Diversity)")
    print("=" * 50)
    for i, doc in enumerate(mmr, 1):
        print(f"{i}. {doc.page_content[:100]}")

compare_retrieval("team productivity tips", vectorstore)
```

---

## 6. Industry Use Cases

### 6.1 Enterprise Search & Knowledge Management

**Problem:** A company's Confluence wiki has 200 articles about "deployment procedures." All 200 are slightly similar. Standard search returns 5 nearly identical articles.

**MMR Solution:** Retrieves 5 articles covering: CI/CD pipeline setup, rollback procedures, environment configs, monitoring setup, and disaster recovery — different dimensions of the same topic.

**Products Built:**
- Internal knowledge base chatbots (IT, HR, Engineering)
- Confluence / Notion AI assistants
- Employee onboarding Q&A systems
- Runbook and incident response assistants

### 6.2 Legal Research & Contract Analysis

**Problem:** A legal database has 500 case precedents about "breach of contract." Lawyers need diverse precedents — different courts, different industries, different rulings — not 5 similar rulings from the same court.

**MMR Solution:** Retrieves cases that are relevant to the query but each from a different legal angle, jurisdiction, or ruling outcome.

**Products Built:**
- Legal research assistants (westlaw-style AI)
- Contract clause recommendation engines
- Regulatory compliance checkers
- Case law summarization tools

### 6.3 News Aggregation & Media Monitoring

**Problem:** A news monitoring system for "Tesla earnings report" returns 20 nearly identical wire service articles. Clients need diverse coverage — Bloomberg's analysis, Reuters' raw data, WSJ's opinion, analyst commentary, social media sentiment.

**MMR Solution:** Retrieves news articles that are relevant to the topic but each offering a unique perspective or data source.

**Products Built:**
- Brand monitoring dashboards
- Competitive intelligence feeds
- Executive news briefing generators
- Market research summarizers

### 6.4 E-Commerce & Product Recommendations

**Problem:** A customer searches for "comfortable running shoes." Without MMR, the recommendation engine shows 5 nearly identical Nike Air Max models. The customer sees no variety.

**MMR Solution:** Returns shoes that are all relevant to "comfortable running" but span different styles, brands, price points, and features — increasing the chance of a purchase.

**Products Built:**
- AI-powered product discovery engines
- Personalized recommendation carousels
- Chatbot-based shopping assistants
- Upsell/cross-sell recommendation systems

### 6.5 Medical Literature & Clinical Research

**Problem:** A doctor searching for "treatment options for Type 2 diabetes" gets 5 papers all about metformin. A comprehensive clinical picture requires papers on lifestyle interventions, newer drug classes, surgical options, and comorbidity management.

**MMR Solution:** Retrieves research papers that collectively cover the breadth of treatment approaches, not just the most common one.

**Products Built:**
- Clinical decision support tools
- Medical literature review assistants
- Drug research summarizers
- Patient education platforms

### 6.6 Customer Support & FAQ Systems

**Problem:** A support bot for a SaaS product gets asked "my account isn't working." The knowledge base has 50 articles about login issues. Standard search returns 5 login-related articles. But the problem might be billing, permissions, browser compatibility, or VPN issues.

**MMR Solution:** Retrieves articles covering login errors, billing blocks, permission issues, browser compatibility, and account status — giving the support agent or chatbot the full diagnostic picture.

**Products Built:**
- AI-powered helpdesk agents
- Tier-1 support automation
- Guided troubleshooting wizards
- SaaS customer success bots

### 6.7 Education & Adaptive Learning

**Problem:** A student asks "explain photosynthesis." The knowledge base has many textbook chunks about light reactions. MMR ensures the response draws from light reactions, dark reactions, chloroplast structure, and real-world applications — a complete picture.

**Products Built:**
- AI tutoring systems
- Personalized study guide generators
- Exam question generators
- Course content recommendation engines

### 6.8 Financial Research & Analysis

**Problem:** An analyst asks "what are risks in the semiconductor industry?" Standard search returns 5 geopolitical supply chain articles. MMR retrieves supply chain risk, demand cyclicality, R&D costs, talent shortage, and regulatory risk — all different dimensions.

**Products Built:**
- Investment research assistants
- Earnings call analysis tools
- Risk assessment dashboards
- ESG reporting assistants

---

## 7. Where MMR Is Most Useful

MMR delivers maximum value when these conditions are true:

| Condition | Why MMR Helps |
|---|---|
| **Knowledge base has redundant documents** | Prevents returning 5 docs that say the same thing |
| **User needs a comprehensive answer** | Different docs cover different facets of the topic |
| **Topics have many sub-dimensions** | Each retrieved doc adds genuinely new information |
| **Context window is limited** | Every token of context must carry maximum unique information |
| **Recommendation systems** | Users want variety, not repetition |
| **Research & summarization tasks** | Breadth of coverage matters as much as depth |

### The MMR Sweet Spot

```
HIGH VALUE for MMR:                     LOW VALUE for MMR:
──────────────────────────────          ──────────────────────────────
✅ "Explain all risks of X"              ❌ "What is the boiling point of water?"
✅ "What are the options for Y?"         ❌ "Show me the exact clause about X"
✅ "Compare approaches to Z"            ❌ "Find the most similar document to this one"
✅ "Give me a comprehensive view of W"   ❌ "What did CEO say in Q3 earnings call?"
✅ Recommendation carousels             ❌ Exact match lookup
```

---

## 8. Building Real Products with MMR

### Product Architecture: Intelligent Document Q&A

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│ Document     │     │ Ingestion &      │     │ Vector Store            │
│ Sources      │────▶│ Chunking         │────▶│ (Chroma/Pinecone/FAISS) │
│ - PDFs       │     │ - 500 token      │     │ - Dense embeddings      │
│ - Web pages  │     │   chunks         │     │ - Metadata stored       │
│ - Databases  │     │ - 50 token       │     └────────────┬────────────┘
└──────────────┘     │   overlap        │                  │
                     └──────────────────┘                  │
                                                           │
                              ┌────────────────────────────▼───────────────┐
                              │              MMR RETRIEVER                  │
                              │  fetch_k=20 candidates from vector store    │
                              │  Iteratively select k=5 diverse + relevant  │
                              │  λ=0.5 (balanced) or tune per use case     │
                              └────────────────────────────┬───────────────┘
                                                           │
                              ┌────────────────────────────▼───────────────┐
                              │         ANSWER GENERATION (LLM)            │
                              │  5 diverse, non-redundant context docs      │
                              │  → Comprehensive, multi-faceted answer      │
                              └────────────────────────────────────────────┘
```

### Production Code: Document Q&A System

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


class DiverseKnowledgeBot:
    """
    Production Q&A system using MMR for diverse, comprehensive retrieval.
    """

    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.vectorstore = None

    def ingest_documents(self, pdf_paths: list[str]):
        """Load, chunk, and embed documents."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load_and_split(splitter)
            all_docs.extend(docs)

        self.vectorstore = Chroma.from_documents(all_docs, self.embeddings)
        print(f"Ingested {len(all_docs)} chunks from {len(pdf_paths)} documents.")

    def build_retriever(self, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
        """Create MMR-powered retriever."""
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )

    def ask(self, question: str, diversity: float = 0.5) -> dict:
        """
        Ask a question.
        diversity: 0.0 = maximum diversity, 1.0 = maximum relevance
        """
        retriever = self.build_retriever(
            k=5,
            fetch_k=25,
            lambda_mult=diversity  # User-controllable!
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "?") for doc in result["source_documents"]],
            "diversity_setting": diversity
        }


# Usage
bot = DiverseKnowledgeBot(openai_api_key="your-key")
bot.ingest_documents(["company_policy.pdf", "employee_handbook.pdf"])

result = bot.ask(
    question="What are our remote work options?",
    diversity=0.5   # Balanced: relevant AND diverse
)
print(result["answer"])
```

### Product Feature: User-Controlled Diversity Slider

In a real product UI, let users control λ with a slider:

```
Diversity Slider:
  [Focused ←─────────────────→ Comprehensive]
       0.8         0.5              0.2
  (More precise)  (Balanced)  (More diverse)
```

```python
# Map slider value to lambda
def slider_to_lambda(slider_value: float) -> float:
    """
    slider_value: 0.0 (Comprehensive) to 1.0 (Focused)
    Returns lambda for MMR retriever
    """
    return slider_value  # 0.0 = diverse, 1.0 = relevance-focused

# User sets slider to 0.3 (wants comprehensive view)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.3}
)
```

---

## 9. Tuning the Lambda Parameter

The `lambda_mult` parameter is the most important knob in MMR. Here's a practical guide:

### Lambda Values Cheatsheet

| Lambda Value | Behavior | Use When |
|---|---|---|
| `1.0` | Pure relevance (= standard search) | You want the single most relevant doc |
| `0.7 – 0.9` | Relevance-leaning with some diversity | FAQ systems, precise lookups |
| `0.5` | Balanced (recommended default) | General Q&A, most use cases |
| `0.3 – 0.4` | Diversity-leaning | Research, exploration, summaries |
| `0.0 – 0.2` | Near-pure diversity | Serendipitous discovery, brainstorming |

### The fetch_k Multiplier Rule

Always set `fetch_k` to at least `4×k`:

```python
k = 5           # Final results you want
fetch_k = 20    # Candidates to consider (4× is minimum, 6× is better)

# Rule of thumb:
fetch_k = max(k * 4, 20)
```

Why? If `fetch_k` is too small, MMR has nothing diverse to choose from and the results look the same as standard search.

### Tuning by Use Case

```python
# FAQ / Precise lookup — prioritize relevance
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.8}
)

# Research / Comprehensive Q&A — balance relevance + diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.5}
)

# Recommendation / Discovery — prioritize diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.3}
)
```

---

## 10. MMR + MultiQueryRetriever Together

MMR and MultiQueryRetriever solve **different problems** and work powerfully together:

| Tool | Problem Solved |
|---|---|
| **MultiQueryRetriever** | *Recall* — finds relevant docs you might miss with one query |
| **MMR** | *Diversity* — removes redundancy from whatever was retrieved |

### Combined Architecture

```
User Query
    │
    ▼
MultiQueryRetriever
(Generate 4 query variants → Run 4 searches → Merge → 20 candidate docs)
    │
    ▼
MMR Selection
(From 20 candidates, pick 5 that are relevant AND diverse from each other)
    │
    ▼
LLM Answer Generation
(Rich, comprehensive, non-redundant context)
```

### Combined Code

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Step 1: MMR retriever as the BASE retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.5}
)

# Step 2: Wrap MMR retriever in MultiQueryRetriever
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

combined_retriever = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever,  # MMR as the inner retriever
    llm=llm
)

# Result: MultiQuery expands recall, MMR removes redundancy
docs = combined_retriever.get_relevant_documents(
    "What are the risks and benefits of the new drug?"
)
```

### When to Use Each

```
Only MMR                → Large knowledge base, redundant docs, need diversity
Only MultiQuery         → Vocabulary gap, users phrase questions differently
MMR + MultiQuery        → Production systems needing maximum recall + quality
Neither                 → Small, clean knowledge base with precise queries
```

---

## 11. Limitations & When NOT to Use MMR

### Limitations

**1. Slightly Slower Than Standard Search**
MMR is an iterative selection algorithm — it runs K rounds of scoring. For large K or huge candidate pools, this adds latency. In practice, the difference is milliseconds for most setups.

**2. Can Miss the Best Document**
If two nearly identical documents are both highly relevant, MMR might skip the second-best one in favor of a more diverse but slightly less relevant document. In precision-critical scenarios, this is a drawback.

**3. Lambda Tuning Required**
The default lambda of 0.5 doesn't work equally well for all use cases. You must tune it per domain and query type to get optimal results.

**4. Diversity Without Relevance Threshold**
MMR doesn't filter out irrelevant documents — it just penalizes similar ones. If your knowledge base has many loosely related documents, you might still get off-topic results in the name of "diversity."

### When NOT to Use MMR

- **Exact match / fact lookup** — "What is the boiling point of water?" needs the most accurate answer, not a diverse set of answers.
- **Finding the single most similar document** — pure cosine similarity is better.
- **Very small knowledge bases** — if you only have 10 documents, standard search is fine.
- **When you want clustering** — MMR is not a clustering algorithm. Use K-means or HDBSCAN for that.

---

## 12. Best Practices

### ✅ Always Set fetch_k ≥ 4×k

```python
# GOOD
search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.5}

# BAD — fetch_k too small, MMR has no room to be diverse
search_kwargs={"k": 5, "fetch_k": 6, "lambda_mult": 0.5}
```

### ✅ Start with lambda_mult=0.5, Then Tune

Run your test queries with 0.3, 0.5, and 0.7. Pick the value where results feel most useful — not just relevant, and not randomly diverse.

### ✅ Combine with Metadata Filtering

Use metadata filters BEFORE MMR to narrow the candidate pool:

```python
results = vectorstore.max_marginal_relevance_search(
    query="treatment options",
    k=5,
    fetch_k=20,
    lambda_mult=0.5,
    filter={"category": "cardiology"}  # Filter first, then diversify within
)
```

### ✅ Log & Evaluate

Periodically compare standard vs MMR results to confirm diversity is actually helping answer quality:

```python
def evaluate_diversity(docs: list) -> float:
    """Simple diversity metric: average pairwise distance between retrieved docs."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    embeddings_matrix = np.array([doc.metadata.get("embedding") for doc in docs])
    sim_matrix = cosine_similarity(embeddings_matrix)
    n = len(docs)
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # Exclude diagonal
    return 1 - avg_sim  # Diversity score (higher = more diverse)
```

### ✅ Use MMR in Recommendation Systems Always

Any time you're showing a list of items (documents, products, articles) to a user, MMR is almost always better than standard ranking because users want variety, not repetition.

---

## 13. Summary — The One-Page Cheatsheet

### What MMR Does

```
Standard Search:  Query → Top K most similar docs (often redundant)
MMR Search:       Query → Top K most relevant AND diverse docs
```

### The Memory Trick

> Imagine building a fruit basket. A dumb picker takes the top 5 fruits by popularity = 5 apples.
> **MMR is the smart picker** that ensures you get an apple, a banana, an orange, a mango, and grapes — all good choices, all different.

### The Formula in English

```
Next doc = Most relevant to query
           MINUS
           Most similar to docs already selected
```

### The Key Parameter

```
lambda_mult: 0.0 ──────────── 0.5 ──────────── 1.0
           Pure diversity   Balanced    Pure relevance
```

### When to Use MMR

✅ Large knowledge bases with redundant content  
✅ Questions needing comprehensive, multi-faceted answers  
✅ Recommendation systems  
✅ Research and summarization tools  
✅ Any RAG system where context window must be maximally informative  

### Quick Start Code

```python
# 3 lines to add MMR to any LangChain vector store
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)
```

---

*LangChain MMR Docs: https://python.langchain.com/docs/modules/data_connection/retrievers/*  
*Original MMR Paper: Carbonell & Goldstein (1998) — "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"*
