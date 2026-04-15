# LLM Observability & Evaluation System
<img width="1118" height="801" alt="Pasted Graphic 2" src="https://github.com/user-attachments/assets/cde5799c-1d7a-4ddc-8afb-7a547015bdab" />
<img width="1118" height="801" alt="Pasted Graphic 3" src="https://github.com/user-attachments/assets/ace67be8-d5e6-4869-8afb-a93e57c1550e" />


## 📌 Overview

This project focuses on building a production-grade observability and evaluation system for LLM-based applications (e.g., RAG systems and agentic workflows).

The goal is to monitor, debug, and improve AI system performance by tracking:
- Latency
- Token usage
- Cost per request
- Retrieval quality
- Hallucination / faithfulness

This system will integrate with an existing RAG pipeline and provide full visibility into request execution.

---

## 🎯 Objectives

1. Add tracing and instrumentation across the LLM pipeline
2. Track performance metrics (latency, cost, tokens)
3. Evaluate response quality using automated metrics
4. Build a basic dashboard or logging system
5. Enable regression testing for model/system changes

---

## 🧠 System Architecture

User Query
   ↓
FastAPI Backend
   ↓
RAG Pipeline
   ├── Retriever (ChromaDB)
   ├── Re-ranker (Cross Encoder)
   ├── Prompt Builder
   ├── LLM (Groq / Claude)
   ↓
Response

Observability Layer (this project)
   ├── Tracing
   ├── Metrics Logging
   ├── Evaluation Engine
   ├── Storage (JSON / SQLite)
   └── Dashboard (optional)

---

## ⚙️ Core Features

### 1. Tracing & Instrumentation

Track the full lifecycle of every request:

- User query
- Retrieved documents (top-k)
- Re-ranked documents (order + scores)
- Final prompt sent to LLM
- LLM response

Each request should generate a structured trace object.

---

### 2. Metrics Tracking

#### Latency
- Measure total request time
- Measure LLM response time
- Compute:
  - P50 latency
  - P95 latency

#### Token Usage
- Prompt tokens
- Completion tokens
- Total tokens

#### Cost (Approximate)
- Estimate cost per request based on token usage

---

### 3. Evaluation System

Evaluate quality of LLM responses using:

#### Metrics:
- Faithfulness (does answer match retrieved context?)
- Context relevance
- Answer correctness (optional)

#### Methods:
- Rule-based checks
- LLM-based evaluation (optional, low-cost)

---

### 4. Logging & Storage

Store logs in:

- JSONL file OR
- SQLite database

Each record should include:
```json
{
  "query": "...",
  "retrieved_docs": [...],
  "reranked_docs": [...],
  "prompt": "...",
  "response": "...",
  "latency": 0.45,
  "tokens": {
    "prompt": 120,
    "completion": 80
  },
  "cost": 0.0002,
  "evaluation": {
    "faithfulness": 0.9
  }
}
