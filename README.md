# llm-assisted-lightweight-rag
A local lightweight Retrieval-Augmented Generation (RAG) pipeline for extracting scheduling and monitoring data, and determining status from PDF using Large Language Models.  Combines deterministic lexical scoring, structural pattern boosts (dates/terms), and LLM-assisted query expansion for reliable legal document analysis.
---

## Features

- PDF ingestion using `pdfplumber`
- Word-based document chunking
- LLM-assisted query expansion (canonical phrases, synonyms, keywords)
- Weighted lexical retrieval (no vector DB required)
- Structural boosts for:
  - Date patterns
  - Validity/term expressions
  - Agreement-date signals
- Context-restricted LLM answering
- Batch processing with rate limiting
- Prompt logging for evaluation
- CSV response logging

---

## Architecture

### 1. Document Ingestion
- Extracts text from PDFs or text files
- Splits documents into chunks using delimiter-based segmentation

### 2. Query Expansion (Optional)
LLM generates:
- Canonical phrases  
- Synonyms  
- Keywords  

Strict JSON output enforcement ensures deterministic parsing.

### 3. Hybrid Retrieval
Chunks are ranked using weighted scoring:

- Canonical phrase matches  
- Synonym matches  
- Keyword matches  
- Regex-based structural signals:
  - Date detection
  - Validity detection

Fallback:
- Phrase-only search using quoted phrases

### 4. Context-Constrained Answering
Only top-k chunks are sent to the model.

The model is instructed:

- Use only provided context  
- Return deterministic outputs for expiry checks  

### 5. Logging & Evaluation
- Prompt snapshots saved locally
- CSV response logs for batch runs

## Setup

### Requirements

- Python 3.10+
- requests
- pdfplumber (optional but recommended)
Limitations

    Character-based prompt budgeting (not token-based)

    No vector database

    Limited semantic ranking compared to embeddings

    Date arithmetic delegated to the LLM

    Single-process batch execution

## Install dependencies:

```bash
pip install requests pdfplumber
```

## Environment Variables

### Set your Model token:

export GITHUB_PAT=your_token_here

Optional:

export GITHUB_MODEL=openai/gpt-4.1
export GITHUB_SEARCH_MODEL=openai/gpt-4o
