# RAG Agent

A from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline using local models via [Ollama](https://ollama.com/).

## How it works

1. **Chunk** — the knowledge base (`knowledge.txt`) is split into paragraphs
2. **Embed** — each chunk is embedded using `nomic-embed-text` via Ollama
3. **Retrieve** — a user question is embedded and compared against chunk embeddings using cosine similarity
4. **Generate** — the best-matching chunk is injected into a prompt sent to `llama3.2`, which answers using only that context

## Files

| File | Purpose |
|---|---|
| `rag.py` | Step 1 — load and chunk a document |
| `rag_embedding.py` | Step 2 — generate embeddings for each chunk |
| `rag_embedding_chatboy.py` | Step 3 — retrieve the best matching chunk for a query |
| `RAG_final_chatbot.py` | Full pipeline — end-to-end RAG Q&A |
| `knowledge.txt` | Sample knowledge base (TechCorp company info) |

## Setup

**Prerequisites:** [Ollama](https://ollama.com/) installed and running

```bash
# Start Ollama
ollama serve

# Pull the required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Install Python dependencies
pip install ollama numpy
```

## Usage

```bash
python RAG_final_chatbot.py
```

The chatbot answers questions strictly based on the content in `knowledge.txt`. To use your own knowledge base, replace `knowledge.txt` with your own text file (paragraphs separated by blank lines).
