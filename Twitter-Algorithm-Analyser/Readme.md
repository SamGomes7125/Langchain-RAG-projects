# Twitter Algorithm Analyser

A RAG (Retrieval-Augmented Generation) system that answers technical questions about Twitter's open-sourced recommendation algorithm codebase.

## How it works

1. Clones and indexes the entire `the-algorithm` repository (~34,600 code chunks)
2. Embeds all files using HuggingFace sentence-transformers
3. Stores embeddings in a local Deep Lake vector database
4. Uses conversational retrieval (LangChain + Groq/Llama 3.3) to answer natural-language questions grounded in the actual source code, with multi-turn conversation memory

## Stack

- **LLM:** Groq (Llama 3.3 70B)
- **Embeddings:** HuggingFace (sentence-transformers)
- **Vector store:** Deep Lake (local)
- **Framework:** LangChain

## Setup

1. `pip install -r requirements.txt`
2. `git clone https://github.com/twitter/the-algorithm`
3. Create `.env` with `GROQ_API_KEY`
4. `python ingestion.py` (builds the vector database — one-time, ~34k chunks)
5. `python main.py` (runs example queries against the codebase)
