# RAG Q&A System

A Retrieval-Augmented Generation (RAG) pipeline for question answering over custom document collections. Ingest your documents, ask questions, get answers grounded in your data.

## Features

- **Document Ingestion** -- load PDFs, text files, and markdown
- **Chunking Strategies** -- fixed-size, sentence-based, and semantic chunking
- **Vector Store** -- FAISS-based vector storage with persistent indexing
- **Embedding Models** -- sentence-transformers for dense retrieval
- **LLM Integration** -- generate answers using HuggingFace models
- **Source Attribution** -- every answer cites its source documents
- **CLI Interface** -- ingest, query, and manage from the command line

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rag_qa import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Ingest documents
rag.ingest("./documents/")

# Ask questions
answer = rag.query("What are the main findings?")
print(answer.text)
print(answer.sources)  # Source documents used
```

## CLI Usage

```bash
# Ingest a directory of documents
python -m rag_qa ingest --path ./documents/

# Ask a question
python -m rag_qa query "What is the main conclusion?"

# Interactive Q&A session
python -m rag_qa interactive

# Show index statistics
python -m rag_qa stats
```

## Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Embedder       |
                    | (sentence-transformers)
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Vector Store    |
                    |    (FAISS)        |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Context Builder   |
                    | (top-k retrieval) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   LLM Generator   |
                    | (answer + sources)|
                    +------------------+
```

## Configuration

```python
from rag_qa import RAGPipeline, RAGConfig

config = RAGConfig(
    chunk_size=512,
    chunk_overlap=50,
    top_k=5,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.3,
)

rag = RAGPipeline(config=config)
```

## License

MIT
