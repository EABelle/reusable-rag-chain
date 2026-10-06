# Reusable RAG Chain

A small Retrieval-Augmented Generation (RAG) example built with LangChain, Chroma, and OpenAI. It reads a text file, splits it into chunks, embeds them into an in-memory Chroma vector store, and answers questions using the most relevant chunks.

## Requirements

- Python 3.12 (3.9 works but is end-of-life and shows SSL warnings on macOS)
- An OpenAI API key (https://platform.openai.com/api-keys)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

`.env` is git-ignored. Never commit your key.

## Usage

```python
from main import RagChain

r = RagChain()
r.add_file("sample.txt")                 # chunk, embed, and store the file
r.query("capital of Japan")              # print the top 2 matching chunks
print(r.invoke("What is the highest mountain in Japan?"))  # full RAG answer
```

## Quick test

```bash
python -m py_compile main.py && python -c "import main"

python -c "
from main import RagChain
r = RagChain()
r.add_file('sample.txt')
r.query('capital of Japan')
print(r.invoke('What is the highest mountain in Japan?'))
"
```

Expected answer: Mount Fuji, at 3,776 meters.

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | `RagChain` class: reader, vector store, retriever, and LLM chain |
| `sample.txt` | Example document for testing |
| `requirements.txt` | Pinned dependencies |
| `.env` | Your API key (not committed) |

## How it works

1. `Reader` splits the file into 1000-character chunks with 200 overlap.
2. `OpenAIEmbeddings` (`text-embedding-3-large`) embeds each chunk into Chroma.
3. On a question, the top 2 similar chunks are retrieved.
4. `gpt-4o-mini` answers using those chunks as context.

## Notes

- The vector store is in-memory only, so data is lost when the process exits.
- Messages like `Failed to send telemetry event ... capture()` come from a chromadb/posthog version mismatch and are harmless.
