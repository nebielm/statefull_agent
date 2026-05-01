# Stateful Agent

A small LangGraph-based conversational agent with:
- tool calling
- structured and unstructured user memory
- Chroma-backed retrieval
- OpenRouter chat model integration
- Hugging Face embeddings

The app currently runs as a CLI chat loop through [`main.py`](main.py).

## Current Architecture

The codebase is now organized by responsibility under [`app/`](app/):

```text
app/
  api/            placeholder for future routes/controllers
  core/           settings and shared logging setup
  db/             vectorstore initialization
  llm/            LLM client, prompt templates, extraction helpers
  models/         domain constants/models
  repositories/   JSON-backed persistence access
  schemas/        shared state/schema types
  services/       memory, retrieval, tools, graph, chat loop
  utils/          small reusable helpers
prompts/          compatibility wrappers for prompt imports
tests/            placeholder test package
data/             runtime-generated local state
main.py           CLI entry point
```

## Requirements

- Python 3.13
- A virtual environment at `.venv`
- An OpenRouter or OpenAI-compatible API key in `.env`

## Environment Variables

Create a `.env` file with at least one of:

```env
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

Optional:

```env
HF_TOKEN=your_huggingface_token
```

`HF_TOKEN` is not required, but it helps avoid lower rate limits when the embedding model is downloaded from Hugging Face.

## Install Dependencies

If the environment is not already prepared:

```bash
./.venv/bin/pip install requests python-dotenv langchain-core langchain-openai langchain-openrouter langchain-huggingface langchain-community langchain-text-splitters langchain-chroma langgraph chromadb sentence-transformers
```

## Run

Start the CLI:

```bash
./.venv/bin/python main.py
```

You can exit with:

```text
quit
```

or

```text
exit
```

## How It Works

On each user turn, the app:

1. extracts a retrieval plan
2. loads relevant structured memory, unstructured memory, and knowledge
3. ranks the retrieved context
4. selects tools if needed
5. answers the user
6. writes memory updates back to local storage

The orchestration graph lives in [`app/services/graph.py`](app/services/graph.py).

## Data Directory

The current [`data/`](data/) directory is runtime state, not source code. It contains things like:

- Chroma persistence files
- user memory
- local JSON state

Recommended approach:

- Do not commit generated runtime data like `data/user_memory/`, `data/core_knowledge/`, or `data/user_info.json`.
- If you want seed knowledge in git, commit the raw source documents separately, not the generated Chroma database files.
- If you want the directory to exist in the repo, keep only a placeholder such as `.gitkeep` or a small `data/README.md`.

In other words: for this project as it exists now, I would treat `data/` as non-committable generated state.

## Checks

The refactored project has been validated with:

- Python compile checks
- import/startup checks
- CLI smoke test through [`main.py`](main.py)

There is not yet a real automated test suite in [`tests/`](tests/).

## Notes

- Startup initializes embeddings and vectorstores eagerly.
- On first run, the embedding model may be downloaded from Hugging Face.
- The app currently uses a CLI interface only; `app/api/` is a placeholder for future expansion.
