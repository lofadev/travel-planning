Travel Planner – Setup and Run Guide

## Overview

Travel Planner extracts trip details from Vietnamese natural language requests, searches related data (accommodations, dining, attractions, transportation) via Tavily, and generates a detailed itinerary using LangGraph + OpenAI.

Primary source files in `src/`:

-   `src/state.py`: Defines `TravelState`
-   `src/prompt.py`: Prompts for extraction and itinerary generation
-   `src/node.py`: Core nodes (parse, ask for missing info, search, generate)
-   `src/graph.py`: Builds and compiles the `graph`

## Requirements

-   Python >= 3.11
-   One of the following dependency managers:
    -   uv (recommended) – very fast, reads `pyproject.toml` and `uv.lock`
    -   pip + venv (standard)
-   Accounts and API Keys:
    -   OpenAI: to call the chat model
    -   Tavily: to search for travel information

## Environment setup

1. Copy the env template

```bash
cp .env.example .env
```

2. Fill these values in `.env` (example):

```bash
LANGSMITH_API_KEY=lsv2_...    # optional, for LangSmith tracing if you use it
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-...
```

Notes:

-   `OPENAI_MODEL` can be any Chat Completions–compatible model (e.g., `gpt-4o-mini`, `gpt-4o`, `o4-mini`, ...). Use a model you have access to.

## Install dependencies

### Option 1: Using uv (recommended)

```bash
# Install uv if needed (Windows can use pipx or winget)
# pipx install uv

uv sync
```

This creates a virtualenv and installs all packages defined in `pyproject.toml` and `uv.lock`.

Activate the virtualenv (if needed):

```bash
# PowerShell
. .venv\Scripts\Activate.ps1

# Git Bash/CMD
source .venv/Scripts/activate
```

### Option 2: Using pip + venv

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
# Or: .\.venv\Scripts\activate  # Windows (CMD/PowerShell)

python -m pip install --upgrade pip
pip install -e .
```

## Run the app

Two common ways: run LangGraph Dev Server (with UI/devtools) or call the graph from Python.

### A) LangGraph Dev Server (reads `langgraph.json`)

LangGraph provides a CLI to run graphs from `langgraph.json`.

```bash
# With virtualenv activated and `.env` present
langgraph dev
```

-   The dev server will start and load the `travel-planner` graph from `./src/graph.py:graph` as declared in `langgraph.json`.
-   You can use the CLI’s UI/devtools to experiment, or call the dev server’s internal API (see LangGraph docs for custom endpoints).

Tip: If you hit environment errors, ensure `.env` is at repo root and matches `langgraph.json` (`"env": ".env"`).

The result will be the itinerary (Markdown) or a follow-up question if required info is missing.

## Processing flow (brief)

-   `parse_input` (LLM) extracts info from Vietnamese → `extracted_info`
-   `check_info` validates required fields (`destination`, `departure_location`, `duration`, `people_count`)
-   If missing → `ask_for_info` produces a clarifying question
-   If complete → `search_info` runs parallel Tavily searches
-   `generate_itinerary` composes a Markdown itinerary

## Quick test tips

-   Provide a prompt with all required fields: destination, departure, duration, people count
-   Add preferences to see the impact on search/itinerary (e.g., “cà phê chill”, “nature photography”)

## Troubleshooting

-   Missing `OPENAI_API_KEY` or `TAVILY_API_KEY`: check `.env`
-   Invalid `OPENAI_MODEL`: switch to a model you can access
-   `langgraph` CLI not found: ensure it’s installed (declared in `pyproject.toml`) and your virtualenv is active
-   SSL/Proxy issues on Windows: try another network or configure your proxy

## Development

-   Format/lint tooling: `black`, `isort`, `flake8` (declared in `pyproject.toml`)
-   Optional pre-commit: `pre-commit install`

## License

MIT (adjust to your project needs)
