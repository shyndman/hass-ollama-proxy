# Project Overview

This project, "Home Assistant Ollama Proxy", acts as a proxy for the Ollama API, primarily designed to integrate with Home Assistant. It forwards requests to an Ollama server and includes specific logic to strip `<think>` tags from chat responses.

# Technology Stack

- **Language:** Python 3.13
- **Web Framework:** FastAPI
- **HTTP Client:** httpx
- **Logging:** structlog
- **ASGI Server:** uvicorn
- **Dependency Management:** uv
- **Linting/Formatting:** ruff
- **Testing:** pytest
- **Containerization:** Docker

# Project Structure

- `src/home_assistant_ollama_proxy/`: Contains the main application logic.
  - `main.py`: FastAPI application with proxy logic and `<think>` tag stripping.
- `docs/home-assistant-ollama/`: Contains files related to the Home Assistant integration, including configuration flows, conversation agents, and entity definitions.
- `pyproject.toml`: Project metadata, dependencies, and tool configurations (ruff, pyright).
- `uv.lock`: Lock file for `uv` dependency management.
- `Dockerfile`: Dockerfile for building the application image.
- `.pre-commit-config.yaml`: Pre-commit hooks for ruff and uv-lock.

# Key Features/Logic

- **Ollama Proxy:** Forwards all incoming requests to a configurable Ollama server (`http://ollama-rocm.don`).
- **Think Tag Stripping:** Specifically for `/api/chat` endpoints, it processes the Ollama response stream to remove `<think>` tags from the chat content, which is useful for cleaner output in conversational AI contexts.
- **Home Assistant Integration:** Provides components for Home Assistant to interact with Ollama, including configuration flows, conversation entities, and handling of LLM interactions.

# Development Workflow

- **Dependency Installation:** `uv sync`
- **Linting & Formatting:** Managed by `ruff` via pre-commit hooks.
- **Testing:** `pytest` (configured in `pyproject.toml`).

# Important Notes/Conventions

- The Ollama server URL is hardcoded in `src/home_assistant_ollama_proxy/main.py` as `http://ollama-rocm.don`.
- The project uses `uv` for dependency management, and `uv.lock` is committed to version control.
- Pre-commit hooks are configured to ensure code quality and dependency lock file consistency.