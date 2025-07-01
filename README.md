# Home Assistant Ollama Proxy

A lightweight proxy that sits between Home Assistant and Ollama, providing think tag filtering for cleaner LLM responses.

## What it does

This proxy forwards requests to an Ollama server while filtering out `<think>` tags from chat responses. This ensures that Home Assistant receives clean, presentable output from language models that use thinking/reasoning tags.

## Running with Docker

```bash
docker run -d \
  --name ollama-proxy \
  -p 8000:8000 \
  -e OLLAMA_URL=http://your-ollama-server:11434 \
  -e LOG_LEVEL=INFO \
  your-image-name
```

## Running with Python

```bash
# Install dependencies
uv sync

# Run the proxy
OLLAMA_URL=http://localhost:11434 uv run uvicorn home_assistant_ollama_proxy.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

- `OLLAMA_URL` - The URL of your Ollama server (default: `http://ollama-rocm.don`)
- `LOG_LEVEL` - Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: `INFO`)

## Endpoints

- `/health` - Health check endpoint
- `/api/chat` - Chat endpoint with think tag filtering
- `/*` - All other requests are proxied directly to Ollama

## Home Assistant Configuration

Point your Home Assistant Ollama integration to this proxy instead of directly to Ollama:

```yaml
# Example configuration
ollama_url: http://your-proxy:8000
```