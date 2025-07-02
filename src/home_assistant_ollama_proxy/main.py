import json
import os

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama-rocm.don")

level = os.environ.get("LOG_LEVEL", "INFO").upper()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
log = structlog.get_logger()

log.info("Ollama proxy starting", ollama_url=OLLAMA_URL, log_level=level)


@app.get("/health")
async def health_check():
  """Health check endpoint."""
  return {"status": "healthy", "ollama_url": OLLAMA_URL}


async def _forward_request(
  request: Request, full_path: str, client: httpx.AsyncClient
) -> httpx.Response:
  """
  Forwards the incoming request to the Ollama server.
  """
  url = httpx.URL(OLLAMA_URL, path=f"/{full_path}")

  # Debug: Log incoming request details
  log.debug(
    "Building forward request",
    full_path=full_path,
    method=request.method,
  )

  # Use pre-read body if available, otherwise stream
  content = getattr(request, "_body", None) or request.stream()

  req = client.build_request(
    method=request.method,
    url=url,
    headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
    params=request.query_params.multi_items(),
    content=content,
  )

  log.info("Forwarding request", url=str(req.url), method=req.method)

  try:
    response = await client.send(req, stream=True)
    log.debug("Received response", status_code=response.status_code)
    return response
  except Exception as e:
    log.error(
      "Failed to forward request", error_type=type(e).__name__, error_message=str(e), exc_info=True
    )
    raise


def _filter_think_content(content: str, inside_think_block: bool) -> tuple[str, bool]:
  """
  Filter content based on <think> tags, returning the filtered content and updated state.

  Args:
      content: The content to filter
      inside_think_block: Whether we're currently inside a think block

  Returns:
      tuple of (filtered_content, new_inside_think_block_state)
  """
  if not content:
    return content, inside_think_block

  log.debug(
    "Filtering content",
    content_length=len(content),
    inside_think_block=inside_think_block,
  )

  result = ""
  remaining = content
  current_state = inside_think_block

  while remaining:
    if current_state:
      # We're inside a think block, look for closing tag
      if "</think>" in remaining:
        # Found closing tag, skip everything up to and including it
        _, remaining = remaining.split("</think>", 1)
        current_state = False
      else:
        # No closing tag found, skip all remaining content
        break
    else:
      # We're outside a think block, look for opening tag
      if "<think>" in remaining:
        # Found opening tag, keep content before it
        before_tag, after_tag = remaining.split("<think>", 1)
        result += before_tag
        remaining = after_tag
        current_state = True
      else:
        # No opening tag found, keep all remaining content
        result += remaining
        break

  log.debug(
    "Filter result",
    original_length=len(content),
    filtered_length=len(result),
    state_changed=inside_think_block != current_state,
    final_state=current_state,
  )

  return result, current_state


async def _strip_think_tags_stream(response_stream: httpx.Response, request_id: int | None = None):
  """
  Stream response data while stripping <think> tags from chat message content.

  Handles think tags that span multiple JSON chunks by maintaining state
  across chunk boundaries.
  """
  buffer = b""
  inside_think_block = False
  chunk_count = 0
  bytes_processed = 0
  lines_processed = 0

  log.info("Starting stream processing", request_id=request_id)

  try:
    # Check if response stream is already closed
    if response_stream.is_closed:
      log.warning("Response stream already closed", request_id=request_id)
      return

    async for chunk in response_stream.aiter_bytes():
      chunk_count += 1
      chunk_size = len(chunk)
      bytes_processed += chunk_size

      log.debug(
        "Received chunk",
        request_id=request_id,
        chunk_number=chunk_count,
        chunk_size=chunk_size,
      )
      buffer += chunk

      # Process complete lines
      while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        lines_processed += 1

        try:
          decoded_line = line.decode("utf-8")
          json_data = json.loads(decoded_line)

          # Process message content if present
          if "message" in json_data and "content" in json_data["message"]:
            original_content = json_data["message"]["content"]
            filtered_content, inside_think_block = _filter_think_content(
              original_content, inside_think_block
            )
            json_data["message"]["content"] = filtered_content

            if original_content != filtered_content:
              log.debug(
                "Content modified",
                request_id=request_id,
                original_length=len(original_content),
                filtered_length=len(filtered_content),
              )

          # Skip empty content messages when inside think blocks
          # unless they contain important status information
          should_yield = True
          if inside_think_block and "message" in json_data:
            # Only yield if done=true or content is non-empty
            is_final = json_data.get("done", False)
            has_content = bool(json_data.get("message", {}).get("content", "").strip())
            should_yield = is_final or has_content

          if should_yield:
            output = json.dumps(json_data).encode("utf-8") + b"\n"

            try:
              yield output
              log.debug(
                "Yielded message",
                request_id=request_id,
                inside_think_block=inside_think_block,
                is_final=json_data.get("done", False),
                has_content=bool(json_data.get("message", {}).get("content", "").strip()),
                message_content=json_data.get("message", {}).get("content", "")[:100] + "..."
                if len(json_data.get("message", {}).get("content", "")) > 100
                else json_data.get("message", {}).get("content", ""),
                total_duration=json_data.get("total_duration"),
                load_duration=json_data.get("load_duration"),
                eval_count=json_data.get("eval_count"),
              )
            except Exception as yield_error:
              log.error(
                "Error yielding output",
                request_id=request_id,
                error_type=type(yield_error).__name__,
                error_message=str(yield_error),
              )
              raise
          else:
            log.debug(
              "Skipped empty message during think block",
              request_id=request_id,
            )

        except json.JSONDecodeError as e:
          log.warning(
            "JSON decode error",
            request_id=request_id,
            line_content=line.decode("utf-8", errors="replace"),
            error_message=str(e),
          )
          # If JSON parsing fails, yield the original line
          yield line + b"\n"

    # Process any remaining content in the buffer
    if buffer.strip():
      log.debug(
        "Processing remaining buffer",
        request_id=request_id,
        buffer_size=len(buffer),
      )

      try:
        json_data = json.loads(buffer.decode("utf-8"))

        if "message" in json_data and "content" in json_data["message"]:
          original_content = json_data["message"]["content"]
          filtered_content, _ = _filter_think_content(original_content, inside_think_block)
          json_data["message"]["content"] = filtered_content

        log.debug(
          "Yielding final buffer message",
          request_id=request_id,
          is_final=json_data.get("done", False),
          has_content=bool(json_data.get("message", {}).get("content", "").strip()),
          message_content=json_data.get("message", {}).get("content", "")[:100] + "..."
          if len(json_data.get("message", {}).get("content", "")) > 100
          else json_data.get("message", {}).get("content", ""),
        )
        yield json.dumps(json_data).encode("utf-8") + b"\n"

      except json.JSONDecodeError:
        log.debug(
          "Yielding non-JSON buffer content", request_id=request_id, buffer_size=len(buffer)
        )
        yield buffer + b"\n"

  except httpx.ReadError as e:
    # ReadError with empty message often means the stream ended normally
    if not str(e) and chunk_count > 0:
      log.debug(
        "Stream ended (possibly normal completion)",
        request_id=request_id,
        chunks_processed=chunk_count,
        bytes_processed=bytes_processed,
        lines_processed=lines_processed,
        inside_think_block=inside_think_block,
      )
    else:
      log.warning(
        "Stream read error",
        request_id=request_id,
        error_message=str(e) or "No error message",
        chunks_processed=chunk_count,
        bytes_processed=bytes_processed,
        lines_processed=lines_processed,
        buffer_remaining=len(buffer),
        inside_think_block=inside_think_block,
      )
    # Don't re-raise - let the stream end gracefully
  except (httpx.ConnectError, httpx.TimeoutException) as e:
    log.warning(
      "Stream connection error",
      request_id=request_id,
      error_type=type(e).__name__,
      error_message=str(e),
      chunks_processed=chunk_count,
      bytes_processed=bytes_processed,
      lines_processed=lines_processed,
      buffer_remaining=len(buffer),
      inside_think_block=inside_think_block,
    )
    # Don't re-raise - let the stream end gracefully
  except Exception as e:
    log.error(
      "Unexpected error in stream processing",
      request_id=request_id,
      error_type=type(e).__name__,
      error_message=str(e),
      chunks_processed=chunk_count,
      bytes_processed=bytes_processed,
      lines_processed=lines_processed,
      buffer_remaining=len(buffer),
      inside_think_block=inside_think_block,
      exc_info=True,
    )
    # Don't re-raise - let the stream end gracefully
  finally:
    log.info(
      "Stream processing completed",
      request_id=request_id,
      chunks_processed=chunk_count,
      bytes_processed=bytes_processed,
      lines_processed=lines_processed,
      final_buffer_size=len(buffer),
      ended_inside_think_block=inside_think_block,
    )


@app.post("/api/chat")
async def chat_proxy(request: Request):
  """
  Proxies /api/chat requests to Ollama, stripping <think> tags from the response.
  """
  request_id = id(request)
  log.info(
    "Incoming chat request",
    request_id=request_id,
    path=request.url.path,
    method=request.method,
    content_type=request.headers.get("content-type"),
  )

  # Log request body for debugging
  try:
    body = await request.body()
    if body:
      try:
        body_json = json.loads(body)
        log.debug(
          "Chat request body",
          request_id=request_id,
          model=body_json.get("model"),
          messages_count=len(body_json.get("messages", [])),
          stream=body_json.get("stream", False),
          options=body_json.get("options", {}),
        )
        # Log messages content at debug level
        for i, msg in enumerate(body_json.get("messages", [])):
          log.debug(
            "Chat message",
            request_id=request_id,
            message_index=i,
            role=msg.get("role"),
            content_length=len(msg.get("content", "")),
            content_preview=msg.get("content", "")[:100] + "..."
            if len(msg.get("content", "")) > 100
            else msg.get("content", ""),
          )
      except json.JSONDecodeError:
        log.debug("Chat request body (non-JSON)", request_id=request_id, body_size=len(body))
      # Store the body for later use in forward_request
      request._body = body
  except Exception as e:
    log.warning("Failed to read request body", request_id=request_id, error=str(e))

  async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
    try:
      resp = await _forward_request(request, "api/chat", client)

      log.debug(
        "Chat response received",
        request_id=request_id,
        status_code=resp.status_code,
        content_type=resp.headers.get("content-type"),
        headers={k: v for k, v in resp.headers.items() if k not in {"authorization", "api-key"}},
      )

      # Remove headers that can conflict with streaming
      headers_to_remove = {"content-length", "content-encoding", "transfer-encoding"}
      headers = {k: v for k, v in resp.headers.items() if k.lower() not in headers_to_remove}
      # Ensure we have chunked transfer encoding for streaming
      headers["transfer-encoding"] = "chunked"

      return StreamingResponse(
        _strip_think_tags_stream(resp, request_id),
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type", "application/x-ndjson"),
      )
    except httpx.TimeoutException as e:
      log.error(
        "Chat proxy timeout",
        request_id=request_id,
        error_type=type(e).__name__,
        error_message=str(e),
      )
      return StreamingResponse(content="", status_code=504)
    except Exception as e:
      log.error(
        "Error in chat proxy",
        request_id=request_id,
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True,
      )
      return StreamingResponse(content="", status_code=500)


@app.api_route(
  "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
async def reverse_proxy(request: Request, full_path: str):
  """
  A general-purpose reverse proxy that forwards all requests to the OLLAMA_URL.
  """
  log.info("Incoming request", path=request.url.path, method=request.method)
  async with httpx.AsyncClient() as client:
    try:
      resp = await _forward_request(request, full_path, client)
      # Remove Content-Length header to avoid conflicts with StreamingResponse
      headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
      return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        headers=headers,
      )
    except Exception as e:
      log.error("Error in general proxy", error=e, exc_info=True)
      return StreamingResponse(content="", status_code=500)
