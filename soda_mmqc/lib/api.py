import os
import json
import base64
import io
import math
import numbers
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Dict, Any, Tuple, Optional
from soda_mmqc import logger
from soda_mmqc.config import API_PROVIDER, DEFAULT_MODEL, DEFAULT_MODELS

# Try to import PIL for image compression
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available. Image compression will be disabled.")

# API clients will be imported dynamically when needed


# --- Langfuse helpers: simplified wrapper using langfuse.get_client() ---
try:
    # Prefer the documented entrypoint used by tests/examples
    from langfuse import get_client as _lf_get_client  # type: ignore
except Exception:
    _lf_get_client = None


def _get_langfuse_client() -> Optional[Any]:
    """Return a Langfuse client instance, or None if unavailable.

    This is intentionally minimal: it mirrors the example in
    `test_langfuse.py` and prefers the `get_client()` helper the SDK
    exposes. The calling code should treat a None return as "no tracing".
    Returns None immediately when LANGFUSE_PUBLIC_KEY is not configured so
    the SDK never emits authentication warnings.
    """
    import os as _os
    if not _os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        if _lf_get_client is None:
            from langfuse import get_client as _gc  # type: ignore
            return _gc()
        return _lf_get_client()
    except Exception:
        return None


def _lf_start_trace(client: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Start a top-level Langfuse span/trace and return it, or None.

    The returned object is the SDK span/trace; callers can use the
    helper `_lf_log_generation` below to record a generation and finish
    the trace. All operations are best-effort and swallow exceptions so
    tracing never breaks normal program flow.
    """
    if client is None:
        return None
    try:
        return client.start_span(name=name, metadata=(metadata or {}))
    except Exception:
        # Try a couple of common alternate names the SDK might provide
        try:
            if hasattr(client, "start_trace"):
                return client.start_trace(name=name, metadata=(metadata or {}))
        except Exception:
            pass
    return None


def _sanitize_trace_token(value: Any, default: str) -> str:
    """Return a trace-safe token (lowercase alnum, dash, underscore)."""
    try:
        text = str(value).strip().lower()
    except Exception:
        text = ""
    if not text:
        return default
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    token = "".join(safe).strip("-")
    return token or default


def _infer_model_type(model: str) -> str:
    """Infer a stable model-family token for trace naming."""
    model_token = _sanitize_trace_token(model, "unknown")
    parts = model_token.split("-")
    if model_token.startswith("gpt-") and len(parts) >= 2:
        return "-".join(parts[:2])
    if model_token.startswith("claude-") and len(parts) >= 3:
        return "-".join(parts[:3])
    if model_token.startswith("o") and parts:
        return parts[0]
    return parts[0] if parts else model_token


def _build_lf_trace_name(provider: str, model: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Build trace name from provider + model type + sub-function."""
    metadata = metadata or {}
    sub_function = (
        metadata.get("sub_function")
        or metadata.get("phase")
        or metadata.get("operation")
        or metadata.get("mode")
        or "evaluate"
    )
    provider_token = _sanitize_trace_token(provider, "provider")
    model_type = _infer_model_type(model)
    sub_function_token = _sanitize_trace_token(sub_function, "evaluate")
    return f"{provider_token}_{model_type}_{sub_function_token}_generate_response"


def _lf_log_generation(client: Any, trace: Any, span_name: str, gen_name: str, model: str, input_text: str, output_text: str, prompt_obj: Optional[Any] = None) -> None:
    """Best-effort: create a child span + generation observation and finish.

    This mirrors the sequence used in `test_langfuse.py`:
      - trace.start_span(...) -> child span
      - span.update(input=...)
      - span.start_observation(as_type="generation", ...)
      - end observation, update span with output, end span, end trace
      - flush client
    All steps are wrapped in try/except and won't raise.
    """
    try:
        if not (client and trace):
            return
        span = None
        try:
            if hasattr(trace, "start_span"):
                span = trace.start_span(name=span_name)
            elif hasattr(trace, "start_observation"):
                span = trace.start_observation(name=span_name)
        except Exception:
            span = None

        if span is None:
            return

        try:
            if hasattr(span, "update"):
                span.update(input={"prompt": input_text})
        except Exception:
            pass

        try:
            if hasattr(span, "start_observation"):
                # If a Langfuse prompt object is provided, pass it so the
                # observation is linked to the prompt in Langfuse UI.
                kwargs = dict(
                    as_type="generation",
                    name=gen_name,
                    model=model,
                    input=input_text,
                    output=output_text,
                )
                if prompt_obj is not None:
                    try:
                        kwargs["prompt"] = prompt_obj
                    except Exception:
                        pass

                gen = span.start_observation(**kwargs)
                try:
                    gen.end()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(span, "update"):
                span.update(output=output_text)
        except Exception:
            pass

        try:
            if hasattr(span, "end"):
                span.end()
        except Exception:
            pass

        try:
            if hasattr(trace, "end"):
                trace.end()
        except Exception:
            pass

        try:
            if hasattr(client, "flush"):
                client.flush()
        except Exception:
            pass
    except Exception:
        pass

# --- end Langfuse helpers ---


def _create_tool_from_schema(schema: dict) -> dict:
    """Convert a JSON schema to an Anthropic tool definition.

    Args:
        schema: JSON schema for structured output (may be nested with 
        format.schema)

    Returns:
        Anthropic tool definition
    """
    # Extract the actual schema if it's nested in a format structure
    if "format" in schema and "schema" in schema["format"]:
        actual_schema = schema["format"]["schema"]
    else:
        actual_schema = schema
    
    return {
        "name": "structured_output",
        "description": "Provide structured output according to the schema",
        "input_schema": actual_schema
    }


def _compress_image_if_needed(image_data: str, mime_type: str, 
                            max_size_bytes: int = 5 * 1024 * 1024) -> tuple[str, str]:
    """Compress image if it exceeds the maximum size.
    
    Args:
        image_data: Base64 encoded image data
        mime_type: MIME type of the image
        max_size_bytes: Maximum allowed size in bytes (default: 5MB)
        
    Returns:
        Tuple of (compressed_base64_data, mime_type)
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, skipping image compression")
        return image_data, mime_type
    
    try:
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Check if compression is needed by testing the actual base64 size
        # We need to ensure the base64-encoded result stays under the limit
        if len(image_data) <= max_size_bytes:
            return image_data, mime_type
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Determine output format based on MIME type
        if mime_type == "image/png":
            output_format = "PNG"
            # For PNG, try different optimization strategies
            compressed_bytes = io.BytesIO()
            image.save(compressed_bytes, format=output_format, optimize=True)
            compressed_data = compressed_bytes.getvalue()
            
            # If still too large, try converting to JPEG for better compression
            # Check the base64-encoded size, not the binary size
            compressed_base64 = base64.b64encode(compressed_data).decode()
            if len(compressed_base64) > max_size_bytes:
                logger.info(f"PNG too large ({len(compressed_data)} bytes), "
                           f"converting to JPEG for better compression")
                output_format = "JPEG"
                mime_type = "image/jpeg"
                quality = 85
                
                # Convert RGBA to RGB if needed for JPEG
                if image.mode == 'RGBA':
                    # Create a white background
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = rgb_image
            else:
                quality = None  # PNG doesn't use quality parameter
        elif mime_type == "image/jpeg" or mime_type == "image/jpg":
            output_format = "JPEG"
            quality = 85  # Start with high quality
        else:
            # Convert to JPEG for other formats
            output_format = "JPEG"
            quality = 85
            mime_type = "image/jpeg"
        
        # For JPEG or converted images, compress with quality reduction if needed
        if output_format == "JPEG":
            compressed_bytes = io.BytesIO()
            image.save(compressed_bytes, format=output_format, quality=quality, 
                      optimize=True)
            compressed_data = compressed_bytes.getvalue()
            
            # If still too large, reduce quality further
            # Check the base64-encoded size, not the binary size
            compressed_base64 = base64.b64encode(compressed_data).decode()
            if len(compressed_base64) > max_size_bytes:
                for quality_level in [70, 60, 50, 40, 30]:
                    compressed_bytes = io.BytesIO()
                    image.save(compressed_bytes, format=output_format, 
                              quality=quality_level, optimize=True)
                    compressed_data = compressed_bytes.getvalue()
                    compressed_base64 = base64.b64encode(compressed_data).decode()
                    if len(compressed_base64) <= max_size_bytes:
                        break
        
        # If still too large, resize the image
        if len(compressed_data) > max_size_bytes:
            # Calculate new size to fit within limit
            scale_factor = (max_size_bytes / len(compressed_data)) ** 0.5
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            resized_image = image.resize((new_width, new_height), 
                                       Image.Resampling.LANCZOS)
            compressed_bytes = io.BytesIO()
            resized_image.save(compressed_bytes, format=output_format, 
                             quality=30, optimize=True)
            compressed_data = compressed_bytes.getvalue()
        
        # Encode back to base64
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
        
        logger.info(f"Compressed image from {len(image_bytes)} to "
                   f"{len(compressed_data)} bytes")
        return compressed_base64, mime_type
        
    except Exception as e:
        logger.warning(f"Failed to compress image: {e}")
        return image_data, mime_type


def _convert_content_for_anthropic(content: list) -> list:
    """Convert content from OpenAI format to Anthropic format.
    
    Args:
        content: List of content items in OpenAI format
        
    Returns:
        List of content items in Anthropic format
    """
    converted_content = []
    
    for item in content:
        if not isinstance(item, dict) or "type" not in item:
            # Skip invalid items
            continue
            
        item_type = item["type"]
        
        if item_type == "input_text":
            # Convert input_text to text
            converted_content.append({
                "type": "text",
                "text": item["text"]
            })
        elif item_type == "input_image":
            # Convert input_image to image
            image_url = item["image_url"]
            
            # Extract MIME type and data from data URL
            if image_url.startswith("data:"):
                # Parse data URL: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
                parts = image_url.split(",", 1)
                if len(parts) == 2:
                    mime_part = parts[0]
                    data = parts[1]
                    
                    # Extract MIME type from "data:image/png;base64"
                    if ";" in mime_part:
                        mime_type = mime_part.split(";")[0].split(":", 1)[1]
                    else:
                        mime_type = mime_part.split(":", 1)[1]
                else:
                    # Fallback if parsing fails
                    mime_type = "image/jpeg"
                    data = image_url
            else:
                # Not a data URL, assume JPEG
                mime_type = "image/jpeg"
                data = image_url
            
            # Compress image if it's too large
            compressed_data, final_mime_type = _compress_image_if_needed(data, mime_type)
            
            converted_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": final_mime_type,
                    "data": compressed_data
                }
            })
        else:
            # Keep other types as-is (they might already be in correct format)
            converted_content.append(item)
    
    return converted_content


def get_openai_models() -> list:
    """Get list of available OpenAI models by querying the API.
    
    Returns:
        List of available OpenAI model names
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        logger.warning(f"Could not fetch OpenAI models: {e}")
        return []


def get_anthropic_models() -> list:
    """Get list of available Anthropic models by querying the API.
    
    Returns:
        List of available Anthropic model names
    """
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        logger.warning(f"Could not fetch Anthropic models: {e}")
        return []


def get_compatible_models(provider: str = None) -> list:
    """Get list of compatible models for the specified provider.
    
    Args:
        provider: The API provider (defaults to current API_PROVIDER)
        
    Returns:
        list: List of compatible model names
    """
    if provider is None:
        from soda_mmqc.config import API_PROVIDER
        provider = API_PROVIDER
    
    if provider == "openai":
        return get_openai_models()
    elif provider == "anthropic":
        return get_anthropic_models()
    else:
        return []


def validate_model_for_provider(model: str, provider: str = "") -> bool:
    """Validate that a model is compatible with the specified provider.
    
    Args:
        model: The model name to validate
        provider: The API provider to check against (defaults to current API_PROVIDER)
        
    Returns:
        bool: True if the model is compatible with the provider
    """
    if not provider:
        from soda_mmqc.config import API_PROVIDER
        provider = API_PROVIDER
    
    compatible_models = get_compatible_models(provider)
    return model in compatible_models


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    reraise=True
)
def _extract_output_text_from_response(raw_response) -> str:
    """Get the final assistant text from a Responses API response.
    
    When tools are used, output may contain multiple items (e.g. web_search_call
    then message). Prefer output_text; if empty, take the last message's
    output_text from output items.
    """
    if getattr(raw_response, "output_text", None):
        return raw_response.output_text
    output = getattr(raw_response, "output", None) or []
    for item in reversed(output):
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for block in content:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", None)
                if text:
                    return text
    return ""


def _sanitize_metadata_for_model(obj: Any, *, max_preview_rows: int = 3, max_string_len: int = 200) -> Any:
    """Recursively sanitize metadata so it's JSON-safe and compact for model input.

    - Converts NaN to None
    - Truncates previews to `max_preview_rows`
    - Shortens long strings
    - Replaces full file paths with basenames for brevity
    - Converts non-serializable objects to strings as a last resort
    """
    # Primitive types
    try:
        if obj is None:
            return None
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            # Convert NaN/Inf to None and convert numpy scalars to python native
            try:
                # If it's a numpy scalar, get python value
                if isinstance(obj, numbers.Number) and not isinstance(obj, bool):
                    try:
                        val = obj.item()
                    except Exception:
                        val = obj
                else:
                    val = obj
            except Exception:
                val = obj

            # If float-like, ensure it's finite (not NaN/inf)
            if isinstance(val, float):
                if not math.isfinite(val):
                    return None
            return val
        if isinstance(obj, str):
            if len(obj) > max_string_len:
                return obj[: max_string_len - 3] + "..."
            return obj

        # Lists
        if isinstance(obj, list):
            sanitized_list = [_sanitize_metadata_for_model(v, max_preview_rows=max_preview_rows, max_string_len=max_string_len) for v in obj]
            return sanitized_list

        # Dictionaries
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                # If a preview, truncate rows
                if k == "preview" and isinstance(v, list):
                    truncated = v[:max_preview_rows]
                    new[k] = [_sanitize_metadata_for_model(r, max_preview_rows=max_preview_rows, max_string_len=max_string_len) for r in truncated]
                    continue

                # If the key looks like a file path, shorten it
                if isinstance(v, str) and (k == "file" or "path" in k or "file" in k):
                    try:
                        new[k] = Path(v).name
                        continue
                    except Exception:
                        pass

                new[k] = _sanitize_metadata_for_model(v, max_preview_rows=max_preview_rows, max_string_len=max_string_len)
            return new

        # Fallback for other types: try to convert to native python or string
        # numpy types, pandas types, etc. often have .tolist() or .item()
        try:
            if hasattr(obj, "tolist"):
                return _sanitize_metadata_for_model(obj.tolist(), max_preview_rows=max_preview_rows, max_string_len=max_string_len)
            if hasattr(obj, "item"):
                return _sanitize_metadata_for_model(obj.item(), max_preview_rows=max_preview_rows, max_string_len=max_string_len)
        except Exception:
            pass

        # As a last resort, stringize
        return str(obj)
    except Exception:
        return str(obj)


def generate_response_openai(
    example,
    prompt: str,
    schema: dict,
    model: str,
    metadata: dict,
    model_config: Optional[Dict[str, Any]] = None,
    prompt_obj: Optional[Any] = None,
) -> Tuple[dict, dict]:
    """Generate response using OpenAI API with structured output.

    Args:
        example: The example to process
        prompt: The prompt to use
        schema: The schema for structured output
        model: The model to use
        metadata: Additional metadata for the API call
        model_config: Optional API options (e.g. tools, tool_choice,
            max_tool_calls, include). When tools is empty or absent, no
            tools are passed.

    Returns:
        Tuple of (parsed response, response metadata with response_id and
        model)
    """
    # Lazy-init Langfuse client and start per-call trace (best-effort)
    _lf_client = _get_langfuse_client()
    _lf_trace = None
    trace_name = _build_lf_trace_name("openai", model, metadata)
    try:
        if _lf_client:
            _lf_trace = _lf_start_trace(
                _lf_client,
                name=trace_name,
                metadata={
                    "model": model,
                    "doc_id": metadata.get("doc_id") if metadata else None,
                    "prompt_name": metadata.get("prompt_name") if metadata else None,
                    "sub_function": (metadata or {}).get("sub_function", "evaluate"),
                },
            )
            if _lf_trace:
                try:
                    try:
                        print(f"Langfuse: started trace '{trace_name}' for doc_id=", (metadata or {}).get("doc_id"))
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        _lf_trace = None
    # Import and initialize OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "OpenAI package not found. Install with: pip install openai"
        )
        raise

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare model input (supports multimodal content; model_config can enable source_data)
    model_input = example.prepare_model_input(prompt, model_config=model_config)

    # Attach a small preview to the trace if available
    try:
        if _lf_trace:
            preview = model_input.get("content", "") if isinstance(model_input, dict) else str(prompt)
            if isinstance(preview, str):
                preview = preview[:200]
            try:
                if hasattr(_lf_trace, "update"):
                    _lf_trace.update(input={"prompt_preview": preview, "metadata": (metadata or {})})
            except Exception:
                pass
    except Exception:
        pass

    # Build request kwargs
    create_kwargs = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are a scientific figure quality control expert."
                )
            },
            {
                "role": "user",
                "content": model_input["content"]
            }
        ],
        "text": schema,
        # We'll set metadata after coercing to API-safe strings below
    }
    # Convert metadata to API-safe format: the Responses API may expect
    # metadata values to be primitive/string values. Stringify complex
    # objects (lists/dicts) so the request doesn't fail with
    # "expected a string, but got an object" errors.
    metadata_for_api = {}
    if metadata:
        for k, v in metadata.items():
            # Keep strings as-is
            if isinstance(v, str):
                metadata_for_api[k] = v
                continue
            # Special-case large 'tables' metadata: produce a short summary
            if k == "tables":
                try:
                    parts = []
                    if isinstance(v, dict):
                        for panel, files in v.items():
                            # files is expected to be a list of file entries
                            if isinstance(files, list):
                                for fe in files:
                                    try:
                                        fname = Path(fe.get("file", "")).name if isinstance(fe, dict) else str(fe)
                                    except Exception:
                                        fname = str(fe)
                                    # Only include panel and file basename (no rows/cols)
                                    parts.append(f"{panel}:{fname}")
                    summary = "; ".join(parts)
                    # Truncate to API limit (keep margin)
                    if len(summary) > 500:
                        summary = summary[:497] + "..."
                    metadata_for_api[k] = summary
                    continue
                except Exception:
                    # Fallback to string conversion below
                    pass
            # For simple primitives, convert to string representation
            if isinstance(v, (int, float, bool)) or v is None:
                try:
                    metadata_for_api[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    metadata_for_api[k] = str(v)
                continue
            # For dicts/lists/complex objects, JSON-stringify
            try:
                s = json.dumps(v, ensure_ascii=False)
                # Truncate long strings to comply with API limits (512 chars)
                if isinstance(s, str) and len(s) > 512:
                    s = s[:509] + "..."
                metadata_for_api[k] = s
            except Exception:
                metadata_for_api[k] = str(v)
    else:
        metadata_for_api = {}

    create_kwargs["metadata"] = metadata_for_api

    # Embed sanitized metadata into the user messages so the model receives it textually.
    # Some API clients may not surface request-level `metadata` into the token stream,
    # so adding a user message with the JSON ensures the model can read and act on it.
    if metadata:
        try:
            sanitized = _sanitize_metadata_for_model(metadata)
            try:
                metadata_json = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
            except Exception as dump_exc:
                # Diagnostic: walk the sanitized structure to find problematic values
                def _find_problems(obj, path="root"):
                    problems = []
                    try:
                        if obj is None or isinstance(obj, (str, bool, int)):
                            return problems
                        if isinstance(obj, float):
                            if not math.isfinite(obj):
                                problems.append((path, type(obj).__name__, obj))
                            return problems
                        # numpy scalars / objects
                        try:
                            if hasattr(obj, "item"):
                                v = obj.item()
                                return _find_problems(v, path)
                            if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
                                v = obj.tolist()
                                return _find_problems(v, path)
                        except Exception:
                            pass

                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                problems.extend(_find_problems(v, f"{path}.{k}"))
                            return problems
                        if isinstance(obj, (list, tuple)):
                            for i, v in enumerate(obj):
                                problems.extend(_find_problems(v, f"{path}[{i}]"))
                            return problems
                        # If object isn't a basic serializable type, note it
                        try:
                            json.dumps(obj)
                        except Exception:
                            problems.append((path, type(obj).__name__, repr(obj)))
                        return problems
                    except Exception as _:
                        return [(path, type(obj).__name__, "inspection-failed")]

                problems = _find_problems(sanitized)
                if problems:
                    for p in problems[:10]:
                        logger.error(f"Metadata serialization problem at {p[0]}: {p[1]} -> {p[2]}")
                logger.warning(f"json.dumps failed for sanitized metadata: {dump_exc}")
                # Best-effort fallback: coerce to JSON-safe representation
                def _coerce_safe(obj):
                    if obj is None or isinstance(obj, (str, bool, int)):
                        return obj
                    if isinstance(obj, float):
                        if not math.isfinite(obj):
                            return None
                        return float(obj)
                    try:
                        if hasattr(obj, "item"):
                            return _coerce_safe(obj.item())
                        if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
                            return _coerce_safe(obj.tolist())
                    except Exception:
                        pass
                    if isinstance(obj, dict):
                        return {k: _coerce_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_coerce_safe(v) for v in obj]
                    # Fallback to string
                    return str(obj)

                safe_sanitized = _coerce_safe(sanitized)
                try:
                    metadata_json = json.dumps(safe_sanitized, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    # Last resort: stringify the whole metadata
                    metadata_json = json.dumps(str(metadata), ensure_ascii=False)

            metadata_message = {
                "role": "user",
                "content": (
                    "Please consult the following METADATA_JSON block for table and file metadata. "
                    "Use it when producing your structured output.\nMETADATA_JSON:\n" + metadata_json
                )
            }
            create_kwargs["input"].append(metadata_message)
        except Exception as e:
            logger.warning(f"Failed to sanitize/attach metadata: {e}")

    # Add optional tool/config from model_config. Valid tool types (OpenAI
    # Responses API) include: web_search_preview, web_search_preview_2025_03_11,
    # file_search, code_interpreter, function, mcp, image_generation, shell,
    # computer_use_preview, apply_patch, custom. Do not use web_search or
    # web_fetch (unsupported).
    if model_config:
        tools = model_config.get("tools")
        if tools and len(tools) > 0:
            create_kwargs["tools"] = tools
            if "tool_choice" in model_config:
                create_kwargs["tool_choice"] = model_config["tool_choice"]
            if "max_tool_calls" in model_config:
                create_kwargs["max_tool_calls"] = model_config["max_tool_calls"]
        if "include" in model_config and model_config["include"]:
            create_kwargs["include"] = model_config["include"]
        # Reasoning (gpt-5, o-series): effort "low"|"medium"|"high", optional summary
        if "reasoning" in model_config and model_config["reasoning"]:
            create_kwargs["reasoning"] = model_config["reasoning"]
        if "max_output_tokens" in model_config and model_config["max_output_tokens"] is not None:
            create_kwargs["max_output_tokens"] = model_config["max_output_tokens"]

    raw_response = client.responses.create(**create_kwargs)

    # Parse response (support both direct output_text and tool-augmented)
    output_text = _extract_output_text_from_response(raw_response)
    if not output_text:
        raise ValueError(
            "No output text in response (empty or missing output_text)"
        )
    try:
        response = json.loads(output_text)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing response: {str(e)}")
        raise
    
    # Extract response metadata, ID, model, and usage
    try:
        response_metadata = raw_response.metadata or {}
        response_id = raw_response.id or ""
        response_model = raw_response.model or ""
        
        # Add response_id and model to metadata
        response_metadata.update({
            "response_id": response_id,
            "model": response_model
        })
        
        # Add usage data if available
        if raw_response.usage:
            response_metadata["usage"] = {
                "input_tokens": raw_response.usage.input_tokens,
                "output_tokens": raw_response.usage.output_tokens
            }
        
        # Add status (equivalent to stop_reason in chat completions)
        if hasattr(raw_response, 'status'):
            response_metadata["status"] = raw_response.status
            
    except AttributeError:
        logger.error(f"Error getting metadata, ID, model, or usage: {raw_response}")
        response_metadata = {}
    # Log generation and finish trace (best-effort)
    try:
        if _lf_trace:
            try:
                input_text = model_input.get("content", "") if isinstance(model_input, dict) else str(prompt)
            except Exception:
                input_text = str(prompt)
            try:
                    try:
                        _lf_log_generation(
                            _lf_client,
                            _lf_trace,
                            span_name="openai-generation-span",
                            gen_name="openai-generation",
                            model=response_model or model,
                            input_text=input_text,
                            output_text=output_text,
                            prompt_obj=prompt_obj,
                        )
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    
    return response, response_metadata


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    reraise=True
)
def generate_response_anthropic(
    example,
    prompt: str,
    schema: dict,
    model: str,
    metadata: dict,
    model_config: Optional[Dict[str, Any]] = None,
    prompt_obj: Optional[Any] = None,
) -> Tuple[dict, dict]:
    """Generate response using Anthropic API with structured output via tools.

    Args:
        example: The example to process
        prompt: The prompt to use
        schema: The schema for structured output
        model: The model to use (e.g., 'claude-3-5-sonnet-20241022')
        metadata: Additional metadata for the API call
        model_config: Optional check-level config (e.g. source_data_files.enabled)

    Returns:
        Tuple of (parsed response, response metadata with response_id and
        model)
    """
    # Import and initialize Anthropic client
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error(
            "Anthropic package not found. Install with: pip install anthropic"
        )
        raise

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Prepare model input (supports multimodal content; model_config can enable source_data)
    model_input_content = example.prepare_model_input(prompt, model_config=model_config)

    # Convert content from OpenAI format to Anthropic format
    converted_content = _convert_content_for_anthropic(model_input_content["content"])

    # Create tool from schema
    tool = _create_tool_from_schema(schema)

    # Prepare messages for Anthropic API
    messages = [
        {
            "role": "user",
            "content": converted_content
        }
    ]

    # Call Anthropic API with tool use for structured output
    # Lazy-init Langfuse client and start per-call trace (best-effort)
    _lf_client = _get_langfuse_client()
    _lf_trace = None
    trace_name = _build_lf_trace_name("anthropic", model, metadata)
    try:
        if _lf_client:
            _lf_trace = _lf_start_trace(
                _lf_client,
                name=trace_name,
                metadata={
                    "model": model,
                    "doc_id": metadata.get("doc_id") if metadata else None,
                    "prompt_name": metadata.get("prompt_name") if metadata else None,
                    "sub_function": (metadata or {}).get("sub_function", "evaluate"),
                },
            )
            if _lf_trace:
                try:
                    # Try to attach a small request preview/update
                    try:
                        if hasattr(_lf_trace, "update"):
                            _lf_trace.update(input={"messages_preview": str(messages)[:200]})
                    except Exception:
                        pass
                    try:
                        print(f"Langfuse: started trace '{trace_name}' for doc_id=", (metadata or {}).get("doc_id"))
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        _lf_trace = None

    raw_response = client.messages.create(  # type: ignore
        model=model,
        max_tokens=4096,
        system=(
            "You are a scientific figure quality control expert. "
            "You must use the structured_output tool to provide your "
            "response in the required format."
        ),
        messages=messages,  # type: ignore
        tools=[tool],  # type: ignore
        tool_choice={"type": "tool", "name": "structured_output"}
    )

    # Extract structured output from tool use
    if not raw_response.content or len(raw_response.content) == 0:
        raise ValueError("No content in response")

    # Find the tool use block
    tool_use_block = None
    for content_block in raw_response.content:
        if hasattr(content_block, 'type') and content_block.type == 'tool_use':
            tool_use_block = content_block
            break

    if not tool_use_block:
        raise ValueError("No tool use block found in response")

    # Get the structured output
    response = dict(tool_use_block.input)  # type: ignore

    # Create response metadata
    response_metadata = {
        "response_id": raw_response.id,
        "model": raw_response.model,
        "usage": {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
        }
    }

    # Add custom metadata if provided
    response_metadata.update(metadata)

    # Log response and finish trace if any
    try:
        if _lf_trace:
            try:
                # Serialize inputs/outputs conservatively for the trace
                try:
                    input_text = json.dumps(messages, ensure_ascii=False)
                except Exception:
                    input_text = str(messages)
                try:
                    output_text = json.dumps(response, ensure_ascii=False)
                except Exception:
                    output_text = str(response)
                try:
                    _lf_log_generation(
                        _lf_client,
                        _lf_trace,
                        span_name="anthropic-generation-span",
                        gen_name="anthropic-generation",
                        model=model,
                        input_text=input_text,
                        output_text=output_text,
                        prompt_obj=prompt_obj,
                    )
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    return response, response_metadata


def generate_response(
    model_input,
    model: str = "",
    metadata: Dict[str, Any] = None,  # type: ignore
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[dict, dict]:
    """Generate response using the configured API provider.

    Supports both OpenAI and Anthropic APIs with structured output.

    Args:
        model_input: ModelInput object containing:
            - example: The example to process
            - prompt: The prompt to use
            - schema: The schema for structured output
        model: The model to use (provider-specific)
        metadata: Additional metadata for the API call
        model_config: Optional API options (e.g. tools for OpenAI). Used when
            provider is OpenAI; ignored for Anthropic for now.

    Returns:
        Tuple of (parsed response, response metadata with response_id and
        model)
    """
    # Extract inputs from ModelInput object
    example = model_input.example
    prompt = model_input.prompt
    schema = model_input.schema

    # Handle default metadata
    if metadata is None:
        metadata = {}

    # Set default model based on provider configuration
    if not model:
        model = DEFAULT_MODEL

    # Route to appropriate API
    prompt_obj = getattr(model_input, "prompt_obj", None)
    if API_PROVIDER == "anthropic":
        return generate_response_anthropic(
            example=example,
            prompt=prompt,
            schema=schema,
            model=model,
            metadata=metadata,
            model_config=model_config,
            prompt_obj=prompt_obj,
        )
    else:
        return generate_response_openai(
            example=example,
            prompt=prompt,
            schema=schema,
            model=model,
            metadata=metadata,
            model_config=model_config,
            prompt_obj=prompt_obj,
        )
