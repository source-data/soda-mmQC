import os
import json
import base64
import io
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Dict, Any, Tuple
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
def generate_response_openai(
    example,
    prompt: str,
    schema: dict,
    model: str,
    metadata: dict
) -> Tuple[dict, dict]:
    """Generate response using OpenAI API with structured output.

    Args:
        example: The example to process
        prompt: The prompt to use
        schema: The schema for structured output
        model: The model to use
        metadata: Additional metadata for the API call

    Returns:
        Tuple of (parsed response, response metadata with response_id and
        model)
    """
    # Import and initialize OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "OpenAI package not found. Install with: pip install openai"
        )
        raise

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare model input (supports multimodal content)
    model_input = example.prepare_model_input(prompt)

    # Call API with structured output
    raw_response = client.responses.create(
        model=model,
        input=[
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
        text=schema,
        metadata=metadata
    )
    # Parse response
    try:
        response = json.loads(raw_response.output_text)
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
    metadata: dict
) -> Tuple[dict, dict]:
    """Generate response using Anthropic API with structured output via tools.

    Args:
        example: The example to process
        prompt: The prompt to use
        schema: The schema for structured output
        model: The model to use (e.g., 'claude-3-5-sonnet-20241022')
        metadata: Additional metadata for the API call

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

    # Prepare model input (supports multimodal content)
    model_input_content = example.prepare_model_input(prompt)

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

    return response, response_metadata


def generate_response(
    model_input,
    model: str = "",
    metadata: Dict[str, Any] = None  # type: ignore
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
    if API_PROVIDER == "anthropic":
        return generate_response_anthropic(
            example=example,
            prompt=prompt,
            schema=schema,
            model=model,
            metadata=metadata
        )
    else:
        return generate_response_openai(
            example=example,
            prompt=prompt,
            schema=schema,
            model=model,
            metadata=metadata
        )
