import os
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI
import logging
import mimetypes
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Dict, Any, Tuple
from soda_mmqc.examples import Example
from soda_mmqc import logger

# Load environment variables
load_dotenv()

# Determine which API to use
API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()

# Set up logger
logger = logging.getLogger(__name__)


def get_image_mime_type(image_path):
    """Get the MIME type of an image file based on its extension."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        if mime_type.startswith('image/'):
            return mime_type
        else:
            raise ValueError(f"Not an image: {image_path}")
    else:
        # Default to JPEG if we can't determine the type
        raise ValueError(f"Could not guess mime type: {image_path}")


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    reraise=True
)
def generate_response_openai(
    example: Example,
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
        Tuple of (parsed response, raw response)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare model input
    model_input = example.prepare_model_input(prompt)

    # Call API with structured output
    raw_response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a scientific figure quality control expert."
            },
            {
                "role": "user",
                "content": model_input["content"]
            }
        ],
        text=schema,
        metadata={**metadata, **model_input["metadata"]}
    )

    # Parse response
    try:
        response = json.loads(raw_response.text)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing response: {str(e)}")
        raise

    return response, raw_response


def generate_response(
    model_input: Dict[str, Any],
    model: str = "gpt-4o-2024-08-06",
    metadata: Dict[str, Any] = None
) -> Tuple[dict, dict]:
    """Generate response using the specified model.
    
    Args:
        model_input: Dictionary containing:
            - example: The example to process
            - prompt: The prompt to use
            - schema: The schema for structured output
        model: The model to use
        metadata: Additional metadata for the API call
        
    Returns:
        Tuple of (parsed response, raw response)
    """
    if metadata is None:
        metadata = {}

    # Extract inputs
    example = model_input["example"]
    prompt = model_input["prompt"]
    schema = model_input["schema"]

    # Generate response
    return generate_response_openai(
        example=example,
        prompt=prompt,
        schema=schema,
        model=model,
        metadata=metadata
    )