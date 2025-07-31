import os
import json
from dotenv import load_dotenv
from openai import OpenAI
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
        Tuple of (parsed response, response metadata with response_id and model)
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
    
    # Extract response metadata, ID, and model
    try:
        response_metadata = raw_response.metadata or {}
        response_id = raw_response.id or ""
        response_model = raw_response.model or ""
        
        # Add response_id and model to metadata
        response_metadata.update({
            "response_id": response_id,
            "model": response_model
        })
    except AttributeError:
        logger.error(f"Error getting metadata, ID, or model: {raw_response}")
        response_metadata = {}
    
    return response, response_metadata


def generate_response(
    model_input,
    model: str = "gpt-4o-2024-08-06",
    metadata: Dict[str, Any] = {}
) -> Tuple[dict, dict]:
    """Generate response using the specified model.
    
    Args:
        model_input: ModelInput object containing:
            - example: The example to process
            - prompt: The prompt to use
            - schema: The schema for structured output
        model: The model to use
        metadata: Additional metadata for the API call
        
    Returns:
        Tuple of (parsed response, response metadata with response_id and model)
    """

    # Extract inputs from ModelInput object
    example = model_input.example
    prompt = model_input.prompt
    schema = model_input.schema

    # Generate response
    return generate_response_openai(
        example=example,
        prompt=prompt,
        schema=schema,
        model=model,
        metadata=metadata
    )
