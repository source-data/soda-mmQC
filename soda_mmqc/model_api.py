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
    encoded_image: str,
    mime_type: str,
    caption: str,
    prompt: str,
    schema: dict,
    model: str,
    metadata: dict
) -> dict:
    """Generate response using OpenAI API with structured output."""

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{prompt}\n\nFigure Caption:\n{caption}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded_image}"
                    }
                ]
            }
        ],
        text=schema,
        metadata=metadata
    )
    # Extract and return response
    try:
        content = raw_response.output_text
        if content is None:
            raise ValueError("API returned empty response")
        parsed_response = json.loads(content)
        # Format the response with proper indentation
        return parsed_response, raw_response
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {raw_response}")
        raise ValueError(f"Invalid JSON response from API: {str(e)}")


def generate_response(
    model_input,
    model="gpt-4o-2024-08-06",
    metadata={}
):
    """Generate response using the selected API provider.
    
    Args:
        model_input (dict): Dictionary containing image_path, caption, prompt, and 
            schema
        model (str): The model to use for generation. Defaults to 
            "gpt-4o-2024-08-06"
    """
    # Extract inputs
    try:
        image_path = model_input["image_path"]
        caption = model_input["caption"]
        prompt = model_input["prompt"]
        schema = model_input["schema"]
    except KeyError as e:
        logger.error(
            f"Missing required key in model_input: {e}. "
            f"Available keys: {list(model_input.keys())}"
        )
        raise

    # Get the correct MIME type
    try:
        mime_type = get_image_mime_type(image_path)
    except Exception as e:
        logger.error(
            f"Error getting MIME type for {image_path}: {str(e)}"
        )
        raise

    # Encode image (both APIs use base64 encoding)
    try:
        encoded_image = encode_image(image_path)
    except Exception as e:
        logger.error(
            f"Error encoding image at {image_path}: {str(e)}"
        )
        raise

    # Call the appropriate API provider
    try:
        if API_PROVIDER == "openai":
            return generate_response_openai(
                encoded_image, mime_type, caption, prompt, schema, model, metadata
            )
        else:
            raise ValueError(f"Unsupported API provider: {API_PROVIDER}")
    except Exception as e:
        logger.error(
            f"Error calling {API_PROVIDER} API: {str(e)}"
        )
        raise