import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()

# Determine which API to use
API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()

# Set up logger
logger = logging.getLogger(__name__)


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_response_openai(encoded_image, caption, prompt):
    """Generate response using OpenAI API with structured output."""
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Call API with structured output
    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are a scientific figure quality control expert."
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
                        "image_url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "figure_quality_check",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the check being performed"
                        },
                        "panels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "panel_label": {
                                        "type": "string",
                                        "description": "Label of the panel (e.g., A, B, C)"
                                    },
                                    "error_bar_on_figure": {
                                        "type": "string",
                                        "enum": ["yes", "no"],
                                        "description": "Whether error bars are present"
                                    },
                                    "error_bar_defined_in_legend": {
                                        "type": "string",
                                        "enum": ["yes", "no", "not needed"],
                                        "description": "Whether error bars are defined in legend"
                                    },
                                    "error_bar_defined_in_caption": {
                                        "type": "string",
                                        "enum": ["yes", "no", "not needed"],
                                        "description": "Whether error bars are defined in caption"
                                    },
                                    "error_bar_meaning": {
                                        "type": "string",
                                        "enum": [
                                            "standard deviation",
                                            "standard error",
                                            "confidence interval",
                                            "not applicable"
                                        ],
                                        "description": "What the error bars represent"
                                    },
                                    "from_the_caption": {
                                        "type": "string",
                                        "description": "Text from caption describing error bars"
                                    }
                                },
                                "required": [
                                    "panel_label",
                                    "error_bar_on_figure",
                                    "error_bar_defined_in_legend",
                                    "error_bar_defined_in_caption",
                                    "error_bar_meaning",
                                    "from_the_caption"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["name", "panels"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    
    # Extract and return response
    return response.output_text


def generate_response_anthropic(encoded_image, caption, prompt):
    """Generate response using Anthropic API."""
    import anthropic
    
    # Set API key
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}\n\nFigure Caption:\n{caption}"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded_image
                    }
                }
            ]
        }
    ]
    
    # Call API
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=messages
    )
    
    # Extract and return response
    return response.content[0].text


def generate_response(model_input):
    """Generate response using the selected API provider."""
    # Extract inputs
    try:
        image_path = model_input["image"]
        caption = model_input["caption"]
        prompt = model_input["prompt"]
    except KeyError as e:
        logger.error(
            f"Missing required key in model_input: {e}. "
            f"Available keys: {list(model_input.keys())}"
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
            return generate_response_openai(encoded_image, caption, prompt)
        elif API_PROVIDER == "anthropic":
            return generate_response_anthropic(encoded_image, caption, prompt)
        else:
            raise ValueError(f"Unsupported API provider: {API_PROVIDER}")
    except Exception as e:
        logger.error(
            f"Error calling {API_PROVIDER} API: {str(e)}"
        )
        raise 