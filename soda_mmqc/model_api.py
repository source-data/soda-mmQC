import os
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine which API to use
API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_response_openai(encoded_image, caption, prompt):
    """Generate response using OpenAI API."""
    import openai
    
    # Set API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are a scientific figure quality control expert."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}\n\nFigure Caption:\n{caption}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    
    # Call API
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1000
    )
    
    # Extract and return response
    return response.choices[0].message.content


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
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=messages
    )
    
    # Extract and return response
    return response.content[0].text


def generate_response(model_input):
    """Generate response using the selected API provider."""
    # Extract inputs
    image_path = model_input["image"]
    caption = model_input["caption"]
    prompt = model_input["prompt"]
    
    # Encode image (both APIs use base64 encoding)
    encoded_image = encode_image(image_path)
    
    # Call the appropriate API provider
    if API_PROVIDER == "openai":
        return generate_response_openai(encoded_image, caption, prompt)
    elif API_PROVIDER == "anthropic":
        return generate_response_anthropic(encoded_image, caption, prompt)
    else:
        raise ValueError(f"Unsupported API provider: {API_PROVIDER}") 