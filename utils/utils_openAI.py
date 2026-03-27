import os
from typing import List, Dict
from openai import OpenAI


def get_openai_key():
    """
    Get the OpenAI API key from environment variables.
    """
    return os.environ.get("OPENAI_API_KEY", "")


def get_client(openai_api_key=None):
    """
    Initialize OpenAI client
    """
    if openai_api_key is None:
        openai_api_key = get_openai_key()

    if not openai_api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables")

    return OpenAI(api_key=openai_api_key)


def _convert_messages_to_input(messages: List[Dict]):
    """
    Convert OpenAI chat format -> responses API format
    """
    formatted = []

    for m in messages:
        formatted.append({
            "role": m["role"],
            "content": [
                {"type": "text", "text": m["content"]}
            ]
        })

    return formatted


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "gpt-4.1-mini",
    openai_api_key=None,
    **kwargs
):

    client = get_client(openai_api_key)

    messages = [{"role": role, "content": prompt}]
    formatted_input = _convert_messages_to_input(messages)

    payload = {
        "model": model,
        "input": formatted_input,
        "max_output_tokens": max_tokens,
        **kwargs
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if top_p is not None:
        payload["top_p"] = top_p

    try:
        response = client.responses.create(**payload)

        output_dict = {
            "role": "assistant",
            "content": response.output_text
        }

    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

    return output_dict


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "gpt-4.1-mini",
    openai_api_key=None,
    **kwargs
):

    client = get_client(openai_api_key)

    formatted_input = _convert_messages_to_input(messages)

    payload = {
        "model": model,
        "input": formatted_input,
        "max_output_tokens": max_tokens,
        **kwargs
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if top_p is not None:
        payload["top_p"] = top_p

    try:
        response = client.responses.create(**payload)

        output_dict = {
            "role": "assistant",
            "content": response.output_text
        }

    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

    return output_dict