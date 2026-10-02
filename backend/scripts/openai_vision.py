"""OpenAI-compatible vision endpoint — minimal sync client, manual probe.

Run directly:  python backend/scripts/openai_vision.py path/to/image.png

Credentials come from .env at the PhysicsLENS root (OPENAI_API_KEY,
OPENAI_BASE_URL — the latter optional, omit to hit api.openai.com) via
python-dotenv — never hardcode the key. (tools/llm_api.py is the async
counterpart used by the pipelines.)
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def query_vision(query: str, image_path: str, *,
                  model: str = "gpt-4o-mini",
                  system_prompt: str = None, response_format: dict = None,
                  api_key: str = None, base_url: str = None) -> str:
    import base64
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)  # None -> read from env

    image_bytes = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".") or "png"
    encoded_image = f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": query},
        {"type": "image_url", "image_url": {"url": encoded_image}},
    ]})

    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/openai_vision.py <image_path>")
        sys.exit(1)
    result = query_vision(
        "What is shown in this image?",
        sys.argv[1],
        system_prompt="You are a helpful assistant that describes images.",
    )
    print(result)
