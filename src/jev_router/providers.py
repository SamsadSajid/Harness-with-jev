"""OpenAI-compatible downstream execution through a single OpenRouter key."""

import json
import urllib.request
from typing import Optional

from .models import ModelSpec


def run_openrouter(model: ModelSpec, prompt: str, api_key: str, system: Optional[str] = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model.provider_model, "messages": messages}).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"]
