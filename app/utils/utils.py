import os
import json
import numpy as np
from dotenv import load_dotenv
import requests

load_dotenv()

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llava:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

def ollama_generate(prompt: str, model: str = None, temperature: float = 0.2) -> str:
    model = model or LLM_MODEL
    url = f"{OLLAMA}/api/generate"
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False  # important
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "") or ""

def ollama_embed(texts):
    url = f"{OLLAMA}/api/embeddings"
    resp = requests.post(url, json={"model": EMBED_MODEL, "input": texts})
    resp.raise_for_status()
    data = resp.json()
    # supports single or batch
    if isinstance(texts, str):
        return np.array(data["embedding"], dtype=np.float32)
    return np.array([d["embedding"] for d in data["embeddings"]], dtype=np.float32)

def safe_json(s: str, fallback: dict):
    try:
        # heuristics to find JSON inside
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end >= 0:
            return json.loads(s[start:end+1])
        return fallback
    except Exception:
        return fallback
