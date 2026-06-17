import json, os, urllib.request
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
def complete(prompt, system="", temperature=0.0):
    if LLM_BACKEND == "ollama":
        return _ollama(prompt, system, temperature)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
def embed(text):
    if LLM_BACKEND == "ollama":
        return _ollama_embed(text)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")
def _ollama(prompt, system, temperature):
    payload = {"model": OLLAMA_MODEL,
        "messages": (([{"role":"system","content":system}] if system else []) + [{"role":"user","content":prompt}]),
        "stream": False, "options": {"temperature": temperature}}
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["message"]["content"]
def _ollama_embed(text):
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["embedding"]
if __name__ == "__main__":
    print(complete("Reply with exactly the word: ready"))
    print(f"embed dim: {len(embed('insertion loss 1.1 dB exceeds spec'))}")
