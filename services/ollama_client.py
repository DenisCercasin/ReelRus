"""One local Ollama/Qwen integration shared by translation, summary, and Q&A."""

import json
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", model="qwen3:8b", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages, json_mode=False):
        payload = {"model": self.model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        request = Request(self.base_url + "/api/chat",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return result["message"]["content"].strip()
