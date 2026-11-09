import os
import re
import requests
from typing import Dict, Optional

class LocalLLM:
    _ENV_PATTERN = re.compile(r"^\$\{([^}]+)\}$")
    _DOTENV_LOADED = False

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        provider: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.provider = (provider or self._infer_provider(self.base_url)).lower()
        self.api_key = self._resolve_api_key(api_key, self.provider)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    @staticmethod
    def _infer_provider(base_url: str) -> str:
        if "generativelanguage.googleapis.com" in base_url:
            return "gemini"
        elif "luxiacloud.com" in base_url:
            return "luxia"
        return "openai"

    @classmethod
    def _load_dotenv(cls, path: str = ".env") -> None:
        if cls._DOTENV_LOADED:
            return
        cls._DOTENV_LOADED = True
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    key, value = s.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"").strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except FileNotFoundError:
            return

    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str], provider: str) -> str:
        key = (api_key or "").strip()
        env_var = None

        match = cls._ENV_PATTERN.match(key)
        if match:
            env_var = match.group(1)

        if not key or key.lower() in {"dummy-api-key", "your_gemini_api_key", "your_openai_api_key", "your_luxia_api_key"}:
            if not env_var:
                if provider == "gemini":
                    env_var = "GEMINI_API_KEY"
                elif provider == "luxia":
                    env_var = "LUXIA_API_KEY"
                else:
                    env_var = "OPENAI_API_KEY"

        if env_var:
            value = os.environ.get(env_var)
            if not value:
                cls._load_dotenv()
                value = os.environ.get(env_var)
            if value:
                return value

        return key

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "gemini":
            return self._chat_gemini(system_prompt, user_prompt)
        elif self.provider == "luxia":
            return self._chat_luxia(system_prompt, user_prompt)
        return self._chat_openai(system_prompt, user_prompt)

    def _chat_openai(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: Dict = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _chat_luxia(self, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url
        if not url.endswith("/create"):
            url = f"{url}/create"
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        payload: Dict = {
            "model": "llm",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _chat_gemini(self, system_prompt: str, user_prompt: str) -> str:
        base_url = self.base_url
        if not (base_url.endswith("/v1") or base_url.endswith("/v1beta")):
            base_url = f"{base_url}/v1beta"

        model = self.model
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"{base_url}/{model}:generateContent?key={self.api_key}"
        payload: Dict = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
