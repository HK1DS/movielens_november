import os
import re
import threading
from typing import Dict, Optional

import requests

from .key_pool import AllKeysExhausted, KeyPool, classify_failure, collect_keys, get_pool  # noqa: F401


class LLMResponseError(RuntimeError):
    """HTTP 200 whose body is not a completion (e.g. a JSON error object)."""

    def __init__(self, message, body=""):
        super().__init__(message)
        self.body = body


def _snippet(text, n=200):
    return " ".join(str(text or "").split())[:n]


def _body_text(resp):
    if resp is None:
        return ""
    try:
        return resp.text[:2000]
    except Exception:
        return ""


class LocalLLM:
    """Chat client for Luxia / OpenAI-compatible / Gemini endpoints.

    API keys: `api_key` may be a literal key or "${ENV_VAR}". For an env var, every key
    configured as ENV_VAR, ENV_VAR_1, ENV_VAR_2, ... or ENV_VAR + "S" (comma separated)
    in the environment or .env is used, in that order; when a key runs out of credit
    the next one is used automatically (ikgr_core/key_pool.py). When every key is
    exhausted, chat() raises AllKeysExhausted.
    """
    _ENV_PATTERN = re.compile(r"^\$\{([^}]+)\}$")
    _DOTENV_LOADED = False
    _SEEN_ERRORS = set()
    _SEEN_LOCK = threading.Lock()

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
        self.keys = self._resolve_key_pool(api_key, self.provider)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        """The key currently in use (kept for backward compatibility)."""
        return self.keys.acquire()[1]

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
    def _resolve_key_pool(cls, api_key: Optional[str], provider: str) -> KeyPool:
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
            cls._load_dotenv()
            keys = collect_keys(env_var)
            if not keys:
                raise ValueError(f"no API key found: set {env_var} (or {env_var}_1, {env_var}_2, ... "
                                 f"or {env_var}S=key1,key2) in the environment or .env")
            return get_pool(env_var, keys)
        return get_pool("config", [key])

    def _log_error_once(self, status, body, kind):
        sig = (status, _snippet(body, 120))
        with self._SEEN_LOCK:
            if sig in self._SEEN_ERRORS:
                return
            self._SEEN_ERRORS.add(sig)
        print(f"\n[llm] {self.provider} HTTP {status} ({kind}): {_snippet(body) or '<empty body>'}", flush=True)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        while True:
            idx, key = self.keys.acquire()          # raises AllKeysExhausted
            try:
                if self.provider == "gemini":
                    out = self._chat_gemini(system_prompt, user_prompt, key)
                elif self.provider == "luxia":
                    out = self._chat_luxia(system_prompt, user_prompt, key)
                else:
                    out = self._chat_openai(system_prompt, user_prompt, key)
            except requests.exceptions.HTTPError as e:
                resp = e.response
                status = resp.status_code if resp is not None else None
                body = _body_text(resp)
                kind = classify_failure(status, body)
                self._log_error_once(status, body, kind)
                if kind == "exhausted":
                    self.keys.mark_exhausted(idx, f"HTTP {status}: {_snippet(body, 120) or 'no body'}")
                    continue                          # same request, next key
                if kind == "rate_limited":
                    self.keys.note_rate_limited(idx)  # rotates only after a long 429-only streak
                raise
            except LLMResponseError as e:
                kind = classify_failure(200, e.body)
                self._log_error_once(200, e.body, kind)
                if kind == "exhausted":
                    self.keys.mark_exhausted(idx, f"HTTP 200 error body: {_snippet(e.body, 120)}")
                    continue
                raise
            self.keys.note_success(idx)
            return out

    @staticmethod
    def _json(r):
        try:
            return r.json()
        except ValueError:
            raise LLMResponseError(f"non-JSON response: {_snippet(r.text)}", r.text[:2000])

    @staticmethod
    def _openai_content(data, r):
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMResponseError(f"response without choices: {_snippet(r.text)}", r.text[:2000])

    def _chat_openai(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
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
        return self._openai_content(self._json(r), r)

    def _chat_luxia(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        url = self.base_url
        if not url.endswith("/create"):
            url = f"{url}/create"
        headers = {
            "apikey": api_key,
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
        return self._openai_content(self._json(r), r)

    def _chat_gemini(self, system_prompt: str, user_prompt: str, api_key: str) -> str:
        base_url = self.base_url
        if not (base_url.endswith("/v1") or base_url.endswith("/v1beta")):
            base_url = f"{base_url}/v1beta"

        model = self.model
        if not model.startswith("models/"):
            model = f"models/{model}"

        # key in a header, not the URL: requests puts the URL into exception messages/logs
        url = f"{base_url}/{model}:generateContent"
        headers = {"x-goog-api-key": api_key}
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

        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = self._json(r)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise LLMResponseError(f"response without candidates: {_snippet(r.text)}", r.text[:2000])
