"""LLM client for compile agents (OpenAI-compatible HTTP, or local vLLM)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
CLOUD_HOSTS = (
    "api.openai.com",
    "openai.com",
    "api.deepseek.com",
    "api.anthropic.com",
    "openrouter.ai",
    "googleapis.com",
    "api.moonshot.cn",
)

LLM_DOWN_EXIT = 78


class LLMUnreachable(RuntimeError):
    """Local endpoint is down. Do not fall back to a cloud API; abort the suite."""


def local_only() -> bool:
    return os.environ.get("GKO_LLM_LOCAL", "").strip() not in ("", "0", "false", "False")


def _is_cloud_url(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in CLOUD_HOSTS)


def _is_down_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (502, 503, 504):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError, TimeoutError)):
            return True
        msg = str(reason or exc).lower()
        needles = (
            "connection refused",
            "connection reset",
            "name or service not known",
            "nodename nor servname",
            "network is unreachable",
            "timed out",
            "temporary failure in name resolution",
        )
        return any(n in msg for n in needles)
    msg = str(exc).lower()
    return "connection refused" in msg or "timed out" in msg


def parse_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    end = text.rfind("}")
    while end > 0:
        start = text.rfind("{", 0, end)
        if start < 0:
            break
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        end = text.rfind("}", 0, end)
    return None


def _message_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("text"):
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        content = "\n".join(parts)
    if content:
        return str(content)
    for k in ("reasoning_content", "reasoning"):
        v = msg.get(k)
        if v:
            return str(v)
    return ""


class LLMClient:
    def __init__(
        self,
        *,
        model: str = "default",
        server_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.server_url = server_url.rstrip("/")
        if not self.server_url.endswith("/v1"):
            self.server_url = self.server_url + "/v1"
        self.local_only = local_only()
        if self.local_only and _is_cloud_url(self.server_url):
            raise LLMUnreachable(
                f"GKO_LLM_LOCAL=1 forbids cloud URL {self.server_url}. "
                "Refusing to call a paid API."
            )
        if self.local_only:
            self.api_key = api_key or os.environ.get("GKO_LLM_API_KEY") or ""
        else:
            self.api_key = (
                api_key
                or os.environ.get("GKO_LLM_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
        self.temperature = temperature
        self.max_tokens = int(os.environ.get("GKO_LLM_MAX_TOKENS", str(max_tokens)))
        self.timeout_s = float(os.environ.get("GKO_LLM_TIMEOUT", "600"))
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        # localhost / 127.0.0.1 vLLM and SGLang are still OpenAI-compatible HTTP.
        # An empty API key must not switch to agent.sever.llm_local (that imports
        # the openai package, which the eval image does not have).
        self._remote = True

    def usage_dict(self) -> dict:
        return dict(self.usage)

    def _record_usage(self, body: dict) -> None:
        u = body.get("usage") or {}
        self.usage["calls"] += 1
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        tt = int(u.get("total_tokens") or (pt + ct))
        self.usage["prompt_tokens"] += pt
        self.usage["completion_tokens"] += ct
        self.usage["total_tokens"] += tt
        details = u.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is None:
            cached = u.get("prompt_cache_hit_tokens")
        if cached is not None:
            self.usage["cached_tokens"] += int(cached)

    def chat(self, system: str, user: str) -> str:
        return self._chat_http(system, user)

    def _chat_http(self, system: str, user: str) -> str:
        url = self.server_url + "/chat/completions"
        model = (self.model or "").lower()
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # GPT-5.x chat completions: max_completion_tokens; temperature often unsupported.
        if model.startswith("gpt-5") or "gpt-5." in model:
            payload["max_completion_tokens"] = max(int(self.max_tokens), 4096)
            # Sol is the flagship; Luna/Terra stay low-effort.
            payload["reasoning_effort"] = "medium" if "sol" in model else "low"
        else:
            payload["temperature"] = self.temperature
            payload["max_tokens"] = self.max_tokens
        # V4-Flash thinking-on-by-default can dump chain-of-thought instead of JSON.
        if "deepseek" in model:
            payload["thinking"] = {"type": "disabled"}
        # Local Qwen3 / Kimi-K2: disable thinking so the reply is recipe JSON, not CoT.
        if "qwen" in model or "kimi" in model:
            payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            payload["separate_reasoning"] = False
            payload["enable_thinking"] = False
            payload["thinking"] = False
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        last_err = ""
        attempts = 2 if self.local_only else 6
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError(f"empty choices: {body!r}"[:800])
                text = _message_text(choices[0].get("message") or {})
                if not text:
                    raise RuntimeError(f"empty message: {body!r}"[:800])
                self._record_usage(body)
                return text
            except urllib.error.HTTPError as e:
                last_err = e.read().decode("utf-8", errors="replace")[:800]
                code = e.code
                if self.local_only and code in (502, 503, 504) and attempt + 1 >= attempts:
                    raise LLMUnreachable(
                        f"local llm HTTP {code} at {self.server_url}: {last_err}"
                    ) from e
                if code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                    print(f"  llm HTTP {code}, retry {attempt+1} in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"llm HTTP {code}: {last_err}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, ConnectionResetError, json.JSONDecodeError) as e:
                last_err = str(e)
                if self.local_only and _is_down_error(e):
                    if attempt + 1 < attempts and not isinstance(e, ConnectionRefusedError):
                        print(f"  llm {type(e).__name__}, retry {attempt+1} in {delay:.0f}s", flush=True)
                        time.sleep(delay)
                        delay = min(delay * 2, 10)
                        continue
                    raise LLMUnreachable(
                        f"local llm unreachable at {self.server_url}: {last_err}"
                    ) from e
                if attempt + 1 < attempts:
                    print(f"  llm {type(e).__name__}, retry {attempt+1} in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"llm request failed: {last_err}") from e
        raise RuntimeError(f"llm request failed: {last_err}")
