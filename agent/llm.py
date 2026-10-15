"""LLM 客户端：OpenAI 兼容 + iChat(HMAC) 双适配器。

从 agent/.env 读取配置（模板见 agent/.env.template）。
用法：
    from agent.llm import get_client
    client = get_client()
    text = client.chat(messages, temperature=0.3, max_tokens=4096)
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from . import net

AGENT_DIR = Path(__file__).resolve().parent


def load_env() -> dict:
    """读取 agent/.env，兼容 export、引号和行尾注释。"""
    return net.read_env(AGENT_DIR / ".env")


class OpenAICompatClient:
    def __init__(self, api_key: str, base_url: str, model: str, proxy: str | None = None):
        # proxy 参数保留兼容签名，实际由 agent/net 直连->代理回退接管
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, temperature=0.3, max_tokens=4096, timeout=None) -> str:
        r = net.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=net.request_timeout(timeout))
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        self.last_usage = {"prompt_tokens": int(u.get("prompt_tokens", 0)),
                           "completion_tokens": int(u.get("completion_tokens", 0)),
                           "total_tokens": int(u.get("total_tokens", 0))}
        return data["choices"][0]["message"]["content"]


class IChatClient:
    """内网 iChat 网关（HMAC 签名）。AGENT_BASE_URL 为完整 chat_completions URL。"""

    def __init__(self, app_id: str, app_key: str, source: str,
                 endpoint: str, model: str, proxy: str | None = None):
        self.app_id, self.app_key, self.source = app_id, app_key, source
        self.endpoint, self.model = endpoint, model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, temperature=0.3, max_tokens=4096, timeout=None) -> str:
        import time
        ts = str(int(time.time()))
        sign = hmac.new(self.app_key.encode(),
                        f"x-timestamp: {ts}\nx-source: {self.source}".encode(),
                        hashlib.sha256).hexdigest()
        r = net.post(self.endpoint, timeout=net.request_timeout(timeout),
                     headers={"X-AppID": self.app_id, "X-Source": self.source,
                              "X-Timestamp": ts, "X-Authorization": sign,
                              "Content-Type": "application/json"},
                     json={"model": self.model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        self.last_usage = {"prompt_tokens": int(u.get("prompt_tokens", 0)),
                           "completion_tokens": int(u.get("completion_tokens", 0)),
                           "total_tokens": int(u.get("total_tokens", 0))}
        msg = data["choices"][0]["message"]["content"]
        return msg if isinstance(msg, str) else "".join(
            p.get("text", "") for p in msg if isinstance(p, dict))


def get_client(env: dict | None = None):
    """根据 .env 构建客户端；配置缺失/不完整时抛 RuntimeError（含修复指引）。"""
    env = env if env is not None else load_env()
    net.configure(env)
    kind = (env.get("AGENT_LLM_KIND") or os.environ.get("AGENT_LLM_KIND") or "").lower()
    proxy = env.get("AGENT_PROXY") or os.environ.get("AGENT_PROXY")
    if not kind:
        raise RuntimeError(
            "未配置 LLM。请 cp agent/.env.template agent/.env 并填入凭证，"
            "然后运行 python3 agent/check_api.py 自检。")
    if kind == "openai":
        missing = [k for k in ("AGENT_API_KEY", "AGENT_BASE_URL", "AGENT_MODEL") if not env.get(k)]
        if missing:
            raise RuntimeError(f".env 缺少字段: {missing}")
        return OpenAICompatClient(env["AGENT_API_KEY"], env["AGENT_BASE_URL"],
                                  env["AGENT_MODEL"], proxy)
    if kind == "ichat":
        missing = [k for k in ("AGENT_APP_ID", "AGENT_API_KEY", "AGENT_BASE_URL")
                   if not env.get(k)]
        if missing:
            raise RuntimeError(f".env 缺少字段: {missing}")
        return IChatClient(env["AGENT_APP_ID"], env["AGENT_API_KEY"],
                           env.get("AGENT_SOURCE", "default"), env["AGENT_BASE_URL"],
                           env.get("AGENT_MODEL", "gpt-4o"), proxy)
    raise RuntimeError(f"未知 AGENT_LLM_KIND: {kind}（可选 openai / ichat）")
