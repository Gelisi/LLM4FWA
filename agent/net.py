"""统一 HTTP 层：直连优先，失败自动回退到 .env 里的代理。

策略（用户要求）：
  1. 第一次尝试直连（禁用环境代理，避免杂散变量干扰）；
  2. 失败（网络类错误）→ 用 .env 中解析出的代理重试一次；
  3. 代理值支持 $var 引用系统环境变量；解析为空则视为无代理。

使用：
    from agent.net import get, post
    r = get("https://...", params={...}, timeout=20)
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) autoresearch-agent/1.0"}
_ENV_FILE = Path(__file__).resolve().parent / ".env"

_PROXY = None
_RESOLVED = False


def _load_env_file() -> dict:
    env = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_proxy():
    """返回 {'http':..., 'https':...} 或 None（直连）。结果缓存。"""
    global _PROXY, _RESOLVED
    if _RESOLVED:
        return _PROXY
    vals = _load_env_file()
    cand = (vals.get("AGENT_PROXY") or vals.get("https_proxy")
            or vals.get("HTTPS_PROXY") or vals.get("http_proxy"))
    if cand:
        if cand.startswith("$"):
            cand = os.environ.get(cand[1:], "")
        if cand and not cand.startswith("$"):
            _PROXY = {"http": cand, "https": cand}
    _RESOLVED = True
    return _PROXY


def request(method: str, url: str, **kw):
    """直连优先、代理回退的 HTTP 请求。kw 同 requests。"""
    kw.setdefault("timeout", 60)
    headers = dict(UA)
    headers.update(kw.pop("headers", {}) or {})
    kw["headers"] = headers

    proxy = resolve_proxy()
    attempts = [False, True] if proxy else [False]  # False=直连, True=走代理
    last_exc = None
    for use_proxy in attempts:
        try:
            if use_proxy:
                return requests.request(method, url, proxies=proxy, **kw)
            with requests.Session() as s:
                s.trust_env = False  # 直连：无视环境代理变量
                return s.request(method, url, **kw)
        except requests.RequestException as e:
            last_exc = e
    raise last_exc


def get(url: str, **kw):
    return request("GET", url, **kw)


def post(url: str, **kw):
    return request("POST", url, **kw)
