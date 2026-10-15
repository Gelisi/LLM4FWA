"""统一 HTTP 层：支持 auto（直连后代理）、direct、proxy 三种路由。

代理与路由读取 agent/.env 或当前客户端配置，环境变量优先；
连接超时和响应等待分开，等待期间输出不包含密钥的进度。

使用：
    from agent.net import get, post
    r = get("https://...", params={...}, timeout=20)
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) autoresearch-agent/1.0"}
_ENV_FILE = Path(__file__).resolve().parent / ".env"
_CONFIG = None


def _load_env_file() -> dict:
    env = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def configure(values: dict):
    global _CONFIG
    _CONFIG = dict(values)


def settings() -> dict:
    vals = _load_env_file() if _CONFIG is None else _CONFIG
    return {**vals, **os.environ}


def network_mode() -> str:
    mode = settings().get("AGENT_NETWORK_MODE", "auto").strip().lower()
    if mode not in ("auto", "direct", "proxy"):
        raise ValueError("AGENT_NETWORK_MODE 必须为 auto/direct/proxy")
    return mode


def resolve_proxy():
    vals = settings()
    cand = (vals.get("AGENT_PROXY") or vals.get("https_proxy")
            or vals.get("HTTPS_PROXY") or vals.get("http_proxy") or vals.get("HTTP_PROXY"))
    if cand:
        cand = os.path.expandvars(cand)
        if "$" not in cand:
            return {"http": cand, "https": cand}
    return None


def request_timeout(read_timeout=None):
    vals = settings()
    connect = float(vals.get("AGENT_CONNECT_TIMEOUT_S", 10))
    read = float(vals.get("AGENT_LLM_TIMEOUT_S", 300) if read_timeout is None else read_timeout)
    if connect <= 0 or read <= 0:
        raise ValueError("连接与响应超时必须为正数")
    return connect, read


def error_summary(exc) -> str:
    response = getattr(exc, "response", None)
    status = f" HTTP {response.status_code}" if response is not None else ""
    return f"{type(exc).__name__}{status}"


@contextmanager
def _progress(label):
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            print(f"[net] {label} 仍在等待响应，已耗时 {time.monotonic() - started:.0f}s", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def request(method: str, url: str, **kw):
    """显式路由，不输出代理凭证、请求头或响应正文。"""
    timeout = kw.pop("timeout", 60)
    kw["timeout"] = timeout if isinstance(timeout, tuple) else request_timeout(timeout)
    headers = dict(UA)
    headers.update(kw.pop("headers", {}) or {})
    kw["headers"] = headers
    mode = network_mode()
    proxy = resolve_proxy() if mode != "direct" else None
    if mode == "proxy" and not proxy:
        raise ValueError("proxy 模式未配置有效 AGENT_PROXY，请填写远端可访问的代理地址")
    attempts = [True] if mode == "proxy" else ([False, True] if mode == "auto" and proxy else [False])
    last_exc = None
    for use_proxy in attempts:
        label = f"{'代理' if use_proxy else '直连'} {method} {urlsplit(url).hostname}"
        print(f"[net] {label}，连接/响应超时={kw['timeout']}s", flush=True)
        started = time.monotonic()
        try:
            with _progress(label), requests.Session() as s:
                s.trust_env = False
                r = s.request(method, url, proxies=proxy if use_proxy else {}, **kw)
            print(f"[net] {label} 返回 HTTP {r.status_code}，耗时 {time.monotonic() - started:.1f}s", flush=True)
            return r
        except requests.RequestException as e:
            last_exc = e
            print(f"[net] {label} 失败：{error_summary(e)}，耗时 {time.monotonic() - started:.1f}s", flush=True)
    raise last_exc


def get(url: str, **kw):
    return request("GET", url, **kw)


def post(url: str, **kw):
    return request("POST", url, **kw)
