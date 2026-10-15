"""统一 HTTP 层：auto 直连优先、网络失败后代理；direct/proxy 强制指定路由。

代理与路由读取 agent/.env 或当前客户端配置，环境变量优先；
连接超时和响应等待分开，等待期间输出不包含密钥的进度。

使用：
    from agent.net import get, post
    r = get("https://...", params={...}, timeout=20)
"""
from __future__ import annotations

import os
import re
import shlex
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) autoresearch-agent/1.0"}
_ENV_FILE = Path(__file__).resolve().parent / ".env"
_CONFIG = None


def read_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            continue
        try:
            tokens = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"{path.name} 第 {number} 行引号不匹配") from None
        env[key] = " ".join(tokens)
    return env


def _load_env_file() -> dict:
    return read_env(_ENV_FILE)


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


def _expand_reference(value, vals, seen=()):
    match = re.fullmatch(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)\})", value)
    if not match:
        return value
    name = match.group(1) or match.group(2)
    if name in seen:
        raise ValueError("代理变量存在循环引用，请检查 .env")
    return _expand_reference(vals.get(name, ""), vals, seen + (name,))


def resolve_proxy():
    vals = settings()

    def choose(*keys):
        candidates = [k for k in keys if k in os.environ] or list(keys)
        for key in candidates:
            value = _expand_reference(vals.get(key, ""), vals).strip()
            if value:
                return value
        return ""

    explicit = choose("AGENT_PROXY")
    proxies = {"http": explicit, "https": explicit} if explicit else {
        "http": choose("http_proxy", "HTTP_PROXY"),
        "https": choose("https_proxy", "HTTPS_PROXY"),
        "all": choose("all_proxy", "ALL_PROXY"),
    }
    proxies = {key: value for key, value in proxies.items() if value}
    bypass = choose("no_proxy", "NO_PROXY")
    if proxies and bypass:
        proxies["no_proxy"] = bypass
    return proxies or None


def proxy_for_url(url, mode=None):
    mode = network_mode() if mode is None else mode
    proxies = resolve_proxy() if mode != "direct" else None
    if mode == "auto" and proxies and requests.utils.should_bypass_proxies(url, no_proxy=proxies.get("no_proxy")):
        return None
    return proxies if proxies and requests.utils.select_proxy(url, proxies) else None


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
    tunnel = re.search(r"Tunnel connection failed:\s*(\d{3})\b", str(exc))
    if tunnel:
        status = f" CONNECT {tunnel.group(1)}（代理拒绝 HTTPS 隧道）"
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
    proxy = proxy_for_url(url, mode)
    if mode == "proxy" and not proxy:
        raise ValueError("proxy 模式没有当前协议的代理，请配置 https_proxy/http_proxy/all_proxy 或 AGENT_PROXY")
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
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            print(f"[net] {label} 失败：{error_summary(e)}，耗时 {time.monotonic() - started:.1f}s", flush=True)
            if not use_proxy and mode == "auto" and proxy:
                print("[net] 直连网络异常，尝试备用代理", flush=True)
    raise last_exc


def get(url: str, **kw):
    return request("GET", url, **kw)


def post(url: str, **kw):
    return request("POST", url, **kw)
