#!/usr/bin/env python3
"""LLM API 自检：校验 agent/.env 配置并发一条真实请求。

用法：
    python3 agent/check_api.py            # 读取 agent/.env 测试
    python3 agent/check_api.py --env x.env
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import get_client, load_env  # noqa: E402
from agent import net  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="指定 .env 路径（默认 agent/.env）")
    ap.add_argument("--network-mode", choices=("auto", "direct", "proxy"))
    ap.add_argument("--timeout", type=float, default=25, help="每条网络路由的响应超时秒数")
    args = ap.parse_args()
    if args.timeout <= 0:
        ap.error("timeout 必须为正数")
    if args.network_mode:
        os.environ["AGENT_NETWORK_MODE"] = args.network_mode

    env = load_env() if not args.env else _load(args.env)
    if not env:
        print("agent/.env 不存在。请执行:\n"
              "  cp agent/.env.template agent/.env  然后填入凭证")
        return 1
    kind = env.get("AGENT_LLM_KIND", "?")
    print(f"适配器: {kind}  模型: {env.get('AGENT_MODEL')}", flush=True)
    print(f"端点主机: {urlsplit(env.get('AGENT_BASE_URL', '')).hostname}", flush=True)
    try:
        client = get_client(env)
        mode = net.network_mode()
        proxy = net.proxy_for_url(env.get("AGENT_BASE_URL", ""), mode)
        print(f"网络模式: {mode}；实际路由={'代理' if proxy else '直连'}", flush=True)
        if mode == "proxy" and not proxy:
            raise ValueError("请配置当前协议的 https_proxy/http_proxy/all_proxy 或 AGENT_PROXY")
    except (RuntimeError, ValueError) as e:
        print(f"配置不完整: {e}", flush=True)
        return 1
    try:
        out = client.chat([{"role": "user", "content": "只回复两个字：正常"}],
                          temperature=0, max_tokens=16, timeout=args.timeout)
        print(f"连通性: OK  模型回复: {out[:50]!r}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"连通性: 失败 -> {net.error_summary(e)}", flush=True)
        print("排查建议：\n"
              "  1) ConnectTimeout/ProxyError：检查远端到端点或代理的连通性\n"
              "  2) ReadTimeout：连接建立后仍无响应，检查网关和模型服务\n"
              "  3) CONNECT 403/407：代理访问许可/认证问题；HTTP 401/403：API 凭证或权限问题\n"
              "  4) 用 --network-mode direct/proxy 分别测试，代理必须是远端可访问的真实地址")
        return 1


def _load(path: str) -> dict:
    return net.read_env(Path(path))


if __name__ == "__main__":
    sys.exit(main())
