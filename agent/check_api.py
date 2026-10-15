#!/usr/bin/env python3
"""LLM API 自检：校验 agent/.env 配置并发一条真实请求。

用法：
    python3 agent/check_api.py            # 读取 agent/.env 测试
    python3 agent/check_api.py --env x.env
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import get_client, load_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="指定 .env 路径（默认 agent/.env）")
    args = ap.parse_args()

    env = load_env() if not args.env else _load(args.env)
    if not env:
        print("agent/.env 不存在。请执行:\n"
              "  cp agent/.env.template agent/.env  然后填入凭证")
        return 1
    kind = env.get("AGENT_LLM_KIND", "?")
    print(f"适配器: {kind}  模型: {env.get('AGENT_MODEL')}")
    print(f"端点:   {env.get('AGENT_BASE_URL')}"
          f"{' (经代理 ' + env['AGENT_PROXY'] + ')' if env.get('AGENT_PROXY') else ''}")
    try:
        client = get_client(env)
    except RuntimeError as e:
        print(f"配置不完整: {e}")
        return 1
    try:
        out = client.chat([{"role": "user", "content": "只回复两个字：正常"}],
                          temperature=0, max_tokens=16)
        print(f"连通性: OK  模型回复: {out[:50]!r}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"连通性: 失败 -> {e}")
        print("排查建议：\n"
              "  1) key 是否有效（curl 直接测端点）\n"
              "  2) 需要外网时设置 AGENT_PROXY（见 .env.template 注释）\n"
              "  3) iChat 用户：确认 app 未被网关停用（403=disabled 即需换 app）")
        return 1


def _load(path: str) -> dict:
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


if __name__ == "__main__":
    sys.exit(main())
