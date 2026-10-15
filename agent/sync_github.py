#!/usr/bin/env python3
"""把当前研究进度（代码+文档+汇总）提交并推送到 GitHub（Gelisi/LLM4FWA）。

用法：
    python3 agent/sync_github.py --message "iter_001: RL 重启控制实现+冒烟通过"
    python3 agent/sync_github.py --message "..." --no-push   # 只 commit 不 push

说明：
  - 遵循 .gitignore（密钥/会话/原始 run 数据不入库）；
  - 若配置了 AGENT_PROXY（agent/.env），git 网络操作自动走代理；
  - 无推送凭证时给出配置指引。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "https://github.com/Gelisi/LLM4FWA.git"


def git(args: list, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git"] + args, cwd=str(ROOT), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败:\n{r.stderr or r.stdout}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    # 代理（来自 agent/.env 的 AGENT_PROXY）：仅作用于 git 网络命令的环境
    env = None
    env_path = ROOT / "agent" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("AGENT_PROXY="):
                proxy = line.split("=", 1)[1].strip()
                env = {"http_proxy": proxy, "https_proxy": proxy,
                       "http_proxy_lower": proxy, "https_proxy_lower": proxy}

    if not (ROOT / ".git").exists():
        git(["init", "-b", "main"])
        print("已初始化 git 仓库")
    r = git(["remote"], check=False)
    if "origin" not in r.stdout.split():
        git(["remote", "add", "origin", REMOTE])
        print(f"已添加 remote origin -> {REMOTE}")

    # 远端已有内容时首次同步：合并保留远端文件
    r = git(["rev-parse", "--verify", "origin/main"], check=False)
    if r.returncode != 0:
        fetch = subprocess.run(["git", "fetch", "origin", "main"], cwd=str(ROOT),
                               capture_output=True, text=True,
                               env={**dict(__import__("os").environ), **(env or {})})
        if fetch.returncode != 0:
            print(f"fetch 失败（无凭证或网络）: {fetch.stderr.strip()[:200]}")
        else:
            m = git(["merge", "origin/main", "--allow-unrelated-histories",
                     "--no-edit"], check=False)
            print("已合并远端 main（保留远端既有文件）" if m.returncode == 0
                  else f"合并失败: {m.stderr.strip()[:200]}")

    git(["add", "-A"])
    r = git(["status", "--porcelain"], check=False)
    if not r.stdout.strip():
        print("没有待提交的变更")
        return 0
    cid = git(["commit", "-m", args.message], check=False)
    if cid.returncode != 0:
        print(f"commit 失败: {cid.stderr.strip()[:200]}")
        return 1
    sha = git(["rev-parse", "--short", "HEAD"]).stdout.strip()
    print(f"已提交 {sha}: {args.message}")

    if args.no_push:
        print("（--no-push，跳过推送）")
        return 0
    push = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=str(ROOT),
                          capture_output=True, text=True,
                          env={**dict(__import__("os").environ), **(env or {})})
    if push.returncode == 0:
        print("已推送到 github.com/Gelisi/LLM4FWA (main)")
        return 0
    print(f"push 失败: {push.stderr.strip()[:300]}")
    print("配置凭证（二选一）：\n"
          "  a) git remote set-url origin https://<PAT>@github.com/Gelisi/LLM4FWA.git\n"
          "  b) git config credential.helper store 后手动 push 一次输入 token")
    return 1


if __name__ == "__main__":
    sys.exit(main())
