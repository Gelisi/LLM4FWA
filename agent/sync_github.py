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


def load_env() -> dict:
    sys.path.insert(0, str(ROOT))
    from agent.net import read_env
    return read_env(ROOT / "agent" / ".env")


def push_url() -> str:
    """带 token 的推送 URL（token 支持 github-tokens / GITHUB_TOKEN / github_token）。"""
    env = load_env()
    tok = env.get("github-tokens") or env.get("GITHUB_TOKEN") or env.get("github_token")
    return f"https://{tok}@github.com/Gelisi/LLM4FWA.git" if tok else REMOTE


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

    import os
    values = load_env()
    from agent import net
    net.configure(values)
    proxies = net.proxy_for_url(REMOTE)
    if net.network_mode() == "proxy" and not proxies:
        print("proxy 模式没有 GitHub HTTPS 代理，请配置 https_proxy/all_proxy 或 AGENT_PROXY")
        return 1
    env = {f"{key}_proxy": value for key, value in (proxies or {}).items() if key != "no_proxy"}
    if proxies and proxies.get("no_proxy") and net.network_mode() != "proxy":
        env["no_proxy"] = proxies["no_proxy"]
    network_env = {key: value for key, value in os.environ.items()
                   if key.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
                                          "http_proxy_lower", "https_proxy_lower")}
    network_env.update(env)

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
                               env=network_env)
        if fetch.returncode != 0:
            print(f"fetch 失败（无凭证或网络）: {fetch.stderr.strip()[:200]}")
        else:
            m = git(["merge", "origin/main", "--allow-unrelated-histories",
                     "--no-edit"], check=False)
            print("已合并远端 main（保留远端既有文件）" if m.returncode == 0
                  else f"合并失败: {m.stderr.strip()[:200]}")

    ahead = False
    fetch = subprocess.run(["git", "fetch", "origin", "main"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=120, env=network_env)
    if fetch.returncode == 0:
        ab = git(["rev-list", "--left-right", "--count", "origin/main...HEAD"],
                 check=False).stdout.split()
        if len(ab) == 2 and ab[1] != "0":
            ahead = True
    else:
        # fetch 失败无法对比，保守视为可能有未推送提交（push 对已同步仓库是无害 no-op）
        ahead = True

    git(["add", "-A"])
    r = git(["status", "--porcelain"], check=False)
    if not r.stdout.strip():
        if ahead:
            print("没有待提交的变更；本地领先 origin/main，继续推送")
        else:
            print("没有待提交的变更，且与 origin/main 同步")
            return 0
    else:
        cid = git(["commit", "-m", args.message], check=False)
        if cid.returncode != 0:
            print(f"commit 失败: {cid.stderr.strip()[:200]}")
            return 1
        sha = git(["rev-parse", "--short", "HEAD"]).stdout.strip()
        print(f"已提交 {sha}: {args.message}")

    if args.no_push:
        print("（--no-push，跳过推送）")
        return 0
    url = push_url()
    last = None
    for label, penv in (("代理" if proxies else "直连", network_env),):
        push = subprocess.run(["git", "push", url, "main:main"], cwd=str(ROOT),
                              capture_output=True, text=True, env=penv, timeout=600)
        if push.returncode == 0:
            print(f"已推送到 github.com/Gelisi/LLM4FWA (main, {label})")
            return 0
        last = push
        # 认证/权限类错误换网络路径没用，直接停
        if "403" in push.stderr or "denied" in push.stderr.lower() or \
                "could not read Username" in push.stderr:
            break
    err = (last.stderr if last else "").strip()
    print(f"push 失败: {err[:300]}")
    if "403" in err or "denied" in err.lower():
        print("token 已认证但无写权限：请在 GitHub 给该 token 增加对本仓库的写权限\n"
              "  (classic PAT 勾 repo scope / fine-grained 对本仓库设 Contents: Read and write)")
    elif "could not read Username" in err or "Authentication" in err:
        print("缺少凭证：请在 agent/.env 写入 github-tokens=<PAT>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
