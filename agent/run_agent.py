#!/usr/bin/env python3
"""autoresearch 研究代理。

两种模式：
  1) 单任务模式（默认）：一次会话完成一个聚焦任务。
     python3 agent/run_agent.py --task "..."
  2) 目标驱动模式：--goal 给定总目标，agent 自主循环"读状态→立项→写码→编译→
     实验→汇总→记录"，一轮会话接一轮会话，直到达成目标/达到上限。
     python3 agent/run_agent.py --goal "各 benchmark 综合实力 Top-2" --max-sessions 15

Token 记账：每次 LLM 调用写一行 JSONL 到 agent/token_usage.jsonl
  {ts, mode, session, task, turn, prompt_tokens, completion_tokens, total_tokens}
每会话结束追加一行 {"type":"session_summary", ...}；每 iter 归集可用
  grep <iter_slug> token_usage.jsonl 汇总。

会话留痕：agent/sessions/<时间戳>[_s<N>].md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.llm import get_client, load_env  # noqa: E402
from agent.prompts import build_system_prompt  # noqa: E402
from agent.tools import ToolBox  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "agent" / "agent_config.json").read_text())
LEDGER = ROOT / CFG.get("token_ledger", "agent/token_usage.jsonl")


def parse_action(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, f"输出中没有 JSON 动作。原文前200字: {text[:200]}"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}。请输出单个完整 JSON 对象。"
    if not obj.get("tool"):
        return None, "缺少 tool 字段"
    return {"thought": obj.get("thought", ""), "tool": obj["tool"],
            "args": obj.get("args", {})}, None


def log_token(usage: dict, meta: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **meta, **usage}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def goal_achieved(target_rank: int) -> bool:
    """检查最新 status.json：所有 benchmark achieved 即达成。"""
    latest = None
    for p in sorted((ROOT / "results").glob("*/status.json")):
        latest = p
    if latest is None:
        return False
    try:
        d = json.loads(latest.read_text())
        return bool(d.get("achieved")) and all(d["achieved"].values())
    except Exception:
        return False


def log_session(path: Path, role: str, content: str):
    with open(path, "a") as f:
        f.write(f"\n### [{role}]\n{content}\n")


def run_session(client, box, system, task, sess: Path, max_turns: int,
                meta: dict, log_call) -> dict:
    """单次 ReAct 会话。返回 {turns, tokens, finished, summary}。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"任务：{task}\n请开始。先读状态，再动手。"}]
    stats = {"turns": 0, "tokens": 0, "finished": False, "summary": "", "hit_max": False}

    for turn in range(1, max_turns + 1):
        t0 = time.time()
        reply = None
        for attempt in range(3):
            try:
                reply = client.chat(messages, temperature=CFG.get("temperature", 0.3),
                                    max_tokens=CFG.get("max_tokens", 4096))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    feedback = f"[LLM 调用失败，跳过本轮] {e}"
                    log_session(sess, "error", feedback)
                    break
                time.sleep(10 * (attempt + 1))
        if reply is None:
            break

        dt = time.time() - t0
        usage = dict(getattr(client, "last_usage", {}))
        stats["tokens"] += usage.get("total_tokens", 0)
        stats["turns"] = turn
        log_token(usage, {**meta, "turn": turn})
        log_session(sess, f"llm turn{turn} ({dt:.0f}s, {usage.get('total_tokens', 0)} tok)", reply)
        print(f"--- turn {turn}/{max_turns} ({dt:.0f}s, {usage.get('total_tokens', 0)} tok) ---")

        action, err = parse_action(reply)
        if err:
            feedback = f"[格式错误] {err}"
        elif action["tool"] == "finish":
            stats["finished"] = True
            stats["summary"] = action["args"].get("summary", "")
            log_session(sess, "finish", stats["summary"])
            print(f"会话完成: {stats['summary'][:300]}")
            return stats
        else:
            verr = ToolBox.validate_args(action["tool"], action["args"])
            result = verr if verr else box.run(action["tool"], action["args"])
            feedback = f"[{action['tool']} 结果]\n{result}"

        print(feedback[:400] + ("..." if len(feedback) > 400 else ""))
        log_session(sess, "tool", feedback)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": feedback})

    stats["hit_max"] = True
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="单任务模式：本次会话任务")
    ap.add_argument("--goal", default=None, help="目标驱动模式：总目标（多会话循环）")
    ap.add_argument("--max-turns", type=int, default=CFG["max_turns"])
    ap.add_argument("--max-sessions", type=int, default=10, help="goal 模式最多会话数")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="goal 模式 token 总预算（达到即停）")
    ap.add_argument("--target-rank", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.goal and not args.task and not (args.dry_run or args.mock):
        ap.error("需要 --task 或 --goal 之一")

    box = ToolBox(CFG)
    # 代理策略（net.py）：直连优先，失败自动回退到 .env 中解析出的代理。
    # 不再全局注入 os.environ，避免把代理强加给所有子进程。
    _env = load_env()
    if not (_env.get("AGENT_PROXY") or _env.get("https_proxy") or _env.get("http_proxy")):
        print("[net] 未配置代理，全部直连")
    system = build_system_prompt(args.target_rank)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = ROOT / CFG["session_log_dir"]
    sess_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[dry-run] 上下文与工具自检：")
        print(f"  系统提示 {len(system)} 字符")
        print("  bash 'ls results' ->", box.run("bash", {"command": "ls results"})[:80])
        print("  编译白名单 'make --version' ->",
              box.run("bash", {"command": "make --version"})[:40])
        print("  黑名单 'curl' ->", box.run("bash", {"command": "curl http://x"})[:40])
        print("dry-run 通过。")
        return 0

    if args.mock:
        scripted = [
            ('{"thought":"读研究日志","tool":"bash","args":{"command":"tail -4 research_log.md"}}'),
            ('{"thought":"读模板","tool":"read_file","args":{"path":"iterations/_TEMPLATE/IDEA.md","limit":5}}'),
            ('{"thought":"完成","tool":"finish","args":{"summary":"mock 全链路自测通过"}}'),
        ]

        class MockClient:
            def __init__(self):
                self.i = -1
                self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50,
                                   "total_tokens": 150}

            def chat(self, messages, **kw):
                self.i += 1
                return scripted[min(self.i, len(scripted) - 1)]

        client = MockClient()
        print("[mock] 使用脚本化 LLM 全链路自测")
        stats = run_session(client, box, system, args.task or "mock",
                            sess_dir / f"{stamp}_mock.md", 5,
                            {"mode": "mock", "session": 0, "task": "mock"}, log_token)
        print(json.dumps(stats, ensure_ascii=False))
        return 0

    try:
        client = get_client(load_env())
    except RuntimeError as e:
        print(f"[LLM 未就绪] {e}")
        return 1

    # ---------------- 目标驱动模式 ----------------
    if args.goal:
        used_tokens = 0
        for si in range(1, args.max_sessions + 1):
            if args.token_budget and used_tokens >= args.token_budget:
                print(f"[goal] token 预算 {args.token_budget} 已用 {used_tokens}，停止")
                break
            if goal_achieved(args.target_rank):
                print(f"[goal] 已达成 Top-{args.target_rank}（status.json 全 achieved），停止")
                break
            task = (f"总目标：{args.goal}。当前是第 {si} 段会话。"
                    "先读 research_log.md 与各 iter 状态，选出当下价值最高的一步"
                    "（立项新 iter / 实现代码 / 编译 / 跑实验 / 统计汇总 / 记录结论），"
                    "把它实质推进：能跑完实验就跑完并回填文档；"
                    "一次会话聚焦一步，收尾时 finish 并写清进展与下一步。")
            sess = sess_dir / f"{stamp}_s{si}.md"
            print(f"\n===== [goal] 会话 {si}/{args.max_sessions}  累计 token={used_tokens} =====")
            log_session(sess, "system", f"GOAL: {args.goal}\nTASK: {task}")
            meta = {"mode": "goal", "session": si, "task": args.goal}
            stats = run_session(client, box, system, task, sess, args.max_turns, meta, log_token)
            used_tokens += stats["tokens"]
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                    "type": "session_summary", "mode": "goal",
                                    "session": si, **stats}, ensure_ascii=False) + "\n")
            print(f"[goal] 会话 {si} 摘要: turns={stats['turns']} "
                  f"tokens={stats['tokens']} finished={stats['finished']}")
            if goal_achieved(args.target_rank):
                print(f"[goal] 已达成 Top-{args.target_rank}！")
                break
        print(f"[goal] 结束。累计 token={used_tokens}，账本={LEDGER}")
        return 0

    # ---------------- 单任务模式 ----------------
    sess = sess_dir / f"{stamp}.md"
    meta = {"mode": "task", "session": 0, "task": args.task}
    stats = run_session(client, box, system, args.task, sess, args.max_turns, meta, log_token)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "type": "session_summary", "mode": "task", **stats},
                           ensure_ascii=False) + "\n")
    print(f"tokens={stats['tokens']}, turns={stats['turns']}, 账本={LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
