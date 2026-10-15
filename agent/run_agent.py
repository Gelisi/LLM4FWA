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
    """从 LLM 输出提取第一个可解析的 JSON 对象（容忍前后缀/多段文本）。"""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("tool"):
            return {"thought": obj.get("thought", ""), "tool": obj["tool"],
                    "args": obj.get("args", {})}, None
        if isinstance(obj, dict):
            return None, f"JSON 缺少 tool 字段: {json.dumps(obj, ensure_ascii=False)[:150]}"
    return None, f"输出中没有可解析的 JSON 动作。原文前200字: {text[:200]}"


def log_token(usage: dict, meta: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **meta, **usage}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def goal_achieved(target_rank: int) -> bool:
    """仅完整、冻结、五个 benchmark 的 RL 评测可触发总目标停止。"""
    paths = list((ROOT / "results").glob("fwa_rl_*/status.json"))
    if not paths:
        return False
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    try:
        from fwarl.bench import BENCH_SPECS, CEC2019_FUNC_DIMS
        d = json.loads(latest.read_text())
        cfg = json.loads((latest.parent / "experiment_config.json").read_text())
        if cfg.get("train") or not cfg.get("policy_sha256") or cfg.get("df_control") != "rl_replace":
            return False
        ranks = d.get("bench_ranks", {})
        if set(ranks) != set(BENCH_SPECS) or any(r > target_rank for r in ranks.values()):
            return False
        for bench, spec in BENCH_SPECS.items():
            for fn in range(1, spec["num_functions"] + 1):
                dims = [CEC2019_FUNC_DIMS.get(fn, 10)] if bench == "cec2019" else spec["dims"]
                for dim in dims:
                    runs = [json.loads(p.read_text()) for p in
                            (latest.parent / bench / f"F{fn}_D{dim}").glob("run_*.json")]
                    if len(runs) < 30 or any(r.get("status") != "success" or
                                            r.get("max_evals") != spec["eval_factor"] * dim for r in runs):
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def log_session(path: Path, role: str, content: str):
    with open(path, "a") as f:
        f.write(f"\n### [{role}]\n{content}\n")


def run_session(client, box, system, task, sess: Path, max_turns: int,
                meta: dict, log_call) -> dict:
    """单次 ReAct 会话。返回 {turns, tokens, finished, summary}。"""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"任务：{task}\n请开始。先读状态，再动手。"}]
    stats = {"turns": 0, "tokens": 0, "finished": False, "summary": "", "hit_max": False,
             "exit_reason": "running"}
    session_start = time.time()

    turn = 0
    while max_turns is None or turn < max_turns:
        turn += 1
        t0 = time.time()
        reply = None
        for attempt in range(3):
            waiting = f"[LLM] turn={turn} 请求 {attempt + 1}/3，等待模型响应…"
            print(waiting, flush=True)
            log_session(sess, "request", waiting)
            try:
                reply = client.chat(messages, temperature=CFG.get("temperature", 0.3),
                                    max_tokens=CFG.get("max_tokens", 4096))
                break
            except Exception as e:  # noqa: BLE001
                from agent.net import error_summary
                feedback = f"[LLM 请求失败] {error_summary(e)}"
                print(feedback, flush=True)
                log_session(sess, "error", feedback)
                if attempt == 2:
                    break
                delay = 10 * (attempt + 1)
                print(f"[LLM] {delay}s 后重试", flush=True)
                time.sleep(delay)
        if reply is None:
            stats["exit_reason"] = "llm_error"
            break

        dt = time.time() - t0
        usage = dict(getattr(client, "last_usage", {}))
        stats["tokens"] += usage.get("total_tokens", 0)
        stats["turns"] = turn
        log_token(usage, {**meta, "turn": turn})
        log_session(sess, f"llm turn{turn} ({dt:.0f}s, {usage.get('total_tokens', 0)} tok)", reply)
        limit = max_turns if max_turns is not None else "∞"
        print(f"--- turn {turn}/{limit} ({dt:.0f}s, {usage.get('total_tokens', 0)} tok) ---")

        action, err = parse_action(reply)
        if err:
            feedback = f"[格式错误] {err}"
        elif action["tool"] == "finish":
            blocker = None if meta.get("mode") == "mock" else box.workflow.finish_error(after=session_start)
            if blocker:
                feedback = f"[finish 被拒绝] {blocker}"
            else:
                stats["finished"] = True
                stats["exit_reason"] = "workflow_completed"
                stats["elapsed_s"] = round(time.time() - session_start, 3)
                stats["summary"] = action["args"].get("summary", "")
                log_session(sess, "finish", stats["summary"])
                print(f"会话完成: {stats['summary'][:300]}")
                return stats
        else:
            verr = ToolBox.validate_args(action["tool"], action["args"])
            try:
                result = verr if verr else box.run(action["tool"], action["args"])
            except Exception as e:
                result = f"工具执行失败（需反思修订或重试，不能视为完成）: {type(e).__name__}: {e}"
            feedback = f"[{action['tool']} 结果]\n{result}"

        print(feedback[:400] + ("..." if len(feedback) > 400 else ""))
        log_session(sess, "tool", feedback)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": feedback})

    stats["hit_max"] = stats["exit_reason"] == "running"
    if stats["hit_max"]:
        stats["exit_reason"] = "max_turns"
    stats["elapsed_s"] = round(time.time() - session_start, 3)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="单任务模式：本次会话任务")
    ap.add_argument("--goal", default=None, help="目标驱动模式：总目标（多会话循环）")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="单段会话 LLM 交互轮数上限；默认不设限（一轮 iter 跑到底）")
    ap.add_argument("--max-sessions", type=int, default=40,
                    help="goal 模式最多迭代多少个 iter（默认 40）")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="goal 模式 token 总预算（达到即停，默认不限）")
    ap.add_argument("--target-rank", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.goal and not args.task and not (args.dry_run or args.mock):
        ap.error("需要 --task 或 --goal 之一")

    box = ToolBox(CFG)
    from agent import net
    net.configure(load_env())
    mode = net.network_mode()
    proxy = net.proxy_for_url(net.settings().get("AGENT_BASE_URL", ""), mode)
    print(f"[net] 模式={mode}；API 实际路由={'代理' if proxy else '直连'}", flush=True)
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
        run_start = time.time()
        manifest = {"run_id": stamp, "started_at": run_start, "arguments": vars(args), "sessions": []}
        manifest_path = sess_dir / f"{stamp}_run.json"
        exit_reason = "max_sessions"
        for si in range(1, args.max_sessions + 1):
            system = build_system_prompt(args.target_rank)
            if args.token_budget and used_tokens >= args.token_budget:
                print(f"[goal] token 预算 {args.token_budget} 已用 {used_tokens}，停止")
                exit_reason = "token_budget"
                break
            if goal_achieved(args.target_rank):
                print(f"[goal] 已达成 Top-{args.target_rank}（完整冻结评测），停止")
                exit_reason = "goal_achieved"
                break
            task = (f"总目标：{args.goal}。当前是第 {si} 段会话。"
                    "本段会话 = 一个 iter 的完整落地：先读 research_log.md 与各 iter 状态，"
                    "选定一个 iter（新立项或继续未完成的），在本段会话内完成它——"
                    "IDEA/代码/冒烟/正式实验/统计汇总/RESULTS 回填/research_log 记录/"
                    "sync_github 同步，全部做完才 finish。"
                    "不设交互轮数上限，不要中途草草收尾。")
            sess = sess_dir / f"{stamp}_s{si}.md"
            print(f"\n===== [goal] 会话 {si}/{args.max_sessions}  累计 token={used_tokens} =====")
            log_session(sess, "system", f"GOAL: {args.goal}\nTASK: {task}")
            meta = {"mode": "goal", "session": si, "task": args.goal}
            stats = run_session(client, box, system, task, sess, args.max_turns, meta, log_token)
            used_tokens += stats["tokens"]
            manifest["sessions"].append({"session": si, **stats})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                    "type": "session_summary", "mode": "goal",
                                    "session": si, **stats}, ensure_ascii=False) + "\n")
            print(f"[goal] 会话 {si} 摘要: turns={stats['turns']} "
                  f"tokens={stats['tokens']} finished={stats['finished']} reason={stats['exit_reason']}")
            if stats["exit_reason"] == "llm_error":
                exit_reason = "llm_error"
                break
            if goal_achieved(args.target_rank):
                print(f"[goal] 已达成 Top-{args.target_rank}！")
                exit_reason = "goal_achieved"
                break
        manifest.update(exit_reason=exit_reason, tokens=used_tokens,
                        ended_at=time.time(), elapsed_s=round(time.time() - run_start, 3))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"[goal] 结束 reason={exit_reason}。累计 token={used_tokens}，运行记录={manifest_path}")
        return 1 if exit_reason == "llm_error" else 0

    # ---------------- 单任务模式 ----------------
    sess = sess_dir / f"{stamp}.md"
    meta = {"mode": "task", "session": 0, "task": args.task}
    stats = run_session(client, box, system, args.task, sess, args.max_turns, meta, log_token)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "type": "session_summary", "mode": "task", **stats},
                           ensure_ascii=False) + "\n")
    print(f"tokens={stats['tokens']}, turns={stats['turns']}, reason={stats['exit_reason']}, 账本={LEDGER}")
    return 0 if stats["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
