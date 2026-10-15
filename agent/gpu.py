#!/usr/bin/env python3
"""GPU 管理：状态查询、空闲检测、选卡、空闲告警。

组成：
  - query()          解析 nvidia-smi，返回各卡 (util%, mem_used/total, 进程)
  - status_text()    给 agent 看的状态表
  - pick_gpu()       选"最空闲"的卡（util 最低，其次显存余量最大），无卡返回 None
  - idle_report()    依据 history 文件报告"空闲过久"的卡（跨会话记忆）
  - CLI watch 模式   周期采样并写告警到 agent/gpu_alerts.log

设计原则：当前负载大多为 CPU（run_eval 多进程），GPU 主要供 RL 训练/批量评测使用；
目标=不让任何一张卡长期空转（用户要求），由 agent 在 goal 循环中定期检查并安排任务。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "agent" / "gpu_history.json"
ALERTS = ROOT / "agent" / "gpu_alerts.log"


def query() -> list[dict]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return []
    gpus = []
    for line in r.stdout.strip().splitlines():
        idx, name, util, mu, mt = [c.strip() for c in line.split(",")]
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", "-i", idx],
            capture_output=True, text=True, timeout=30).stdout.strip()
        gpus.append({
            "index": int(idx), "name": name, "util": int(util),
            "mem_used": int(mu), "mem_total": int(mt),
            "procs": procs.splitlines() if procs else [],
        })
    return gpus


def status_text() -> str:
    gpus = query()
    if not gpus:
        return "未检测到 GPU（nvidia-smi 不可用或无卡）"
    lines = ["idx  name        util%  mem_used/total(MiB)  procs"]
    for g in gpus:
        lines.append(f"{g['index']:<4} {g['name'][:12]:<12} {g['util']:>4}   "
                     f"{g['mem_used']:>6}/{g['mem_total']:<6}     {len(g['procs'])} 个进程")
    # 空闲提示
    idle = [g["index"] for g in gpus if g["util"] < 5 and len(g["procs"]) == 0]
    if idle:
        lines.append(f"⚠ 空闲卡: {idle} —— 若超过 30 分钟仍空闲，请安排 GPU 任务"
                     f"（训练/评测批次），用法: CUDA_VISIBLE_DEVICES={idle[0]} <命令>")
    return "\n".join(lines)


def pick_gpu() -> int | None:
    """选当前最空闲的卡。返回卡号；无卡返回 None。"""
    gpus = query()
    if not gpus:
        return None
    return min(gpus, key=lambda g: (g["util"], g["mem_used"] / max(g["mem_total"], 1)))["index"]


def _sample_and_update() -> list[dict]:
    gpus = query()
    hist = {"samples": []}
    if HISTORY.exists():
        try:
            hist = json.loads(HISTORY.read_text())
        except Exception:
            pass
    now = time.time()
    hist["samples"] = [s for s in hist["samples"] if now - s["ts"] < 24 * 3600][-2000:]
    for g in gpus:
        hist["samples"].append({"ts": now, "index": g["index"], "util": g["util"],
                                "busy": g["util"] >= 5 or len(g["procs"]) > 0})
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(hist))
    return gpus


def idle_report(max_idle_min: float = 30) -> str:
    """基于采样历史，报告空闲超过阈值的卡。"""
    gpus = _sample_and_update()
    hist = json.loads(HISTORY.read_text()) if HISTORY.exists() else {"samples": []}
    now = time.time()
    out = []
    for g in gpus:
        samples = [s for s in hist["samples"] if s["index"] == g["index"]
                   and now - s["ts"] <= max_idle_min * 60]
        if len(samples) < 3:
            continue  # 历史不足，暂不判定
        idle_secs = 0
        for s in reversed(samples):
            if s["busy"]:
                break
            idle_secs += 60  # 相邻采样近似 1min 间隔
        idle_min = idle_secs / 60
        if idle_min >= max_idle_min and len(g["procs"]) == 0:
            out.append(f"GPU{g['index']} 已空闲约 {idle_min:.0f} 分钟")
    if out:
        msg = f"[{datetime.now():%F %T}] " + "; ".join(out)
        ALERTS.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS, "a") as f:
            f.write(msg + "\n")
        return "⚠ " + msg + "（已记录到 agent/gpu_alerts.log，请在 goal 循环中安排任务）"
    return "无长期空闲的卡"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="status", choices=["status", "pick", "watch"])
    ap.add_argument("--interval", type=int, default=300, help="watch 采样间隔秒")
    ap.add_argument("--max-idle-min", type=float, default=30)
    args = ap.parse_args()
    if args.mode == "status":
        print(status_text())
        print(idle_report(args.max_idle_min))
    elif args.mode == "pick":
        print(pick_gpu() if pick_gpu() is not None else "无GPU")
    else:
        while True:
            print(f"[{datetime.now():%F %T}] {idle_report(args.max_idle_min)}")
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
