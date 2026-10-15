#!/usr/bin/env python3
"""聚合训练轨迹，离线更新 RL 策略（避免多进程并发写冲突）。

流程：run_eval --train 产生 results/<algo>_<tag>/traj/*.json（每条含 (s,a,r,s',done) 序列
与该 run 末端的 agent 权重），本脚本把所有轨迹按文件名顺序回放 TD 更新，
与各 run 末端权重的简单平均做融合，写入 policies/fwa_rl_policy.json。

用法：
    python3 experiments/train_policy.py --glob "results/fwa_rl_train_c22d10/traj/*.json"
    python3 experiments/train_policy.py --glob "..." --replay-epochs 2
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fwarl.rl.learner import LinearQAgent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "policies" / "fwa_rl_policy.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="轨迹 JSON 的 glob")
    p.add_argument("--replay-epochs", type=int, default=1)
    p.add_argument("--out", default=str(POLICY_PATH))
    args = p.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        print(f"没有轨迹文件: {args.glob}")
        return 1

    trajectories = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        trajectories.append(d)

    n_feat = len(trajectories[0]["trajectory"][0][0])
    n_act = int(max(a for t in trajectories for _, a, _, _, _ in t["trajectory"])) + 1

    # 已有策略作为起点（增量学习）；否则从零开始
    agent = LinearQAgent(n_feat, n_act)
    if Path(args.out).exists():
        try:
            agent = LinearQAgent.load(args.out)
        except ValueError:
            print("已有策略维度不匹配，从零开始训练")

    epss = [t["trajectory"] for t in trajectories]
    print(f"聚合 {len(epss)} 条轨迹, 特征数 {n_feat}, 动作数 {n_act}")
    for epoch in range(args.replay_epochs):
        order = np.random.permutation(len(epss))
        for idx in order:
            for s, a, r, s_next, done in epss[idx]:
                agent.update(np.asarray(s), int(a), float(r), np.asarray(s_next), bool(done))

    # 与各 run 末端权重的平均做融合（在线学习与离线回放的折中）
    finals = [np.asarray(t["weight_norms"], dtype=float) for t in trajectories
              if t.get("weight_norms")]
    if finals:
        mean_norm = np.mean(finals, axis=0)
        cur_norm = np.linalg.norm(agent.w, axis=1)
        scale = np.divide(mean_norm, np.maximum(cur_norm, 1e-12))
        agent.w *= scale[:, None]
        print(f"融合了 {len(finals)} 个 run 末端权重的范数")

    agent.save(args.out)
    print(f"策略已保存: {args.out}  (steps={agent._step}, 权重范数={agent.weight_norms()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
