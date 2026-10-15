#!/usr/bin/env python3
"""聚合训练轨迹，离线更新 RL 策略（避免多进程并发写冲突）。

流程：run_eval --train 产生 results/<algo>_<tag>/traj/*.json，
本脚本用固定随机种子打乱完整轨迹，回放 TD 更新并保存版本化策略。
不混入旧动作语义的轨迹，不以权重范数代替权重平均。

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
from fwarl.rl.actions import N_ACTIONS, POLICY_VERSION  # noqa: E402
from fwarl.fwa_rl import N_FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "policies" / "fwa_rl_policy.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="轨迹 JSON 的 glob")
    p.add_argument("--replay-epochs", type=int, default=1)
    p.add_argument("--out", default=str(POLICY_PATH))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.replay_epochs < 1:
        p.error("replay-epochs 必须为正数")
    rng = np.random.default_rng(args.seed)

    files = sorted(globmod.glob(args.glob))
    if not files:
        print(f"没有轨迹文件: {args.glob}")
        return 1

    trajectories = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        trajectories.append(d)

    n_feat, n_act = N_FEATURES, N_ACTIONS
    for t in trajectories:
        if t.get("policy_version") != POLICY_VERSION or not t.get("trajectory"):
            raise ValueError("轨迹缺失或版本不兼容，不能使用旧版 df 干预数据")
        for s, a, r, sn, done in t["trajectory"]:
            if len(s) != n_feat or len(sn) != n_feat or not 0 <= a < n_act:
                raise ValueError("训练轨迹特征或动作维度不匹配")

    agent = LinearQAgent(n_feat, n_act, policy_version=POLICY_VERSION)
    if Path(args.out).exists():
        agent = LinearQAgent.load(args.out, expected_version=POLICY_VERSION)
        if (agent.n_features, agent.n_actions) != (n_feat, n_act):
            raise ValueError("已有策略维度不匹配，请使用新的输出路径")

    epss = [t["trajectory"] for t in trajectories]
    print(f"聚合 {len(epss)} 条轨迹, 特征数 {n_feat}, 动作数 {n_act}")
    for epoch in range(args.replay_epochs):
        order = rng.permutation(len(epss))
        for idx in order:
            for s, a, r, s_next, done in epss[idx]:
                agent.update(np.asarray(s), int(a), float(r), np.asarray(s_next), bool(done))

    agent.save(args.out)
    print(f"策略已保存: {args.out}  (steps={agent._step}, 权重范数={agent.weight_norms()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
