"""线性 Q-learning agent：Q(s, a) = w_a · s，ε-greedy。

特性：
  - 每个动作一组权重 w_a ∈ R^{N_FEATURES}；
  - ε 从 eps_start 线性衰减到 eps_end；
  - 在线 TD(0) 更新：w += lr * (r + γ max_a' Q(s',a') - Q(s,a)) * s；
  - 可保存/加载（JSON），加载时按特征数/动作数校验维度；
  - 权重范数监控（供检测策略坍缩）。

跨 run 聚合由 experiments/train_policy.py 完成（避免多进程并发写同一文件）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinearQAgent:
    def __init__(self, n_features: int, n_actions: int,
                 lr: float = 0.02, gamma: float = 0.9,
                 eps_start: float = 0.3, eps_end: float = 0.05):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.w = np.zeros((n_actions, n_features), dtype=float)
        self._step = 0

    # ------------------------------------------------------------------
    @property
    def eps(self) -> float:
        # 随更新步数衰减，永不低于 eps_end
        return max(self.eps_end, self.eps_start / (1.0 + self._step / 2000.0))

    def q_values(self, s: np.ndarray) -> np.ndarray:
        return self.w @ np.asarray(s, dtype=float)

    def select(self, s: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.eps:
            return int(np.random.randint(self.n_actions))
        q = self.q_values(s)
        return int(np.argmax(q))

    def update(self, s, a: int, r: float, s_next, done: bool):
        q_next = 0.0 if done else float(np.max(self.q_values(s_next)))
        td = r + self.gamma * q_next - float(self.q_values(s)[a])
        self.w[a] += self.lr * td * np.asarray(s, dtype=float)
        self._step += 1

    # ------------------------------------------------------------------
    def weight_norms(self):
        return np.linalg.norm(self.w, axis=1).tolist()

    def action_counts(self):
        return self._action_counts.copy() if hasattr(self, "_action_counts") else [0] * self.n_actions

    def note_action(self, a: int):
        if not hasattr(self, "_action_counts"):
            self._action_counts = [0] * self.n_actions
        self._action_counts[a] += 1

    # ------------------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "n_features": self.n_features,
                "n_actions": self.n_actions,
                "w": self.w.tolist(),
                "steps": self._step,
                "action_counts": self.action_counts(),
            }, f, indent=1)

    @classmethod
    def load(cls, path, **overrides) -> "LinearQAgent":
        with open(path) as f:
            d = json.load(f)
        agent = cls(d["n_features"], d["n_actions"], **overrides)
        w = np.array(d["w"], dtype=float)
        if w.shape != agent.w.shape:
            raise ValueError(f"策略权重维度不匹配: {w.shape} != {agent.w.shape}")
        agent.w = w
        agent._step = d.get("steps", 0)
        agent._action_counts = d.get("action_counts", [0] * agent.n_actions)
        return agent
