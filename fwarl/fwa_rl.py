"""FWA + RL：真实 TFWA（MPOPT 原版）+ 每代 RL 决策 df 自由度策略。

核心循环来自 /mnt/new_gemini/shipengcen/MPOPT/mpopt/algorithms/TFWA.py（用户原实现），
通过子类化加入 RL 钩子（源码中以 `# [RL]` 标注，其余逻辑保持与原版一致）：
  - 每代开始前：提取状态特征 -> agent 选动作 -> 施加到各烟花 dist.df；
  - 每代结束后：以全局最优的对数改进为奖励，TD(0) 更新。

训练模式（train=True）在线探索+学习；评测模式加载冻结策略、关闭探索。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MPOPT_DIR = Path("/mnt/new_gemini/shipengcen/MPOPT")
if str(MPOPT_DIR) not in sys.path:
    sys.path.insert(0, str(MPOPT_DIR))

from mpopt.tools.objective import Evaluator, ObjFunction  # noqa: E402
from mpopt.algorithms.TFWA import TFWA as _MPOPT_TFWA  # noqa: E402

from .rl.actions import ACTIONS, apply_action  # noqa: E402
from .rl.learner import LinearQAgent  # noqa: E402
from .tfwa import SafeEvaluator  # noqa: E402

N_FEATURES = 7
EPS = 1e-8


class TFWA_RL(_MPOPT_TFWA):
    """TFWA + RL：df 自由度策略由线性 Q agent 决策。"""

    def __init__(self, agent: LinearQAgent | None = None,
                 policy_path: str | None = None, train: bool = False,
                 rl_lr: float = 0.02, rl_gamma: float = 0.9):
        super().__init__()
        self.train = train
        if agent is not None:
            self.agent = agent
        elif policy_path:
            # 评测冻结探索；训练沿用 eps 衰减
            kw = dict(eps_start=0.0, eps_end=0.0) if not train else {}
            self.agent = LinearQAgent.load(policy_path, **kw)
        else:
            self.agent = LinearQAgent(N_FEATURES, len(ACTIONS),
                                      lr=rl_lr, gamma=rl_gamma)
        self.trajectory = []      # [(s, a, r, s_next, done)]
        self._ema_improve = 0.0
        self._last_best_eval = 0

    # ------------------------------------------------------------------
    def _rl_features(self, e, prev_best, cur_best):
        span = float(np.max(self.ub - self.lb))
        progress = e.num_eval / max(1, e.max_eval)

        ratio = (prev_best + 1e-12) / (cur_best + 1e-12)
        r = float(np.clip(np.log(max(ratio, 1e-12)), 0.0, 10.0) / 10.0)
        self._ema_improve = 0.7 * self._ema_improve + 0.3 * r

        stagnation = min(1.0, (e.num_eval - self._last_best_eval) / max(1, e.max_eval))

        scales = np.array([p.dist.scale for p in self.pops])
        scale_ratio = float(np.clip(np.log1p(scales.mean() / max(span, 1e-12) * 10.0) / np.log(11.0), 0.0, 1.0))

        dfs = np.array([max(2, p.dist.df) for p in self.pops], dtype=float)
        df_feat = float(np.clip(np.log2(dfs / self.init_df).mean() / 10.0, -1.0, 1.0))

        spread = float(np.clip((self.fits.max() - self.fits.min()) /
                               (abs(self.fits.min()) + 1e-12 + span), 0.0, 1.0))

        return np.array([1.0, progress, self._ema_improve, scale_ratio,
                         stagnation, df_feat, spread], dtype=float)

    def _rl_reward(self, prev_best, cur_best):
        if cur_best >= prev_best:
            return -0.1 if cur_best > prev_best else 0.0
        ratio = (prev_best + 1e-12) / (cur_best + 1e-12)
        return float(np.clip(np.log(max(ratio, 1.0)) / np.log(10.0), -0.5, 1.0))

    # ------------------------------------------------------------------
    def optimize(self, e):
        # ---- 原版 init ----
        self.init(e)
        self._last_best_eval = 0
        best_y = min(self.fits)

        while not e.terminate():
            # [RL] 决策：状态 -> 动作 -> 施加 df
            s = self._rl_features(e, best_y, best_y)
            a = self.agent.select(s, explore=self.train)
            self.agent.note_action(a)
            apply_action(self.pops, self.factors, self.init_df, ACTIONS[a])

            prev_best = min(self.fits)

            # ---- 原版主循环（逐烟花演化 + 重启判定），仅加入 [RL] 标注 ----
            n_fit = np.array([0.0 for i in range(self.fw_size)])
            for i in range(self.fw_size):
                self.pops[i].evolve(e)
                n_fit[i] = e(self.pops[i].dist.shift.reshape(1, self.dim))[0]

                self.df_his[i].append(self.pops[i].dist.df)
                self.fit_his[i].append(self.fits[i])
            improves = self.fits - n_fit
            rest_iter = (e.max_eval - e.num_eval) / (self.sp_size + self.fw_size)
            restart_ = (improves > EPS) * (improves * rest_iter < (n_fit - e.cur_y))
            for i in range(self.fw_size):
                if self.use_restart:
                    if restart_[i] == True:  # noqa: E712
                        self.restart(i, e)
                        if getattr(self, "repair_control_state", False):
                            n_fit[i] = self.fits[i]
                        if self.verbose:
                            print(f"count:{count}, restart!")
                    if self.pops[i].no_improve_count >= 100:
                        self.restart(i, e, shift=self.pops[i].dist.shift)
                        if getattr(self, "repair_control_state", False):
                            n_fit[i] = self.fits[i]
            self.fits = n_fit

            # [RL] 反馈：全局最优改进 -> 奖励 -> TD 更新
            cur_best = min(self.fits.min(), float(e.cur_y))
            if cur_best < best_y:
                self._last_best_eval = e.num_eval
                best_y = cur_best
            r = self._rl_reward(prev_best, cur_best)
            s_next = self._rl_features(e, prev_best, cur_best)
            done = e.terminate()
            if self.train:
                self.agent.update(s, a, r, s_next, done)
            self.trajectory.append((s, a, r, s_next, done))

        return e.best_y


def tfwa_rl_solve(problem, seed=0, record_history=True, train=False,
                  policy_path=None, **rl_kwargs) -> dict:
    obj = ObjFunction(problem.objective, dim=problem.dim,
                      lb=problem.lb, ub=problem.ub)
    traj_mod = max(100, problem.max_evals // 100) if record_history else 0
    e = SafeEvaluator(obj, max_eval=problem.max_evals, traj_mod=traj_mod)

    algo = TFWA_RL(policy_path=policy_path, train=train, **rl_kwargs)
    p = algo.default_params()
    p["seed"] = int(seed)
    algo.set_params(p)

    best = algo.optimize(e)

    history = []
    if record_history and e.traj:
        history = [{"evals": int((i + 1) * traj_mod), "best_f": float(y)}
                   for i, y in enumerate(e.traj)]

    return {
        "best_f": float(best),
        "evals": int(e.num_eval),
        "history": history,
        "weight_norms": algo.agent.weight_norms(),
        "action_counts": algo.agent.action_counts(),
        "trajectory": [(list(map(float, s)), int(a), float(r),
                        list(map(float, sn)), bool(d)) for s, a, r, sn, d in algo.trajectory],
    }
