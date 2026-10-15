"""TFWA 适配层：直接使用 MPOPT 中的原版 TFWA 实现（沈鹏岑自研）。

原实现位置：/mnt/new_gemini/shipengcen/MPOPT/mpopt/algorithms/TFWA.py
算法要点（以原代码为准）：
  - fw_size=2 个烟花，每个烟花是一个 TMA-ES 式 EDA 种群（TMAESPop），
    分布为多元 t 分布（MultiVariateTstudentDistribution，含自由度 df）；
  - 两个烟花的 df 增长因子 factors=[1.05, 10.0] 不同（一个慢收敛、一个快收敛）；
  - df 越大 t 分布越接近高斯（越收敛），df 越小尾部越重（越探索）；
  - 稳健加权（robust weight，按 t 分布的 Mahalanobis 距离加权）；
  - 重启机制：当 (改进量×剩余迭代 < 与当前最优差距) 或连续 100 代无改进时重启；
  - 越界采样用 uniform random remap。

本模块只做薄适配（Problem -> ObjFunction/Evaluator），保证与原实现零偏差。
"""
from __future__ import annotations

from .paths import MPOPT_DIR, ensure_syspath

ensure_syspath()

from mpopt.tools.objective import Evaluator, ObjFunction  # noqa: E402
from mpopt.algorithms.TFWA import TFWA as _MPOPT_TFWA  # noqa: E402

TFWA = _MPOPT_TFWA  # 供外部直接使用原类


class SafeEvaluator(Evaluator):
    """修复 mpopt Evaluator 的轨迹记录 bug：

    原代码单点评估分支写的是 `(self.num_eval + 1) % self.num_eval == 0`，
    首条评估（num_eval==0）时除零崩溃；按语义应为对 traj_mod 取模。
    其余逻辑与原版完全一致。
    """

    def eval(self, X):
        import numpy as _np

        num = X.shape[0]
        y = self.obj(X)

        if num == 1:
            if self.cur_y is None or y < self.cur_y:
                self.cur_x = X[0, :]
                self.cur_y = y[0]
                self.unimprove_eval = 0
                self.unimprove_batch = 0
            else:
                self.unimprove_eval += 1
                self.unimprove_batch += 1

            if self.traj_mod > 0 and (self.num_eval + 1) % self.traj_mod == 0:
                self.traj.append(self.cur_y)
        else:
            min_idx = _np.argmin(y)
            min_x = X[min_idx, :]
            min_y = y[min_idx]

            if self.cur_y is None or min_y < self.cur_y:
                self.cur_x = min_x
                self.cur_y = min_y
                self.unimprove_eval = 0
                self.unimprove_batch = 0
            else:
                self.unimprove_eval += num
                self.unimprove_batch += 1

            if self.traj_mod > 0:
                mod = self.traj_mod
                r = mod - (self.num_eval % mod)
                cnt = int((num - r) // mod) + 1
                self.traj += [self.cur_y] * cnt

        self.num_eval += num
        self.num_batch += 1
        return y


def tfwa_solve(problem, seed=0, record_history=True, **params) -> dict:
    """在 Problem 上运行原版 TFWA，返回与评测 JSON 兼容的字段。

    params 可覆盖 default_params 中任意项（factors/init_df/use_restart 等）。
    """
    obj = ObjFunction(problem.objective, dim=problem.dim,
                      lb=problem.lb, ub=problem.ub)
    traj_mod = max(100, problem.max_evals // 100) if record_history else 0
    e = SafeEvaluator(obj, max_eval=problem.max_evals, traj_mod=traj_mod)

    algo = _MPOPT_TFWA()
    p = algo.default_params()
    p["seed"] = int(seed)
    p.update(params)
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
    }
