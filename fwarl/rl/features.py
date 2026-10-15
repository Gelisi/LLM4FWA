"""状态特征：把 FWA 的搜索状态映射为定长特征向量，供线性 Q 使用。

设计原则：
  - 全部特征归一化到大致 [-1, 1]，跨 benchmark / 维度可比；
  - 不使用问题编号等泄露信息；
  - 特征即接口：新增特征只需扩展 N_FEATURES 并保持历史兼容（load 时按长度校验）。

特征列表（顺序即向量顺序）：
  0. bias = 1.0
  1. progress：评估预算消耗比例 nfes/max_evals
  2. improve：近期对数误差改进（EMA，截断到 [-1,1]；越大代表改进越快）
  3. diversity：种群归一化平均标准差 / 平均 span
  4. stagnation：距上次全局最优更新已消耗的预算比例（截断到 1）
  5. amp_ratio：当前平均爆炸幅度 / 平均 span（对数压缩）
  6. rel_improve_trend：近期改进减去更早期改进（加速度，正=在加速）
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 7


class FeatureExtractor:
    """跨代维护 EMA / 停滞计数等内部状态。每个 run 一个实例。"""

    def __init__(self, max_evals: int, ema_beta: float = 0.3):
        self.max_evals = max(1, int(max_evals))
        self.ema_beta = ema_beta
        self._ema_improve = 0.0
        self._early_improve = 0.0
        self._last_best_f = None
        self._last_best_evals = 0
        self._have_early = False

    def update(self, best_f_prev: float, best_f_now: float, nfes: int):
        """每代结束后调用一次，记录全局最优的变化。"""
        if self._last_best_f is None:
            self._last_best_f = best_f_now
            return
        # 对数尺度改进（误差可为 0，加 1e-12 平滑；只看相对下降）
        ratio = (best_f_prev + 1e-12) / (best_f_now + 1e-12)
        r = float(np.clip(np.log(max(ratio, 1e-12)), 0.0, 10.0) / 10.0)
        self._ema_improve = (1 - self.ema_beta) * self._ema_improve + self.ema_beta * r
        if not self._have_early and nfes >= self.max_evals * 0.25:
            self._early_improve = self._ema_improve
            self._have_early = True
        if best_f_now < self._last_best_f:
            self._last_best_f = best_f_now
            self._last_best_evals = nfes

    def compute(self, nfes: int, best_f: float, population: np.ndarray,
                amp: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        span = ub - lb
        mean_span = float(np.mean(span))

        pop = np.asarray(population, dtype=float)
        diversity = float(np.clip(np.mean(np.std(pop, axis=0)) / mean_span, 0.0, 1.0))

        stagnation = float(min(1.0, (nfes - self._last_best_evals) / self.max_evals))

        amp_ratio = float(np.clip(np.log1p(np.mean(amp) / max(mean_span, 1e-12) * 100.0) / np.log(101.0), 0.0, 1.0))

        trend = float(np.clip(2.0 * (self._ema_improve - self._early_improve), -1.0, 1.0))

        return np.array([
            1.0,
            nfes / self.max_evals,
            self._ema_improve,
            diversity,
            stagnation,
            amp_ratio,
            trend,
        ], dtype=float)


def reward_log_improvement(best_f_prev: float, best_f_now: float,
                           scale: float = 10.0) -> float:
    """对数改进奖励：log10 相对下降，截断到 [-0.5, 1]。

    无改进时接近 0（惩罚轻微负值，抑制无意义震荡）。
    """
    if best_f_now >= best_f_prev:
        return -0.1 if best_f_now > best_f_prev else 0.0
    ratio = (best_f_prev + 1e-12) / (best_f_now + 1e-12)
    r = float(np.log(max(ratio, 1.0))) / np.log(scale)
    return float(np.clip(r, -0.5, 1.0))
