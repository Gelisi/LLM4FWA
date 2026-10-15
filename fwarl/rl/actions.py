"""动作空间：RL 在 TFWA 上可施加的每代决策。

真实 TFWA 的探索/开发旋钮是 t 分布自由度 df：
  - df 越小 -> 尾部越重 -> 越探索；
  - df 越大 -> 越接近高斯 -> 越收敛（原版按 factors=[1.05, 10.0] 单调增长）。

动作在每代开始前施加到各烟花的 dist.df 上（原版 adapt() 随后仍按 factors 增长）：
  - converge   : df *= 2*factors  （加速收敛）
  - normal     : 不干预（等价原版行为）
  - slow       : df /= factors    （抵消本代增长）
  - heavy_tail : df 减半          （加压探索）
  - reset      : df 回到 init_df  （回到重探索）
"""
from __future__ import annotations

ACTIONS = [
    {"name": "converge", "df_mult": None, "use_factor": 2.0},
    {"name": "normal",   "df_mult": None, "use_factor": 0.0},   # 0 表示不干预
    {"name": "slow",     "df_mult": None, "use_factor": -1.0},  # -1 表示除以 factors
    {"name": "heavy_tail", "df_mult": 0.5, "use_factor": 0.0},
    {"name": "reset",    "df_mult": None, "use_factor": None},  # None 表示回到 init_df
]

N_ACTIONS = len(ACTIONS)


def apply_action(pops, factors, init_df, action: dict):
    """把动作施加到各烟花的分布自由度上（每烟花分别处理）。"""
    for i, pop in enumerate(pops):
        df = pop.dist.df
        f = factors[i]
        if action["use_factor"] is None:
            new_df = init_df
        elif action["df_mult"] is not None:
            new_df = int(df * action["df_mult"])
        elif action["use_factor"] > 0:
            new_df = int(df * f * action["use_factor"])
        elif action["use_factor"] < 0:
            new_df = int(df / f)
        else:
            continue  # normal：不干预
        pop.dist.df = max(2, min(int(new_df), 2**62))
