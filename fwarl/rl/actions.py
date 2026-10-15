"""RL 直接控制 t 分布自由度，不叠加原版 factors 增长。

每代选择增大、保持、减小、重尾或复位；倍率只定义动作语义，
动作的选择由学习策略决定。fixed_nu 锁定本代指令，防止 adapt 再更新 df。
"""
from __future__ import annotations

POLICY_VERSION = "tfwa_df_replace_v1"
DF_MIN = 2
DF_MAX = 2**62

ACTIONS = [
    {"name": "increase", "df_mult": 2.0},
    {"name": "hold", "df_mult": 1.0},
    {"name": "decrease", "df_mult": 0.5},
    {"name": "heavy_tail", "df_mult": 0.25},
    {"name": "reset", "df_mult": None},
]

N_ACTIONS = len(ACTIONS)


def apply_action(pops, init_df, action: dict):
    """更新 df，并通过原版 population 的固定自由度接口锁定本代 df。"""
    for pop in pops:
        mult = action["df_mult"]
        new_df = init_df if mult is None else pop.dist.df * mult
        pop.dist.df = max(DF_MIN, min(int(new_df), DF_MAX))
        pop.fixed_nu = pop.dist.df
