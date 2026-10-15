# FWA + RL Autoresearch：基于 TFWA 的烟花算法 + 强化学习

> 目标：在 CEC2013/2017/2019/2021/2022 全系列 benchmark 上，让 **TFWA + RL** 媲美甚至超越
> 当前最强的 CEC 竞赛算法（EA4eig、L-SRTDE、RDEx、NL-SHADE-RSP、MadDE、AGSK、IMODE、
> jDE21、mLSHADE-RL、LSHADE-RSP、RCMAES、DE-MMOGC）。
> 项目为 **LLM Agent autoresearch** 设计：每轮迭代 = 提出假设 → 修改代码 → 实验 → 记录结论。

---

## 目录结构

```
autoresearch/
├── README.md                  # 本文件
├── RESEARCH_PLAN.md           # 研究路线图（阶段划分与里程碑）
├── AGENT_GUIDE.md             # LLM Agent 迭代协议（每轮做什么/产出什么）
├── requirements.txt
│
├── fwarl/                     # 核心算法库
│   ├── bench.py               #   CEC benchmark 统一封装（复用 reference_algorithms 的 .so）
│   ├── tfwa.py                #   TFWA 基线（Time-Varying FWA, Li & Tan, IEEE TEVC 2022 风格）
│   ├── fwa_rl.py              #   FWA+RL 主算法：TFWA 骨架 + 每代 RL 决策
│   └── rl/
│       ├── features.py        #   状态特征（progress/diversity/stagnation/...）
│       ├── actions.py         #   动作空间（爆炸强度/火花分配/高斯变异率）
│       └── learner.py         #   线性 Q-learning agent（可持久化、可跨 run 聚合）
│
├── experiments/
│   ├── run_eval.py            #   统一评估：算法 × benchmark × 函数 × 维度 × runs（多进程）
│   ├── summarize.py           #   汇总结果 + 与 12 个最强方法的对比表/排名/差距
│   ├── sync_baselines.py      #   从 reference_algorithms/experiment_results 提取基线快照
│   └── train_policy.py        #   聚合训练轨迹，更新 RL 策略（离线合并，避免并发写冲突）
│
├── configs/
│   ├── default.json           #   全量实验配置
│   └── smoke.json             #   快速冒烟测试配置
│
├── baselines/                 # 最强方法成绩快照（sync_baselines.py 生成）
├── results/                   # 实验输出（JSON per run + 汇总报告）
├── policies/                  # RL 策略权重（跨迭代累积）
├── research_log.md            # 研究总日志（每轮追加）
└── iterations/iter_XXX/       # 每轮迭代的假设/改动/结果/结论
```

## 快速开始

```bash
cd /mnt/new_gemini/shipengcen/autoresearch

# 0) 生成基线快照（12 个最强方法在 5 个 CEC benchmark 上的成绩）
python3 experiments/sync_baselines.py

# 1) 冒烟测试：TFWA 与 FWA+RL 各跑少量问题（约1分钟）
python3 experiments/run_eval.py --config configs/smoke.json

# 2) 单独跑某个 benchmark 全量（例：CEC2022, D=10, 30 runs）
python3 experiments/run_eval.py --algo fwa_rl --bench cec2022 --dim 10 --runs 30 --train

# 3) 聚合训练轨迹 → 更新策略
python3 experiments/train_policy.py

# 4) 汇总对比：与最强方法的差距表
python3 experiments/summarize.py --algo fwa_rl --bench cec2022 --dim 10
```

## 算法设计概要

### TFWA 基线（`fwarl/tfwa.py`）
**用户自研算法**（原实现：`../MPOPT/mpopt/algorithms/TFWA.py`），本项目只做薄适配
（直接 import MPOPT，用 `ObjFunction`/`SafeEvaluator` 包装 CEC 问题），保证与原版零偏差：
- 2 个烟花，每个烟花是一个 **TMA-ES 式 EDA 种群**（TMAESPop），
  采样分布为**多元 t 分布**（MVTD，含自由度 df）；
- 两个烟花的 **df 增长因子** factors=[1.05, 10.0]：一个慢收敛、一个快收敛，
  df 增大 → t 分布趋近高斯（开发），df 小 → 重尾（探索）；
- **稳健加权**（按 t 分布 Mahalanobis 距离加权重排）+ 自由度消融开关（fixed_nu）；
- **重启机制**（改进量×剩余迭代 < 最优差距，或连续 100 代无改进）；
- 越界采样 uniform random remap。
- 注：mpopt `Evaluator` 轨迹记录有除零 bug，适配层以 `SafeEvaluator` 子类修复
  （仅改 `% num_eval` → `% traj_mod` 一处，其余一致）。

### FWA + RL（`fwarl/fwa_rl.py`）
子类化原版 TFWA，核心循环保持一致（仅加 `[RL]` 标注的钩子）：
- **决策点**：每代开始前，RL 决定本代各烟花 `dist.df` 的干预方式
  （converge / normal / slow / heavy_tail / reset，见 `rl/actions.py`）——
  df 是 TFWA 探索-开发的天然旋钮；
- 状态 `s`（7 维）：进度、改进 EMA、scale/span、停滞度、df 相对量、烟花间适应度差；
- 奖励 `r`：全局最优的对数改进；
- 学习器：线性 Q(s,a) + ε-greedy（`rl/learner.py`），轨迹落盘由
  `train_policy.py` 离线合并进 `policies/fwa_rl_policy.json`。

### RL 训练协议
1. **探索阶段**：`--train` 模式跑大量 (benchmark × 函数 × 维度) 实例，agent 在线学习并
   保存每条轨迹末端的权重；
2. **聚合阶段**：`train_policy.py` 将所有轨迹按序回放更新，得到一个可跨问题泛化的策略；
3. **评测阶段**：不带 `--train`，加载冻结策略评测，与基线对比。

## 与现有资产的关系

| 资产 | 位置 | 用途 |
|------|------|------|
| **TFWA 原实现** | `../MPOPT/mpopt/algorithms/TFWA.py`（含 TMAESPop、MVTD） | 直接 import，零偏差使用 |
| CEC benchmark .so 接口 | `../reference_algorithms/cec_benchmark/` | 直接 import 复用 |
| 12 个最强方法成绩 | `../reference_algorithms/experiment_results/` | 基线对比（sync_baselines.py） |
| 12 个最强方法 C++ 源码 | `../reference_algorithms/adapted_algorithms/` | 机制借鉴 |
| FWA 四算子框架 | `../fwa_fm/fwa_runner.py` | 可选：把我们的算子适配过去 |
| mLSHADE-RL（RL+DE 先例） | `../reference_algorithms/lshade_rl.py` | RL 机制参考 |

## 注意事项

- `../.codebuddy` 目录是项目数据，勿删。
- benchmark 数据目录（input_data）依赖 `reference_algorithms` 的相对路径，勿移动 `.so`。
- 结果 JSON 格式与 `reference_algorithms/experiment_results` 保持一致，方便统一汇总。
