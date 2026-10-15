# iter_000: 代码改动清单（CODE）

## 改动文件

| 文件 | 改动类型 | 内容摘要 | 影响基线可比性 |
|------|----------|----------|----------------|
| fwarl/bench.py | 新增 | CEC 问题统一封装（复用 reference_algorithms .so） | - |
| fwarl/tfwa.py | 新增 | 原版 TFWA 薄适配（import MPOPT）+ SafeEvaluator 修复 mpopt 轨迹除零 bug | 否 |
| fwarl/fwa_rl.py | 新增 | TFWA_RL 子类：每代 RL 决策 dist.df 干预（[RL] 钩子，主循环与原版一致） | 否 |
| fwarl/rl/actions.py | 新增 | df 干预动作空间（5 动作） | - |
| fwarl/rl/learner.py | 新增 | 线性 Q agent（持久化/加载） | - |
| fwarl/rl/features.py | 新增 | 状态特征（v1，部分被 fwa_rl 内置特征取代，保留备用） | - |
| experiments/run_eval.py | 新增 | 多进程评估器（单写者落盘、suspicious 重试、子进程隔离） | - |
| experiments/sync_baselines.py | 新增 | 12 对手逐 run 成绩快照（samples.json） | - |
| experiments/summarize.py | 新增 | Wilcoxon 逐对手检验 + 竞争排名 + status.json | - |
| experiments/train_policy.py | 新增 | 轨迹离线聚合为策略 | - |

## 关键工程决策

1. TFWA 直接 import MPOPT 原实现，零重写（曾误按 Li&Tan 机制重写，已纠正，旧结果作废）。
2. 每个 (algo,bench,func,dim) 在全新子进程中评测：libcec*.so 同进程切函数存在
   非确定性状态污染（实测复现）。
3. error<0 → suspicious → 子进程内重试 ≤3 次：.so 数据加载偶发失败。
4. worker 不写盘、主进程统一落盘：ceph-fuse 多进程并发写会丢文件（实测复现）。

## 回滚方式

整个目录为初始状态，无回滚需求。
