# 研究路线图（RESEARCH_PLAN）

> 供 LLM Agent autoresearch 使用。每完成一个里程碑，在 `research_log.md` 记录，
> 并在 `iterations/iter_XXX/` 建立对应目录。

## 背景与差距

- **目标算法**：TFWA（用户自研，`../MPOPT/mpopt/algorithms/TFWA.py`：双烟花 TMA-ES
  + 多元 t 分布采样 + df 自适应 + 稳健加权 + 重启）+ RL（本项目）。
- **对手**：12 个 CEC 竞赛顶级算法（基线快照见 `baselines/`，来源
  `../reference_algorithms/experiment_results/`，评测协议：MaxFES = 10000×D，
  30 次独立运行，指标 = error = f(x) - f(x*)）。
- **现状**：真实 TFWA 已相当强（CEC2022 F1 可精确求解），但在部分
  hybrid/composition 函数与高维上与最强方法仍有差距。
- **核心假设**：TFWA 的探索-开发旋钮（df 自由度增长节奏、重启时机/方式、
  scale、火花数分配）存在自适应空间，RL 可按问题状态动态决策，
  进一步逼近/超越 success-history 自适应的 DE 系方法。

## 阶段规划

### 阶段 0：基础设施（本项目骨架，已完成）
- [x] TFWA 基线（直接 import MPOPT 原实现，零偏差适配）
- [x] FWA+RL 框架（RL 挂接 df 旋钮，策略可持久化）
- [x] 统一评估器 + 基线快照 + 汇总对比
- [x] 冒烟测试通过

### 阶段 1：TFWA 基线确认
目标：确认适配层与原实现等价（同 seed 同结果），并摸清 TFWA 在各 benchmark 的强弱点。
- 复核：CEC2022 D=10 上适配层 vs 直接跑 MPOPT examples 的结果一致性
- 弱点画像：哪些函数/维度 TFWA 落后最强对手最多（决定主攻方向）
- 判定：适配层无 bug（等价性抽查通过）
- 产出：`iterations/iter_001/`

### 阶段 2：RL 冷启动与首轮增益
目标：FWA+RL（随机初始化策略在线学习）显著优于原版 TFWA。
- 实验：同一批问题（建议先用 CEC2022 D=10 全 12 函数，30 runs）tfwa vs fwa_rl --train
- 判定：fwa_rl 在 ≥60% 函数上 mean error 不劣于 tfwa 且无显著变差的函数
  （Wilcoxon 秩和检验 p<0.05）
- 风险：奖励稀疏/尺度问题 → 检查 reward 曲线，必要时改用 rank-based 奖励

### 阶段 3：策略聚合与跨问题泛化
目标：离线聚合的策略（policies/fwa_rl_policy.json）在未训练问题上优于在线学习。
- 实验：CEC2013/2017 训练 → CEC2021/2022/2019 零样本迁移评测
- 判定：迁移性能 ≥ 在线随机初始化性能
- 扩展方向：状态特征加入问题类型指示（用无监督特征而非 one-hot 泄露）

### 阶段 4：机制增强（每轮迭代一个假设）
按优先级逐个引入，**每次只改一个变量**（RL 动作空间随之扩展）：
1. **重启策略 RL 化**：重启时机（原版固定阈值）与重启方式（随机 shift vs 保留 shift
   vs 精英扰动）交由 RL 决策
2. **scale 步长 RL 化**：TMA-ES 的 scale 更新倍率按状态调整（早扩晚缩）
3. **火花数分配**：两烟花的 sp_size 配比与总预算按进度调整
4. **稳健加权/df 消融的组合策略**：RL 按函数类型切换 fixed_nu 与 use_robust_weight
5. **末段局部搜索**（借鉴 mLSHADE-RL 的 L-BFGS-B）：RL 控制触发时机与预算
- 判定：每项在 CEC2022 D=10/20 上 mean error 不劣化（配对检验）才保留

### 阶段 5：全量对标 —— 迭代总目标（Top-2）
**最终目标：在每个 benchmark 上综合实力进入 Top-2（13 个算法中平均排名 ≤ 2）**。
- 全部 5 个 benchmark × 全部维度 × 30 runs
- 比较方式：与 12 个对手逐算法、逐问题做 Wilcoxon 秩和检验（+/−/=，α=0.05），
  汇总为逐对手胜负表；每个问题上按 mean error 排名，取平均得各 benchmark 综合名次
- 每轮迭代后运行 `experiments/summarize.py`，用 `results/<algo>_<tag>/status.json`
  检查达成状态（agent 循环的停止条件：所有 benchmark achieved == true）
- 中间里程碑：CEC2022 Top-2 → CEC2021+2019 Top-2 → CEC2013+2017 Top-2
- 若长期卡在 Top-3 之外：分析 pairwise.csv 中被谁压制、在哪类函数上压制，
  针对性进入阶段 4 的对应机制

## 评测协议（与基线严格一致）

| 项 | 取值 |
|----|------|
| MaxFES | 10000×D（CEC2022 为 20000×D，见 configs） |
| 独立运行 | 30（冒烟阶段可 5） |
| 指标 | error = best_f - optimal；报告 mean/std；Wilcoxon 秩和 + Friedman 排名 |
| 种子 | run_id → seed 固定映射，保证可复现 |

## 已知风险

1. **纯 Python 性能**：300k FES 的 Python 循环较慢（约 3-10s/run），30 runs×30 问题
   需要并行（run_eval.py 已支持 32 核）。若过慢，将热路径编译为 C++（参考
   `../reference_algorithms/adapted_algorithms/` 的接口模式）。
2. **CEC2013 F2 数值不稳定**：基线也跳过则跟随；对比时注意剔除说明。
3. **CEC2019 特殊维度/边界**：F1(9D,±8192)、F2(16D,±16384)、F3(18D,±4)，
   幅度初始化必须按 span 缩放（实现已按 span 处理）。
4. **RL 崩坏**：若策略坍缩到单一动作，检查 ε 下限与权重范数，必要时加熵正则。
