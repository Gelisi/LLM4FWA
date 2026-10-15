# AGENT_GUIDE — LLM Agent 迭代协议

> 本文件定义 autoresearch 每一轮迭代的标准流程。任何 agent（人或 LLM）接手时，
> **先读 `research_log.md` 的最后 3 轮记录**，再按本协议工作。

## 迭代目录规范（iterations/iter_XXX/）

每个 iter 目录**固定四件套**（模板在 `iterations/_TEMPLATE/`，新 iter 直接复制改名）：

| 文件 | 内容 | 谁写 |
|------|------|------|
| `IDEA.md` | 动机/证据（引用数据!）、可证伪假设、单一变量改动、预期、实验设计 | 立项时 |
| `CODE.md` | 改动文件清单、动作/特征/超参定义、回滚方式、自查清单 | 改码时 |
| `RESULTS.md` | 运行记录表、主结果、统计检验、结论（负结果也要写） | 实验后 |
| `diag_*.json` 等 | 支撑证据的原始数据（诊断/轨迹摘要） | 随实验 |

规则：
- iter 目录名 `iter_{XXX}_{slug}`（slug 简短描述机制，如 `iter_001_rl_restart`）；
- `IDEA.md` 头部状态字段：PROPOSED → RUNNING → DONE / REJECTED / SUPERSEDED；
- 每轮结束必须回填 RESULTS.md 并在 research_log.md 追加一行；
- 引用数据要给路径（results/... 或 diag_*.json），便于下一个 agent 复核。

## 每轮迭代流程

```
1. [读状态]   research_log.md 末尾 + 各 iter_XXX/IDEA.md 的状态字段
2. [选假设]   从 RESEARCH_PLAN.md 当前阶段 / 上一 iter 的 NEXT 选择下一个假设
3. [建目录]   cp -r iterations/_TEMPLATE iterations/iter_{XXX}_{slug}
              填写 IDEA.md（状态=PROPOSED，必须含可证伪预期与证据引用）
4. [改代码]   修改 fwarl/（保持向后兼容），填写 CODE.md 与自查清单
5. [跑实验]   先 smoke（configs/smoke.json）验证不崩，再按 IDEA.md 的 SETUP 跑
6. [做汇总]   experiments/summarize.py 生成 report/status；回填 RESULTS.md
7. [下结论]   更新 IDEA.md 状态为 DONE/REJECTED；research_log.md 追加一行
8. [做决策]   保留 / 回滚 / 进入下一假设（写进 RESULTS.md 的结论）
```

## 硬性规则

1. **可复现**：每条实验命令必须完整写入 notes.md；种子协议见 RESEARCH_PLAN.md。
2. **单一变量**：一轮迭代只引入一个机制/参数变化，否则结论无法归因。
3. **不覆盖**：`results/<algo>_<tag>/` 用新 tag，绝不覆盖历史结果；`baselines/` 只读。
4. **不破坏基线可比性**：评测协议（MaxFES/维度/runs/error 定义）不得单方面更改；
   如确需更改，必须在 research_log.md 顶部的「协议变更」节记录，并重跑受影响对比。
5. **先 smoke 后全量**：任何代码改动先过 `configs/smoke.json`。
6. **诚实汇报**：失败/变差的实验同样记录在 research_log.md（负结果是重要信息）。

## 工具速查

```bash
# 基线快照（只在 reference_algorithms/experiment_results 更新后重跑）
python3 experiments/sync_baselines.py

# 冒烟
python3 experiments/run_eval.py --config configs/smoke.json

# 训练（在线 RL + 保存轨迹）
python3 experiments/run_eval.py --algo fwa_rl --bench cec2022 --dim 10 --runs 30 --train --tag train_c22d10

# 策略聚合
python3 experiments/train_policy.py --glob "results/fwa_rl_train_c22d10/*/traj_*.json"

# 冻结策略评测
python3 experiments/run_eval.py --algo fwa_rl --bench cec2022 --dim 10 --runs 30 --tag eval_c22d10

# 汇总对比
python3 experiments/summarize.py --algo fwa_rl --tag eval_c22d10 --bench cec2022 --dim 10
```

## 判定标准速查

- 逐对手对比：每个问题上 Wilcoxon 秩和检验（双侧，α=0.05），
  '+' 我们显著优 / '=' 无显著差异 / '-' 我们显著劣
- 综合实力：每个问题上按 mean error 在（我们+12 对手）中排名，取平均 → 各 benchmark 名次
- **总目标（阶段 5）**：每个 benchmark 平均排名 ≤ 2（Top-2）
- 每轮迭代后检查（agent 停止条件）：
  ```
  python3 experiments/summarize.py --algo fwa_rl --tag <tag> --target-rank 2
  cat results/fwa_rl_<tag>/status.json   # achieved 字段全 true 即达标
  ```

## 迭代循环示例（阶段 2-5 通用）

```
while not (status.json 所有 benchmark achieved):
    1. 找最弱 benchmark（status.json 中 bench_ranks 最大者）
    2. 看 pairwise.csv：被谁压制？在哪类函数（unimodal/multimodal/hybrid/composition）？
    3. 从 RESEARCH_PLAN 阶段 4 选一个针对性机制，写假设到 iterations/iter_XXX/
    4. 改代码 → smoke → 训练（--train）→ 聚合策略（train_policy.py）→ 冻结评测
    5. summarize.py 更新 status.json，记录结论
```
