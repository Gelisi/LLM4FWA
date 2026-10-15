# iter_000: 项目初始化与首轮基线

> 状态：DONE
> 日期：2026-10-14
> 前置：无（项目起点）

## 1. IDEA（假设）

- **动机/证据**：目标是在各 CEC benchmark 上把 TFWA+RL 迭代到综合 Top-2。
  首先需要零偏差接入原版 TFWA 并建立可比的评测/统计链路。
- **假设**：适配层能复现原版 TFWA 行为，且 RL 挂接 df 旋钮的冷启动不劣于原版。
- **改动**：项目初始化（算法库 + 评估器 + 基线提取 + 统计汇总）。
- **预期**：冒烟通过；FWA+RL ≈ TFWA（±噪声）。

## 2. SETUP（实验设计）

```
python3 experiments/run_eval.py --algo tfwa fwa_rl --bench cec2022 --dim 10 \
    --runs 5 --workers 16 --train --tag real_tfwa_c22d10
python3 experiments/summarize.py --algo fwa_rl --tag real_tfwa_c22d10 --target-rank 2
```

## 3. RESULT

- 120/120 成功。CEC2022 D10（5 runs，噪声大仅作方向参考）：

| F | TFWA | FWA_RL | 最强基线 |
|---|------|--------|----------|
| 1/2/5/11 | 0 | 0 | 0 |
| 3 | 9.2e-7 | 1.4e-7 | 0 |
| 4 | 4.58 | 3.78 | 8.24（我们超基线） |
| 6 | 1.40 | 1.02 | 65.3 |
| 7 | **0.72** | 7.52 | 0 |
| 9 | 229 | 229 | 0 |
| 10 | 164 | 121 | 0 |
| 12 | 63.8 | 63.5 | 62.7 |

- 统计报告：FWA_RL 对 TFWA 5胜/6平/1负；综合排名 7.08/13（Top-2 未达成）。
- F7 变差（0.72→7.52）：RL 干预过频（约半数代选了非 normal 动作），破坏原版 df 节奏。

## 4. CONCLUSION

- 适配层正确；RL 冷启动无净收益（预期内，阶段 2 判定未过）。
- 主攻弱点明确：F9/F10（composition/hybrid）。
- 工程沉淀（见 research_log 协议变更节）：子进程隔离 .so 状态污染、
  suspicious 重试、单写者落盘、SafeEvaluator 修复 mpopt 除零 bug。

## 5. NEXT

→ iter_001：重启机制 RL 化（动机与诊断证据见 iterations/iter_001_rl_restart/IDEA.md）。
另需：30 runs 复核首轮结论；RL 降低干预频率（提高 normal 先验 / 加干预代价）。
