# iter_000: 实验结果（RESULT）

## 运行记录

| 命令 | 用途 | 状态 |
|------|------|------|
| `python3 experiments/sync_baselines.py` | 基线快照 | 92879 条，12 对手×258 问题 |
| `python3 experiments/run_eval.py --algo tfwa fwa_rl --bench cec2022 --dim 10 --runs 5 --workers 16 --train --tag real_tfwa_c22d10` | 首轮对照 | ok=120, bad=0, 109s |
| `python3 experiments/summarize.py --algo fwa_rl --tag real_tfwa_c22d10 --target-rank 2` | 统计报告 | 报告+status.json 已生成 |

## 主结果

见 IDEA.md §3 的表格；完整报告：
- `results/tfwa_real_tfwa_c22d10/report.md`
- `results/fwa_rl_real_tfwa_c22d10/report.md`（pairwise.csv / status.json 同目录）

## 统计检验

- FWA_RL vs 12 对手（CEC2022 D10, 5 runs）：
  - 压制：de_mmogc（12+/0−）、rcmaes（11+/1−）
  - 被压制：rdex（0+/6−）、lshade_rsp / nl_shade_rsp（各 7−）
  - 综合：7.08/13，Top-2 未达成
- 注意：5 runs 下 Wilcoxon 功效不足（大量 '='），正式对标需 30 runs。

## 结论

- [x] 支持假设（适配层等价、RL 冷启动不崩）
- [ ] RL 增益：未证实（5胜6平1负，F7 显著变差）
- 原因：干预过频 + 状态特征未含 no_improve/重启信息 + df 单一旋钮太弱
- **NEXT** → iter_001（重启 RL 化，证据见该目录 IDEA.md）
