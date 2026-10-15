
## Workflow 实测汇总

仅当前配置覆盖问题；不代表全 benchmark 达标

| 问题 | runs | TFWA mean error | RL mean error | p | +/=/- |
|---|---:|---:|---:|---:|---|
| cec2022/F10_D10 | 30 | 107.379 | 114.438 | 0.2206 | = |
| cec2022/F12_D10 | 30 | 64.3275 | 64.3813 | 0.9354 | = |
| cec2022/F9_D10 | 30 | 229.332 | 229.34 | 0.1048 | = |

检验为逐问题未校正配对 Wilcoxon，不据此宣称全局优越。
策略：`iterations/iter_002_rl_df_control/artifacts/policy_1127b5c3acbb_7.json`；完整命令、退出码、耗时见 `workflow.json`。

## 运行与控制权证据

- 13 项单元回归测试通过，涵盖原生 adapt/重启不能覆盖 RL df、冻结权重不变、真实训练更新和 ReAct 提前 finish 拒绝。
- smoke：1 run；训练：15 runs，ok=15/bad=0，评测器耗时约 7 秒；离线轨迹回放成功。
- 冻结正式评测：原版与 RL 共 180 runs，ok=180/bad=0，评测器耗时约 309 秒；每问题每算法均 30 runs。
- 90 个 RL run 的 290317 次代级决策，commanded/after 不一致次数为 0；实际 df 范围为 2 到 2^62，原规则已不再拥有 df 更新权。
- 冻结动作次数，顺序为 increase/hold/decrease/heavy_tail/reset：F9=[83744,0,0,13036,0]，F10=[66749,0,0,30014,0]，F12=[71342,0,0,25432,0]。
- workflow 自动核验预算、配对种子、结果状态、冻结策略文件未变化，并保留独立实验 tag，不覆盖历史数据。

## 结论与反思

控制权替换成功，但本轮没有实证支持优化性能改善：三个测试问题的配对 p 值均大于 0.05；F10 的 mean error 由 107.379 上升到 114.438。不能把“未显著变差”解释为等价或不劣。
策略只使用增大和强重尾两个动作，未使用保持、缓降与复位；df 还到达极大上界，说明当前奖励/状态与动作语义可能产生饱和和策略集中。具体因果尚未验证。
本轮训练只有三个函数各 5 runs、每 run 20000 FES，属于小规模冷启动；三个短板函数的测试不能代表整个 CEC2022，更不能代表五个 benchmark Top-2。

## NEXT

下一轮仍只研究 df：优先提出“限制有效 df 范围或逐烟花独立状态/动作”中的一个单变量假设，对照本轮已冻结策略及原版 TFWA，重新训练并按同种子 30 runs 评测。不得转移到 RL 重启，也不直接复用改变动作语义前的旧策略。

## 复现与上传

准备本目录 experiment.json 和 IDEA/CODE 后，由 workflow_start、两次 workflow_review 进入 smoke；依次执行 workflow_run 的 smoke/train/freeze/collect/upload。CLI 同样可用 `python3 agent/workflow.py --iteration iterations/iter_002_rl_df_control --stage <阶段>`。
上传是最后独立阶段；失败保持 upload 可重试，不伪报已上传。原始 run/轨迹按现有 .gitignore 留本地，代码、配置、策略与统计汇总提交到仓库。
