# iter_001: 代码改动清单（CODE）

> 状态：设计稿——实现时勾选自查清单

## 改动文件

| 文件 | 改动类型 | 内容摘要 | 影响基线可比性 |
|------|----------|----------|----------------|
| fwarl/restart_rl.py | 新增 | RestartRL 子类：TFWA_RL 基础上，df 动作冻结为 normal，重启决策 RL 化 | 否（新 tag） |
| fwarl/rl/actions.py | 修改 | 新增 RESTART_ACTIONS（不动原 df 动作，避免破坏 iter_000 复现） | 否 |
| fwarl/rl/features.py | 修改 | 状态特征增厚：no_improve_count、重启次数、scale/span、df、两烟花 fit 差 | 否 |
| experiments/run_eval.py | 修改 | algo 新增 `fwa_rl_rs`（或以 policy 区分） | 否 |

## 动作空间设计（三层，联合编码为一个动作 ID 或 2~3 个独立 head，实现取简单方案）

- **时机**：抑制本轮条件重启 / 正常 / 立即重启该烟花（×阈值倍率留待 iter 后续）
- **落点**：随机 shift / 保留 shift / 全局 best 邻域（小 scale 重启）/ 另一烟花位置加噪
- **记忆量**：scale 重置 {init, init/2, 当前}；df 重置 {init_df, 当前}

组合数 = 3×4×3×2 = 72，过多；**先裁剪为 8 个语义动作**（如 "抑制一轮" /
"标准重启" / "best 邻域小 scale 重启" / "保留位置重置 scale" / "交叉重启" ...），
策略证明有效后再细分。

## 状态特征（v2，7+5 维）

新增：no_improve_count/100、重启次数对数、scale/span（各烟花）、df/init_df、
两烟花 fitness 差归一化。保留：progress、improve EMA、停滞度。

## 奖励（防 hacking）

主奖励：对数改进（同 iter_000）；
新增**重启奖励项**：重启后 W 代内改进 >0 给小正奖励，否则小负奖励
（让 agent 学会"重启要看地方"）；干预代价项暂不加（df 已冻结 normal）。

## 自查清单

- [ ] 等价性抽查：动作全部强制 normal+不干预重启时，输出与原版 TFWA 同 seed 一致
- [ ] smoke 通过
- [ ] 训练轨迹 action 分布非坍缩（各动作均有选择）
- [ ] 30 runs 正式对比 + Wilcoxon
