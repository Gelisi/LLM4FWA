# iter_002: RL 替换 TFWA 自由度更新

> 状态：RUNNING
> 日期：2026-10-15
> 前置：iter_000 的 df 干预实验；iter_001 的重启方案暂缓

## 1. IDEA（假设）

- 动机/证据：iter_000 的 RL 动作仍叠加原生 factors 更新，冻结入口不传策略会固定选首动作。iter_001 又转向重启并冻结 df，偏离本轮自由度控制目标。
- 假设：关闭原生 df 更新，将增大、保持、减小、重尾、复位的选择交给 Q-learning；跨函数训练后，冻结策略至少在一个 CEC2022 短板问题上改善 error，同时必须证明每代实际 df 等于策略指令。
- 单一研究变量：自由度更新模块。采样、协方差、步长、火花数及重启判定沿用原版；重启保留策略 df，不再偷偷重置自由度。
- 预期可证伪：若 30 runs 配对检验无改善或出现显著退化，则不宣称优于原版，下一轮从动作分布、停滞奖励及状态表达分析原因。
- 当前为每代共享一个动作控制两个烟花；尚未声称实现逐烟花独立控制或连续动作策略。

## 2. SETUP（实验设计）

- 配置：本目录 experiment.json；workflow 自动保存实际配置、命令、退出码和耗时。
- 训练：CEC2013 F6/F7/F8 D10，各 5 runs，MaxFES=20000，seed0=42；完整轨迹离线回放，策略版本 tfwa_df_replace_v1。
- 冻结测试：CEC2022 F9/F10/F12 D10，各 30 runs，MaxFES=200000，seed0=4242；fwa_rl 与原版 tfwa 使用同种子同预算。
- smoke：一个训练函数、1 run、2000 FES，并运行控制权/冻结策略/workflow 门槛单元测试。
- 统计：逐问题配对 Wilcoxon，alpha=0.05，注明未做多重校正；同时输出 mean/std 和动作分布。只覆盖三个测试问题，不作为 CEC2022 全集或五个 benchmark Top-2 结论。
- 执行：workflow_start → IDEA review → 代码反思 review → smoke → train → freeze → collect → upload。
- 失败循环：保存错误日志；修订代码与 CODE.md，重新 review，从 smoke 开始重新训练和评测。不得以立项或编译代替实验。

## 3. RESULT

真实实验由 workflow 写入 RESULTS.md 与 summary.json，原始数据使用独立 tag 保留。

## 4. CONCLUSION

控制权替换成功：90 个 RL run、290317 次代级决策均无原生 df 覆盖。性能改善假设尚不被本轮数据支持：F9/F10/F12 各 30 runs 均无显著收益，F10 mean error 上升；详见 RESULTS.md 和 summary.json。控制权正确不等于优化效果更好。

## 5. NEXT

依据本轮结果决定是否研究停滞奖励、决策周期或逐烟花动作，不转移到重启模块。
