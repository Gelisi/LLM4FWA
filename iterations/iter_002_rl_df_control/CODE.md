# iter_002: 自由度控制实现与反思

## 代码变更

- fwarl/rl/actions.py：五个动作独立控制 df，不再使用原生 factors；fixed_nu 锁定当前代指令。
- fwarl/fwa_rl.py：重启保留策略 df，全局最优计算奖励，EMA 每代更新一次，记录 before/commanded/after 并断言指令未被覆盖。
- fwarl/rl/learner.py：策略语义版本保存和加载校验；旧干预策略不能直接复用。
- experiments/run_eval.py：配置/CLI 优先级修正，冻结评测必传策略，spawn 子进程明确接收 runs，保存策略指纹和控制轨迹。
- experiments/train_policy.py：固定特征及动作维度，拒绝旧轨迹，固定回放种子，移除范数冒充权重平均。
- agent/workflow.py、tools.py、run_agent.py：有序阶段执行、失败反思、真实实验门槛和上传门槛。

## 反思与风险

仅把 factors 设为 1 仍无法关闭原版 adapt 的 df+1 下限，因此通过 population 的 fixed_nu 接口锁定每代策略指令，而不修改共享 MPOPT 或 TFWA 对照。
原来的 prev_best 是烟花位置最优，cur_best 是全局最优，混用会凭空奖励；现在两者都使用 evaluator 的全局记录。
当前动作由线性 Q-learning 学习选择，倍率本身不是学习参数；两个烟花共享动作。离散控制器和稀疏改进奖励可能坍缩，必须检查实测分布，不能以权重非零证明有效。
spawn 不继承父进程设置的 _RUNS，必须通过任务参数显式传递 30 runs；workflow 检查每个结果文件、预算和种子，避免看见 exit=0 就宣布完成。

## 验证

单元测试检查原生 adapt 不覆盖 df、重启不改 df、上下界、奖励无伪改进、真实训练权重改变、冻结权重不变、旧策略拒绝、提前 finish 拒绝、阶段失败保持和断点恢复。
实际实验按 experiment.json 执行；结果写入 RESULTS.md，不预填成功。
