# iter_XXX: <一句话标题>

> 状态：PROPOSED / RUNNING / DONE / REJECTED / SUPERSEDED
> 日期：YYYY-MM-DD
> 前置：依赖哪个 iter 的结论（或无）

## 1. IDEA（假设）

- **动机/证据**：为什么提出这个假设？引用哪个观察、哪条诊断数据？
- **假设**：一句话，可证伪。
- **改动**：动哪个文件/机制（保持单一变量！）。
- **预期**：可证伪的预期，如"X 指标改善 ≥Y%，且其他指标不劣化"。

## 2. SETUP（实验设计）

- 训练/测试集划分（阶段 3 起）：
- 完整可复现命令（逐条）：
- 统计方法：Wilcoxon（α=0.05）+ 30 runs / 冒烟用 5 runs 须注明。

## 3. RESULT

- 关键数字表（或指向 results/ 下生成的 report.md / status.json）
- 负结果与意外现象同样要记录。

## 4. CONCLUSION

- 支持 / 证伪 / 无法判定；机制解释。
- 对总目标（各 benchmark Top-2）的贡献：排名变化。

## 5. NEXT

- 下一步建议（指向新的 iter 或回滚）。
