# autoresearch 研究代理（agent/）

自主推进 `RESEARCH_PLAN.md` 研究循环的 LLM Agent：读状态 → 立项（IDEA.md）→ 改码 →
实验 → 统计汇总 → 回填 RESULTS.md → research_log 记录 → 下一个 iter。

## 两种运行模式

### 单任务模式（一次会话一个聚焦任务）

```bash
python3 agent/run_agent.py --max-turns 30 \
  --task "按 iter_001_rl_restart/CODE.md 实现 RL 重启控制，通过冒烟测试，回填自查清单"
```

### 目标驱动模式（总目标驱动，自主循环多段会话）⭐

```bash
python3 agent/run_agent.py \
  --goal "在全部 CEC benchmark 上把 TFWA+RL 迭代到综合实力 Top-2" \
  --max-sessions 15 --max-turns 40 --token-budget 3000000
```

每段会话自动：读状态 → 选价值最高的一步（立项/写码/编译/实验/汇总/记录）→ 推进 →
finish。段与段之间状态经 workspace（research_log/iterations/results）传递。
停止条件（任一）：所有 benchmark `status.json.achieved=true`（自动检测）、
`--max-sessions` 用尽、`--token-budget` 用尽。

> 建议：先 `--max-sessions 3 --token-budget 300000` 试跑半天，人工检查 iter 质量后再放开。

## Token 记账

每次 LLM 调用写一行到 `agent/token_usage.jsonl`：

```json
{"ts": "...", "mode": "goal", "session": 3, "task": "...", "turn": 5,
 "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
```

每会话结束追加 `{"type":"session_summary", ...}`。按 iter 归集：

```bash
grep iter_001 agent/token_usage.jsonl | python3 -c "
import json,sys
rows=[json.loads(l) for l in sys.stdin if '\"turn\"' in l]
print('calls:', len(rows), 'total_tokens:', sum(r['total_tokens'] for r in rows))"
```

（在 IDEA.md 的 task 描述里带上 iter 名，归集即自动关联。）

## 首次启用（3 步）

```bash
# 1) 配置 LLM
cp agent/.env.template agent/.env   # 已配置（n.codexapi.cc / gpt-6.1-sol）

# 2) 自检 API 连通性
python3 agent/check_api.py

# 3) 正式运行
python3 agent/run_agent.py --max-turns 20 --task "实现 iter_001 的代码并通过冒烟测试"
```

自测（无需真实调用）：
```bash
python3 agent/run_agent.py --dry-run   # 校验上下文注入 + 工具白名单/黑名单
python3 agent/run_agent.py --mock      # 脚本化 LLM，全链路走一遍
```

## 搜索工具（2026-10-15 实测）

| 工具 | 后端 | 状态 |
|------|------|------|
| `paper_search` | arXiv API + Semantic Scholar（429 自动退避） | ✅ 可靠，**查文献首选** |
| `github_search` | GitHub API（仓库/星标） | ✅ 可靠，找开源实现（查询词别超过 2-3 个） |
| `web_fetch` | 直连抓网页正文（拦截内网地址） | ✅ 可靠，抓具体 URL |
| `web_search` | Bing HTML | ⚠️ 勉强可用：本网络出口对搜索引擎有缓存干扰，结果相关性差，agent 被提示优先用前两个 |

系统提示已写明该优先级，agent 查文献/查实现会自动走可靠通道。

## 安全机制

- **bash 白名单**：`python3 experiments/…`、`python3 -c/-m`、`python3 fwarl/…`、只读命令
  （ls/cat/grep 等）、**编译工具链**（make/g++/gcc/cmake/mkdir，供 agent 写程序并编译）、
  `git status/diff/log/add/commit/init`（iter 版本管理）、模板复制
  （`cp -r iterations/_TEMPLATE …`）；
- **bash 黑名单**：`rm -rf /`、`sudo`、`git push/reset`、`pip/conda install`、`curl/wget`、
  `ssh/scp` 等；装依赖请人工执行（numpy/scipy/requests 已就绪）；
- **路径围栏**：文件读写限制在 autoresearch/ 工作区内；
- **超时**：单命令 1800s（编译大项目可调）；goal 模式有 max-sessions/token-budget 双上限；
- 全部会话留痕：agent/sessions/<时间戳>[ _sN].md。

## 日常使用建议

- 每次会话给一个聚焦任务（一个 iter 或一个诊断），不要"跑到 Top-2 为止"——
  那 should 由人在多个会话间把控节奏，每轮 `--max-turns 30~60`；
- 会话开场任务示例：
  - `"读 iter_001_rl_restart/IDEA.md 和 CODE.md，完成代码实现并通过 smoke"`
  - `"对 iter_001 执行正式训练与冻结评测，回填 RESULTS.md"`
  - `"分析 results/<tag>/pairwise.csv，我们被 rdex 在哪些函数上压制，写入 research_log"`
- agent 每轮开始会自动注入 research_log 尾部、各 iter 状态、最新 status.json。

## Key 调试结论（2026-10-14）

探测了本机全部 9 处 .env 凭证：
- 内网 iChat：所有项目的凭证都解析到同一 app `prc_app_multimodal`，
  网关返回 `HMAC auth disabled for app`（app 已停用）→ **全部失效**；
- 外部 API（aihubmix）：内网代理 403 拒绝该域名，直连 SSL 被拦截 → **不通**；
- `fwa_fm/.env` 的 new_*：端点 404（服务下线）。

→ 需要提供一组可用的 LLM 凭证（任何 OpenAI 兼容端点，或未停用的 iChat app），
填入 `agent/.env` 后 `python3 agent/check_api.py` 验证即可。
