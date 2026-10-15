"""Agent 工具集：bash（白名单+黑名单+超时）、文件读写、web_search / paper_search / web_fetch、finish。

文件操作限制在 workspace_root 内；bash 在 workspace_root 下执行。
网络工具直连（本机直连外网可用，勿走内网代理）。
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree

from . import net

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录（随仓库迁移）

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) autoresearch-agent/1.0"}
_PRIVATE_RE = re.compile(
    r"^(https?://)(localhost|127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|0\.)",
    re.I)

TOOLS_SPEC = [
    {"name": "bash", "desc": "在工作区执行 shell 命令（白名单：python3 experiments/…、"
                             "python3 -c、make/g++ 等编译、ls/cat/grep 等只读命令）。",
     "args": {"command": "str"}},
    {"name": "read_file", "desc": "读取工作区内文件（可指定 offset/limit 行数）。",
     "args": {"path": "str", "offset": "int?", "limit": "int?"}},
    {"name": "write_file", "desc": "写入/覆盖工作区内文件（父目录自动创建）。",
     "args": {"path": "str", "content": "str"}},
    {"name": "edit_file", "desc": "精确替换文件片段（old_str 唯一匹配）。",
     "args": {"path": "str", "old_str": "str", "new_str": "str"}},
    {"name": "web_search", "desc": "通用网页搜索（Bing，尽力而为；本网络出口对搜索引擎有缓存干扰，"
                                   "结果可能不相关——查文献请优先 paper_search，查代码请优先 github_search）。",
     "args": {"query": "str", "max_results": "int?"}},
    {"name": "paper_search", "desc": "学术文献搜索（arXiv + Semantic Scholar：标题/年份/摘要/链接/引用数）。"
                                     "查论文与相关工作的首选。",
     "args": {"query": "str", "max_results": "int?"}},
    {"name": "github_search", "desc": "GitHub 仓库搜索（名称/描述/星标数），找算法开源实现、benchmark 代码。",
     "args": {"query": "str", "max_results": "int?"}},
    {"name": "web_fetch", "desc": "抓取网页正文并转纯文本（最多约 8000 字符），配合 web_search 用。",
     "args": {"url": "str"}},
    {"name": "gpu_status", "desc": "GPU 状态快照（利用率/显存/进程）+ 空闲告警。"
                                   "跑任何 GPU 任务前先调用；发现空闲卡应安排任务"
                                   "（CUDA_VISIBLE_DEVICES=<idx> 前缀）。",
     "args": {}},
    {"name": "workflow_start", "desc": "启动/恢复研究迭代，读取 experiment.json 和历史证据。",
     "args": {"iteration": "str"}},
    {"name": "workflow_review", "desc": "审核 IDEA 或代码；记录反思，实验失败后可反复修订重跑。",
     "args": {"evidence": "str", "reflection": "str"}},
    {"name": "workflow_run", "desc": "执行当前阶段 smoke/train/freeze/collect/upload；真实产物通过才能前进。",
     "args": {"stage": "str"}},
    {"name": "workflow_status", "desc": "查看当前阶段、执行记录和失败日志。", "args": {}},
    {"name": "finish", "desc": "只有本会话真实实验、汇总及上传完成后才能结束。", "args": {"summary": "str"}},
]


class ToolBox:
    def __init__(self, config: dict):
        self.cfg = config
        self.allow = tuple(config.get("bash_allow_prefixes", []))
        self.deny = tuple(config.get("bash_deny_patterns", []))
        self.timeout = config.get("command_timeout_s", 900)
        self.max_out = config.get("tool_output_max_chars", 6000)
        from .workflow import ResearchWorkflow
        self.workflow = ResearchWorkflow(ROOT, self.timeout)

    # ------------------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = (ROOT / rel).resolve()
        if not p.is_relative_to(ROOT) or ".codebuddy" in p.parts:
            raise ValueError(f"路径越界或受保护（必须在 {ROOT} 内且不能访问 .codebuddy）: {rel}")
        return p

    def run(self, name: str, args: dict) -> str:
        if name == "workflow_start":
            return self.workflow.start(args["iteration"])
        if name == "workflow_status":
            return self.workflow.status()
        if name == "workflow_review":
            return self.workflow.review(args["evidence"], args["reflection"])
        if name == "workflow_run":
            return self.workflow.run(args["stage"])
        if name == "bash":
            return self._bash(args["command"])
        if name == "read_file":
            p = self._resolve(args["path"])
            text = p.read_text().splitlines()
            off = int(args.get("offset", 0) or 0)
            lim = int(args.get("limit", 250) or 250)
            seg = text[off:off + lim]
            return "\n".join(f"{off+i+1}: {l}" for i, l in enumerate(seg)) or "(空)"
        if name == "write_file":
            p = self._resolve(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"已写入 {args['path']}（{len(args['content'])} 字符）"
        if name == "edit_file":
            p = self._resolve(args["path"])
            text = p.read_text()
            old = args["old_str"]
            if text.count(old) != 1:
                return f"错误: old_str 匹配 {text.count(old)} 次（需恰好 1 次）"
            p.write_text(text.replace(old, args["new_str"]))
            return f"已编辑 {args['path']}"
        if name == "web_search":
            return self._web_search(args["query"], int(args.get("max_results", 8) or 8))
        if name == "paper_search":
            return self._paper_search(args["query"], int(args.get("max_results", 5) or 5))
        if name == "github_search":
            return self._github_search(args["query"], int(args.get("max_results", 5) or 5))
        if name == "web_fetch":
            return self._web_fetch(args["url"])
        if name == "gpu_status":
            from . import gpu
            return gpu.status_text() + "\n" + gpu.idle_report()
        raise ValueError(f"未知工具: {name}")

    # ---------------- 网络工具 ----------------
    @staticmethod
    def _web_search(query: str, max_results: int = 8) -> str:
        """Bing HTML 优先，DuckDuckGo 兜底。直连（勿用代理）。"""
        errs = []
        for backend in (ToolBox._search_bing, ToolBox._search_ddg):
            try:
                items = backend(query, max_results)
                if items:
                    header = f"[{backend.__name__}·{len(items)} 条结果] {query}\n"
                    return header + "\n".join(items)
                errs.append(f"{backend.__name__}: 0 条")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{backend.__name__}: {e}")
        return "搜索失败（" + "; ".join(errs) + "）"

    @staticmethod
    def _strip_tags(s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s)
        return html.unescape(re.sub(r"\s+", " ", s)).strip()

    @staticmethod
    def _search_bing(query: str, max_results: int) -> list:
        r = net.get("https://www.bing.com/search", params={"q": query, "count": max_results},
                         headers=_UA, timeout=20)
        r.raise_for_status()
        items = []
        for m in re.finditer(r'<li class="b_algo".*?<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>'
                             r'(?:.*?<p[^>]*>(.*?)</p>)?', r.text, re.DOTALL):
            url = m.group(1)
            # Bing 重定向包装: ...u=a1<base64真实url>&ntb=1
            q = re.search(r"[?&]u=a1([A-Za-z0-9_-]+)", url)
            if q:
                from base64 import urlsafe_b64decode
                b64 = q.group(1)
                b64 += "=" * (-len(b64) % 4)
                try:
                    url = urlsafe_b64decode(b64).decode("utf-8", "replace")
                except Exception:
                    pass
            if not url.startswith("http"):
                continue
            title = ToolBox._strip_tags(m.group(2))
            snip = ToolBox._strip_tags(m.group(3) or "")
            items.append(f"- {title}\n  {url}\n  {snip[:220]}")
            if len(items) >= max_results:
                break
        return items

    @staticmethod
    def _search_ddg(query: str, max_results: int) -> list:
        r = net.post("https://html.duckduckgo.com/html/",
                          data={"q": query}, timeout=20)
        r.raise_for_status()
        items = []
        for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                             r'(?:.*?class="result__snippet"[^>]*>(.*?)</a>)?',
                             r.text, re.DOTALL):
            url = m.group(1)
            if url.startswith("//"):
                url = "https:" + url
            q = re.search(r"uddg=([^&]+)", url)
            if q:
                from urllib.parse import unquote
                url = unquote(q.group(1))
            title = ToolBox._strip_tags(m.group(2))
            snip = ToolBox._strip_tags(m.group(3) or "")
            items.append(f"- {title}\n  {url}\n  {snip[:220]}")
            if len(items) >= max_results:
                break
        return items

    @staticmethod
    def _paper_search(query: str, max_results: int = 5) -> str:
        lines = []
        # arXiv
        try:
            r = net.get("http://export.arxiv.org/api/query",
                             params={"search_query": f"all:{query}",
                                     "max_results": max_results, "sortBy": "relevance"},
                             headers=_UA, timeout=20)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for e in ElementTree.fromstring(r.text).findall("a:entry", ns):
                title = re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip()
                link = e.findtext("a:id", "", ns)
                year = e.findtext("a:published", "", ns)[:4]
                abstract = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()
                lines.append(f"[arXiv {year}] {title}\n  {link}\n  {abstract[:260]}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"(arXiv 失败: {e})")
        # Semantic Scholar（429 退避重试一次）
        for attempt in range(2):
            try:
                r = net.get("https://api.semanticscholar.org/graph/v1/paper/search",
                                 params={"query": query, "limit": max_results,
                                         "fields": "title,year,abstract,url,citationCount"},
                                 headers=_UA, timeout=20)
                if r.status_code == 429:
                    time.sleep(3)
                    continue
                for p in r.json().get("data", []):
                    ab = re.sub(r"\s+", " ", p.get("abstract") or "")
                    lines.append(f"[S2 {p.get('year')} 引{p.get('citationCount')}] "
                                 f"{p.get('title')}\n  {p.get('url')}\n  {ab[:260]}")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    lines.append(f"(Semantic Scholar 失败: {e})")
        return "\n".join(lines) if lines else "无结果"

    @staticmethod
    def _github_search(query: str, max_results: int = 5) -> str:
        r = net.get("https://api.github.com/search/repositories",
                         params={"q": query, "per_page": max_results, "sort": "stars"},
                         headers=_UA, timeout=20)
        r.raise_for_status()
        lines = []
        for it in r.json().get("items", []):
            desc = re.sub(r"\s+", " ", it.get("description") or "")
            lines.append(f"- {it['full_name']} (★{it.get('stargazers_count', 0)})\n"
                         f"  {it.get('html_url')}\n  {desc[:220]}")
        return "\n".join(lines) if lines else "无结果"

    @staticmethod
    def _web_fetch(url: str, max_chars: int = 8000) -> str:
        if _PRIVATE_RE.match(url.strip()):
            return "拒绝访问内网地址"
        r = net.get(url, timeout=25)
        r.raise_for_status()
        text = r.text
        text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.I)
        text = ToolBox._strip_tags(text)
        if len(text) > max_chars:
            text = text[:max_chars] + f"...[截断，原始 {len(text)} 字符]"
        return text or "(空页面)"

    def _bash(self, command: str) -> str:
        cmd = command.strip()
        if any(pat in cmd for pat in self.deny):
            return f"拒绝执行（命中黑名单）: {cmd[:200]}"
        if not cmd.startswith(self.allow):
            return (f"拒绝执行（不在白名单）。允许前缀: {list(self.allow)}\n"
                    f"提示：实验请用 python3 experiments/...；快速分析用 python3 -c '...'")
        try:
            r = subprocess.run(cmd, shell=True, cwd=str(ROOT), timeout=self.timeout,
                               capture_output=True, text=True, errors="replace")
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr[-1500:]) if r.stderr else "")
            out = out.strip() or "(无输出)"
            if len(out) > self.max_out:
                out = out[:self.max_out] + f"\n...[截断，共 {len(out)} 字符]"
            return f"exit={r.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return f"超时（>{self.timeout}s）: {cmd[:200]}"

    @staticmethod
    def spec_text() -> str:
        return "\n".join(
            f"- {t['name']}({', '.join(t['args'])}): {t['desc']}" for t in TOOLS_SPEC)

    @staticmethod
    def validate_args(name: str, args: dict) -> str | None:
        spec = next((t for t in TOOLS_SPEC if t["name"] == name), None)
        if spec is None:
            return f"未知工具 {name}，可用: {[t['name'] for t in TOOLS_SPEC]}"
        for a, typ in spec["args"].items():
            if not typ.startswith("int?") and not typ.endswith("?") and a not in args:
                return f"工具 {name} 缺少参数: {a}"
        return None
