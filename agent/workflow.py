"""带实验产物门槛和断点续跑的 TFWA df 研究工作流。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ("idea", "implement", "smoke", "train", "freeze", "collect", "upload", "completed")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchWorkflow:
    def __init__(self, root: Path = ROOT, timeout: int = 1800):
        self.root = root.resolve()
        self.timeout = timeout
        self.path = None
        self.state = None

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root) or ".codebuddy" in p.parts:
            raise ValueError("工作流路径必须位于仓库内，且不能修改 .codebuddy")
        return p

    def code_digest(self) -> str:
        h = hashlib.sha256()
        for folder in ("fwarl", "experiments", "agent", "tests"):
            for p in sorted((self.root / folder).rglob("*.py")):
                h.update(str(p.relative_to(self.root)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def _save(self):
        self.state["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.path)

    def _event(self, kind, **data):
        self.state["events"].append({"kind": kind, "time": time.time(), **data})
        self._save()

    def start(self, iteration: str) -> str:
        d = self.resolve(iteration)
        if d.parent != self.root / "iterations" or not re.fullmatch(r"iter_\d+_[\w-]+", d.name):
            raise ValueError("iteration 应为 iterations/iter_XXX_<名称>")
        if self.state and self.state["stage"] != "completed" and self.path.parent != d:
            raise ValueError("请先完成当前迭代，不能通过新立项跳过实验")
        self.path = d / "workflow.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            return self.status()
        cfg = json.loads((d / "experiment.json").read_text())
        for section in ("train", "evaluate"):
            spec = cfg[section]
            if not spec.get("benchmarks") or not spec.get("functions") or not spec.get("dims"):
                raise ValueError("实验配置必须显式指定 benchmarks/functions/dims")
            if int(spec.get("runs", 0)) < (30 if section == "evaluate" else 1):
                raise ValueError("正式冻结评测至少 30 runs；训练至少 1 run")
        prior = cfg.get("prior_artifacts", [])
        if not prior:
            raise ValueError("新 idea 必须引用之前的 idea 和实验结果")
        sources = {p: digest(self.resolve(p)) for p in prior}
        self.state = {"iteration": str(d.relative_to(self.root)), "stage": "idea",
                      "run_id": uuid.uuid4().hex[:12], "started_at": time.time(),
                      "config": cfg, "prior_artifacts": sources, "events": [],
                      "initial_code": self.code_digest()}
        self._save()
        return self.status()

    def status(self) -> str:
        if self.state is None:
            return "未启动 workflow。先准备 IDEA.md、CODE.md、experiment.json，再 workflow_start。"
        return json.dumps({"iteration": self.state["iteration"], "stage": self.state["stage"],
                           "run_id": self.state["run_id"],
                           "prior_artifacts": list(self.state["prior_artifacts"]),
                           "last_event": self.state["events"][-1] if self.state["events"] else None},
                          ensure_ascii=False, indent=2)

    def _require(self, stage):
        if self.state is None or self.state["stage"] != stage:
            raise ValueError(f"当前阶段不允许 {stage}：{self.status()}")

    def _next(self):
        self.state["stage"] = STAGES[STAGES.index(self.state["stage"]) + 1]
        self._save()
        return self.status()

    def review(self, evidence: str, reflection: str) -> str:
        if not self.state or self.state["stage"] not in STAGES:
            raise ValueError("先启动 workflow 才能反思并修订代码")
        if len(reflection.strip()) < 30:
            raise ValueError("反思需说明历史证据、修改理由、风险和验证方式，不能仅写完成")
        stage = self.state["stage"]
        expected = "IDEA.md" if stage == "idea" else "CODE.md"
        p = self.resolve(evidence)
        if p != self.path.parent / expected or len(p.read_text().strip()) < 100:
            raise ValueError(f"需要本迭代完整的 {expected}")
        if stage == "idea":
            text = p.read_text()
            if "SETUP" not in text or "假设" not in text:
                raise ValueError("IDEA 必须包含可证伪假设和 SETUP")
        else:
            self.state["reviewed_code"] = self.code_digest()
            self.state["stage"] = "implement"
        self._event("review", stage=stage, evidence=evidence,
                    sha256=digest(p), reflection=reflection)
        return self._next()

    def _command(self, label, argv):
        logdir = self.path.parent / "artifacts" / self.state["run_id"]
        logdir.mkdir(parents=True, exist_ok=True)
        logfile = logdir / f"{label}_{len(self.state['events'])}.log"
        start = time.time()
        with logfile.open("w") as f:
            try:
                r = subprocess.run(argv, cwd=self.root, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=self.timeout, text=True)
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = 124
        self._event("command", label=label, argv=argv, returncode=code,
                    elapsed_s=round(time.time() - start, 3),
                    log=str(logfile.relative_to(self.root)), code_sha256=self.code_digest())
        if code:
            raise RuntimeError(f"{label} 失败 exit={code}，保留阶段供反思/重试。日志：{logfile}\n"
                               + logfile.read_text(errors="replace")[-1800:])

    def _eval(self, phase):
        cfg = dict(self.state["config"]["evaluate" if phase == "freeze" else "train"])
        cfg.update(algos=["tfwa", "fwa_rl"] if phase == "freeze" else ["fwa_rl"],
                   train=phase != "freeze", workers=min(16, int(cfg.get("workers", 4))))
        if phase == "smoke":
            cfg.update(runs=1, max_evals_override=2000)
            cfg["functions"] = cfg["functions"][:1]
            cfg["dims"] = cfg["dims"][:1]
        tag = f"{self.path.parent.name}_{self.state['run_id']}_{phase}_{len(self.state['events'])}"
        cfg["tag"] = tag
        cfg["policy"] = self.state.get("policy") if phase == "freeze" else None
        cp = self.path.parent / "artifacts" / f"{tag}.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cfg, indent=2))
        self._command(phase, [sys.executable, "experiments/run_eval.py", "--config", str(cp)])
        self._verify_runs(cfg)
        self.state[phase] = cfg
        self._event("verified_experiment", phase=phase, tag=tag)

    def _verify_runs(self, cfg):
        from fwarl.bench import BENCH_SPECS, CEC2019_FUNC_DIMS
        for algo in cfg["algos"]:
            for bench in cfg["benchmarks"]:
                for fn in cfg["functions"]:
                    dims = [CEC2019_FUNC_DIMS.get(fn, 10)] if bench == "cec2019" else cfg["dims"]
                    for dim in dims:
                        for rid in range(1, cfg["runs"] + 1):
                            p = self.root / "results" / f"{algo}_{cfg['tag']}" / bench / f"F{fn}_D{dim}" / f"run_{rid}.json"
                            r = json.loads(p.read_text())
                            budget = cfg.get("max_evals_override") or BENCH_SPECS[bench]["eval_factor"] * dim
                            if (r.get("status") != "success" or r["evals_used"] < budget
                                    or r["max_evals"] != budget or r["seed"] != cfg.get("seed0", 42) + rid * 1000 + fn):
                                raise ValueError(f"实验失败/预算不足/种子不匹配：{p}")
                            if algo == "fwa_rl":
                                trace = r.get("df_trajectory", [])
                                if r.get("df_control") != "rl_replace" or not trace or any(t["commanded"] != t["after"] for t in trace):
                                    raise ValueError(f"自由度控制证据缺失：{p}")

    def _collect(self):
        import numpy as np
        from scipy.stats import wilcoxon
        cfg = self.state["freeze"]
        self._verify_runs(cfg)
        groups = {}
        for algo in cfg["algos"]:
            folder = self.root / "results" / f"{algo}_{cfg['tag']}"
            for p in folder.glob("*/*/run_*.json"):
                r = json.loads(p.read_text())
                key = f"{r['benchmark']}/F{r['func_num']}_D{r['dim']}"
                groups.setdefault(key, {}).setdefault(algo, {})[r["seed"]] = r["error"]
        rows = []
        for key, algos in sorted(groups.items()):
            seeds = sorted(algos["tfwa"])
            if seeds != sorted(algos["fwa_rl"]) or len(seeds) < 30:
                raise ValueError("对照与 RL 的配对种子/样本数不匹配")
            native = np.array([algos["tfwa"][s] for s in seeds])
            rl = np.array([algos["fwa_rl"][s] for s in seeds])
            p = 1.0 if np.all(rl == native) else float(wilcoxon(rl, native).pvalue)
            rows.append({"problem": key, "runs": len(seeds),
                         "tfwa_mean": float(native.mean()), "rl_mean": float(rl.mean()),
                         "tfwa_std": float(native.std(ddof=1)), "rl_std": float(rl.std(ddof=1)),
                         "p_value": p, "mark": "=" if p >= 0.05 else ("+" if rl.mean() < native.mean() else "-")})
        summary = {"iteration": self.state["iteration"], "tag": cfg["tag"],
                   "test": "paired Wilcoxon, uncorrected per problem, alpha=0.05",
                   "policy_sha256": self.state["policy_sha256"], "rows": rows,
                   "scope": "仅当前配置覆盖问题；不代表全 benchmark 达标"}
        sp = self.path.parent / "summary.json"
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        report = ["## Workflow 实测汇总", "", summary["scope"],
                  "", "| 问题 | runs | TFWA mean error | RL mean error | p | +/=/- |",
                  "|---|---:|---:|---:|---:|---|"]
        for r in rows:
            report.append(f"| {r['problem']} | {r['runs']} | {r['tfwa_mean']:.6g} | {r['rl_mean']:.6g} | {r['p_value']:.4g} | {r['mark']} |")
        report += ["", "检验为逐问题未校正配对 Wilcoxon，不据此宣称全局优越。",
                   f"策略：`{self.state['policy']}`；完整命令、退出码、耗时见 `workflow.json`。", ""]
        rp = self.path.parent / "RESULTS.md"
        old = rp.read_text() if rp.exists() else ""
        rp.write_text(old.split("## Workflow 实测汇总")[0] + "\n" + "\n".join(report))
        log = self.root / "research_log.md"
        line = f"\n- **{self.path.parent.name}** 实验已完成（待上传）：{len(rows)} 个问题 × 30 runs，tag={cfg['tag']}；负结果同样保留，详见 {self.state['iteration']}/summary.json。\n"
        with log.open("a") as f:
            f.write(line)
        self.state["summary_sha256"] = digest(sp)
        self._event("collected", problems=len(rows), summary=str(sp.relative_to(self.root)))

    def run(self, stage: str) -> str:
        self._require(stage)
        if stage not in STAGES[2:-1]:
            raise ValueError("idea/implement 请使用 workflow_review")
        if self.code_digest() != self.state["reviewed_code"]:
            raise ValueError("代码在审核后改变，先 workflow_review 反思修订并重新实验")
        if stage == "smoke":
            self._command("unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            self._eval(stage)
        elif stage == "train":
            self._eval(stage)
            tag = self.state["train"]["tag"]
            policy = self.path.parent / "artifacts" / f"policy_{self.state['run_id']}_{len(self.state['events'])}.json"
            self._command("replay", [sys.executable, "experiments/train_policy.py", "--glob",
                                     str(self.root / "results" / f"fwa_rl_{tag}" / "traj" / "*.json"),
                                     "--out", str(policy), "--seed", "42"])
            from fwarl.rl.learner import LinearQAgent
            from fwarl.rl.actions import POLICY_VERSION
            agent = LinearQAgent.load(policy, expected_version=POLICY_VERSION)
            if not agent._step or not any(agent.weight_norms()):
                raise ValueError("训练没有产生有效策略权重")
            self.state.update(policy=str(policy.relative_to(self.root)), policy_sha256=digest(policy))
        elif stage == "freeze":
            policy = self.resolve(self.state["policy"])
            if digest(policy) != self.state["policy_sha256"]:
                raise ValueError("训练策略被修改，需要重新训练")
            self._eval(stage)
            if digest(policy) != self.state["policy_sha256"]:
                raise ValueError("冻结评测修改了策略权重")
        elif stage == "collect":
            self._collect()
            self._command("summarize", [sys.executable, "experiments/summarize.py", "--algo", "fwa_rl",
                                        "--tag", self.state["freeze"]["tag"]])
        elif stage == "upload":
            sp = self.path.parent / "summary.json"
            if digest(sp) != self.state["summary_sha256"]:
                raise ValueError("汇总结果被修改，请核查后重新收集")
            self._command("upload", [sys.executable, "agent/sync_github.py", "--message",
                                     f"{self.path.parent.name}: RL df control experiments and workflow"])
        return self._next()

    def finish_error(self, after: float | None = None) -> str | None:
        if not self.state or self.state["stage"] != "completed":
            return "禁止 finish：必须实际完成 smoke→train→freeze→collect→upload。" + self.status()
        if self.code_digest() != self.state["reviewed_code"]:
            return "禁止 finish：实验后代码改变，旧结果不能作为完成证据。"
        if after is not None and not any(e["kind"] == "verified_experiment" and e.get("phase") == "freeze"
                                         and e["time"] >= after for e in self.state["events"]):
            return "禁止 finish：本会话没有新冻结评测；请新建迭代或反思后重跑实验。"
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", required=True)
    p.add_argument("--stage", choices=STAGES[:-1] + ("status",), default="status")
    p.add_argument("--evidence")
    p.add_argument("--reflection")
    args = p.parse_args()
    w = ResearchWorkflow()
    w.start(args.iteration)
    try:
        if args.stage in ("idea", "implement"):
            print(w.review(args.evidence or "", args.reflection or ""))
        elif args.stage == "status":
            print(w.status())
        else:
            print(w.run(args.stage))
        return 0
    except (ValueError, RuntimeError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
