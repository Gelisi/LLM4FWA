#!/usr/bin/env python3
"""统一评估器：算法 × benchmark × 函数 × 维度 × runs，多进程并行。

输出格式与 reference_algorithms/experiment_results 的 JSON 保持一致：
    results/<algo>_<tag>/<bench>/<F{n}_D{d}>/run_{k}.json
训练模式额外输出：results/<algo>_<tag>/traj/<bench>_F{n}_D{d}_run{k}.json

用法示例：
    python3 experiments/run_eval.py --config configs/smoke.json
    python3 experiments/run_eval.py --algo fwa_rl --bench cec2022 --dim 10 --runs 30 --train --tag train_c22d10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fwarl.bench import BENCH_SPECS, get_problem  # noqa: E402
from fwarl.tfwa import tfwa_solve  # noqa: E402
from fwarl.fwa_rl import tfwa_rl_solve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALGOS = ("tfwa", "fwa_rl")
_TAG = "exp"    # 由 main() 设置，worker 进程继承（fork）
_RUNS = 5       # 同上


def solve_one(algo: str, problem, seed: int, train: bool, policy_path: str | None) -> dict:
    if algo == "tfwa":
        return tfwa_solve(problem, seed=seed)
    elif algo == "fwa_rl":
        return tfwa_rl_solve(problem, seed=seed, train=train, policy_path=policy_path)
    else:
        raise ValueError(f"未知算法: {algo}")


def run_single(args_tuple) -> dict:
    (algo, bench, func_num, dim, run_id, seed0, max_evals_override,
     train, policy_path, record_history) = args_tuple
    problem = get_problem(bench, func_num, dim=dim)
    if max_evals_override:
        problem.max_evals = int(max_evals_override)
    seed = seed0 + run_id * 1000 + func_num

    t0 = time.time()
    try:
        res = solve_one(algo, problem, seed, train, policy_path)
        best_f = float(res["best_f"])
        err = best_f - problem.optimal
        # 防御：error < 0 说明底层 .so 状态被污染（最优值是下界），标记为可疑
        suspicious = err < -1e-9
        out = {
            "algorithm": algo,
            "benchmark": bench,
            "func_num": func_num,
            "dim": problem.dim,
            "max_evals": problem.max_evals,
            "seed": seed,
            "run_id": run_id,
            "status": "suspicious" if suspicious else "success",
            "best_f": best_f,
            "optimal": problem.optimal,
            "error": err,
            "evals_used": res["evals"],
            "time_seconds": round(time.time() - t0, 3),
        }
        if record_history:
            out["iteration_history"] = res["history"]
        if algo == "fwa_rl":
            out["action_counts"] = res.get("action_counts")
            out["weight_norms"] = res.get("weight_norms")
            out["policy_version"] = res["policy_version"]
            out["df_control"] = res["df_control"]
            out["df_trajectory"] = res["df_trajectory"]
            if train:
                out["_traj"] = res.get("trajectory")
    except Exception as e:  # noqa: BLE001
        import traceback
        out = {
            "algorithm": algo, "benchmark": bench, "func_num": func_num,
            "dim": problem.dim, "max_evals": problem.max_evals, "seed": seed,
            "run_id": run_id, "status": "failed", "error": str(e),
            "traceback": traceback.format_exc(),
        }
    return out, args_tuple


def save_run(out_dir: Path, bench: str, func_num: int, dim: int, run_id: int, result: dict,
             traj: list | None, train: bool):
    d = out_dir / bench / f"F{func_num}_D{dim}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"run_{run_id}.json", "w") as f:
        json.dump(result, f, indent=1)
    if train and traj is not None:
        td = out_dir / "traj"
        td.mkdir(parents=True, exist_ok=True)
        with open(td / f"{bench}_F{func_num}_D{dim}_run{run_id}.json", "w") as f:
            json.dump({
                "benchmark": bench, "func_num": func_num, "dim": dim,
                "run_id": run_id,
                "trajectory": traj,
                "weight_norms": result.get("weight_norms"),
                "action_counts": result.get("action_counts"),
                "policy_version": result.get("policy_version"),
            }, f)


def _worker(args_tuple):
    """一个 worker 进程跑完一个 (algo, bench, func, dim) 的全部 runs。

    任务粒度 = 一个问题的全部独立运行；配合 max_tasks_per_child=1，
    每个 (bench, func, dim) 都在全新子进程中执行，彻底隔离底层 .so 的
    跨函数状态污染（libcec*.so 在同进程内切换 func_num 存在非确定性 bug）。
    注意：worker 不写盘（ceph-fuse 多进程并发写会丢文件），只返回结果，
    由主进程统一落盘。
    """
    (algo, bench, func_num, dim, _zero, seed0, max_evals_override,
     train, policy_path, record_history, runs) = args_tuple
    results = []
    max_attempts = 3
    for run_id in range(1, runs + 1):
        task = (algo, bench, func_num, dim, run_id, seed0, max_evals_override,
                train, policy_path, record_history)
        out, _ = run_single(task)
        # 可疑结果（error<0，意味着底层 .so 数据加载失败导致函数退化为
        # 以 0 为中心的原始函数）在子进程内重试，最多 max_attempts 次
        attempt = 1
        while out.get("status") == "suspicious" and attempt < max_attempts:
            attempt += 1
            out, _ = run_single(task)
        traj = out.pop("_traj", None)
        results.append((algo, bench, func_num, dim, run_id, out, traj))
    return results


def main():
    p = argparse.ArgumentParser(description="FWA+RL 统一评估器")
    p.add_argument("--config", help="JSON 配置文件（其余参数可省略）")
    p.add_argument("--algo", default=["fwa_rl"], choices=ALGOS, nargs="+")
    p.add_argument("--bench", default=["cec2022"], nargs="+")
    p.add_argument("--dim", type=int, nargs="*", default=None)
    p.add_argument("--funcs", type=int, nargs="*", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--max-fes", type=int, default=None, help="覆盖 MaxFES（冒烟测试用）")
    p.add_argument("--workers", type=int, default=min(16, __import__("os").cpu_count()))
    p.add_argument("--train", action="store_true", help="RL 训练模式（在线学习+保存轨迹）")
    p.add_argument("--policy", default=None, help="加载的策略 JSON（fwa_rl）")
    p.add_argument("--tag", default="exp")
    p.add_argument("--no-history", action="store_true", help="不记录收敛历史（省空间）")
    p.add_argument("--config-overrides", default=None, help="内部使用")
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        with open(preliminary.config) as f:
            cfg = json.load(f)
        mapping = {"algos": "algo", "benchmarks": "bench", "dims": "dim",
                   "functions": "funcs", "max_evals_override": "max_fes"}
        defaults = {mapping.get(k, k): v for k, v in cfg.items()
                    if mapping.get(k, k) in vars(preliminary)}
        p.set_defaults(**defaults)
    args = p.parse_args()

    # 配置文件提供默认值，命令行参数优先
    global _TAG
    algo_list, bench_list, dims, funcs, runs = args.algo, args.bench, args.dim, args.funcs, args.runs
    max_evals_override, workers, tag = args.max_fes, args.workers, args.tag
    seed0, train, policy = args.seed0, args.train, args.policy
    if isinstance(algo_list, str):
        algo_list = [algo_list]
    if isinstance(bench_list, str):
        bench_list = [bench_list]
    policy_hash = None
    if "fwa_rl" in algo_list:
        if not train and not policy:
            p.error("冻结 RL 评测必须指定 --policy；训练请指定 --train")
        if policy:
            from fwarl.rl.learner import LinearQAgent
            from fwarl.rl.actions import POLICY_VERSION
            from fwarl.fwa_rl import N_FEATURES, ACTIONS
            import hashlib
            policy = str(Path(policy).resolve())
            agent = LinearQAgent.load(policy, expected_version=POLICY_VERSION)
            if (agent.n_features, agent.n_actions) != (N_FEATURES, len(ACTIONS)):
                p.error("策略维度不匹配")
            policy_hash = hashlib.sha256(Path(policy).read_bytes()).hexdigest()
    if runs < 1 or workers < 1:
        p.error("runs 和 workers 必须为正数")

    tasks = []
    global _RUNS
    _RUNS = runs
    _TAG = tag
    for algo in algo_list:
        for bench in bench_list:
            spec = BENCH_SPECS[bench]
            fns = funcs or list(range(1, spec["num_functions"] + 1))
            ds = dims or spec["dims"]
            for fn in fns:
                for d in (ds if bench != "cec2019" else [None]):
                    tasks.append((algo, bench, fn, d, 0, seed0, max_evals_override,
                                  train and algo == "fwa_rl", policy, not args.no_history, runs))
    total = len(tasks) * runs
    print(f"共 {total} 个 run / {len(tasks)} 个问题任务 (algo={algo_list}, bench={bench_list}, "
          f"dims={dims or 'spec默认'}, runs={runs})")

    cfg_meta = {"algos": algo_list, "benchmarks": bench_list, "dims": dims, "functions": funcs,
                "runs": runs, "max_evals_override": max_evals_override, "tag": tag,
                "train": train, "seed0": seed0, "policy": policy,
                "policy_sha256": policy_hash, "workers": workers,
                "df_control": "rl_replace" if "fwa_rl" in algo_list else "native"}
    for algo in algo_list:
        out_dir = ROOT / "results" / f"{algo}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "experiment_config.json", "w") as f:
            json.dump(cfg_meta, f, indent=1, default=str)

    n_ok = n_bad = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            done += 1
            for algo, bench, fn, d, rid, out, traj in fut.result():
                status = out.get("status")
                if status == "success":
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f"[{str(status).upper()}] {bench} F{fn} D{d} run{rid}: {out.get('error')}")
                out_dir = ROOT / "results" / f"{algo}_{tag}"
                save_run(out_dir, bench, fn, d, rid, out, traj, train)
            print(f"  完成 {done}/{len(tasks)} 个问题任务, ok={n_ok}, bad={n_bad}, elapsed={time.time()-t0:.0f}s")

    print(f"完成: ok={n_ok}, bad={n_bad}, 总耗时 {time.time()-t0:.0f}s")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
