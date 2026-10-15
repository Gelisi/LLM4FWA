#!/usr/bin/env python3
"""从 reference_algorithms/experiment_results 提取 12 个最强方法的逐 run 成绩。

产出：
  baselines/samples.json         逐 run 原始误差（Wilcoxon 检验用）
                                 {bench: {algo: {"F{n}_D{d}": [err, ...]}}}
  baselines/baselines.csv        聚合表 (bench, algo, func, dim, mean, std, median, n_runs)
  baselines/best_per_problem.csv 每问题最优 mean error 的算法
  baselines/summary.md           可读摘要

并行读取（ceph-fuse 上 9 万文件，16 进程约 5-10 分钟）。
只读源目录，可重复执行。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REF_RESULTS = Path("/mnt/new_gemini/shipengcen/reference_algorithms/experiment_results")
OUT_DIR = Path(__file__).resolve().parent.parent / "baselines"
WORKERS = 16


def _scan_chunk(files):
    out = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("status") != "success" or d.get("error") is None:
            continue
        try:
            out.append((d["benchmark"], d["algorithm"], int(d["func_num"]),
                        int(d["dim"]), float(d["error"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def main():
    if not REF_RESULTS.exists():
        print(f"未找到基线目录: {REF_RESULTS}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(REF_RESULTS.glob("*/*/*/run_*.json"))
    print(f"发现 {len(files)} 个 run 文件，{WORKERS} 进程并行读取...")
    chunks = [files[i::WORKERS] for i in range(WORKERS)]
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, part in enumerate(ex.map(_scan_chunk, chunks)):
            records.extend(part)
            print(f"  chunk {i+1}/{WORKERS} 完成，累计 {len(records)} 条")

    # samples[bench][algo]["F{n}_D{d}"] = [errors]
    samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for bench, algo, fn, dim, err in records:
        samples[bench][algo][f"F{fn}_D{dim}"].append(err)

    # 落盘原始样本（统计检验用）
    with open(OUT_DIR / "samples.json", "w") as f:
        json.dump({b: {a: dict(probs) for a, probs in algos.items()}
                   for b, algos in samples.items()}, f)

    # 聚合表
    rows = []
    for bench, algos in samples.items():
        for algo, probs in algos.items():
            for key, errs in probs.items():
                fn, dim = key.split("_D")
                a = np.asarray(errs)
                rows.append({
                    "bench": bench, "algo": algo, "func": int(fn[1:]), "dim": int(dim),
                    "mean": float(a.mean()),
                    "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    "median": float(np.median(a)), "n_runs": len(a),
                })
    rows.sort(key=lambda r: (r["bench"], r["algo"], r["func"], r["dim"]))
    with open(OUT_DIR / "baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "algo", "func", "dim",
                                          "mean", "std", "median", "n_runs"])
        w.writeheader()
        w.writerows(rows)

    # 每问题最优
    best = {}
    for r in rows:
        key = (r["bench"], r["func"], r["dim"])
        if key not in best or r["mean"] < best[key]["mean"]:
            best[key] = r
    with open(OUT_DIR / "best_per_problem.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "algo", "func", "dim", "mean", "std", "n_runs"])
        w.writeheader()
        for key in sorted(best):
            r = best[key]
            w.writerow({k: r[k] for k in ("bench", "algo", "func", "dim", "mean", "std", "n_runs")})

    algo_counts = defaultdict(int)
    for r in rows:
        algo_counts[r["algo"]] += 1
    lines = [
        "# 基线快照（来源: reference_algorithms/experiment_results）",
        "",
        f"- 有效运行记录数: {len(records)}",
        f"- 覆盖算法: {len(algo_counts)} 个",
        f"- 覆盖问题数: {len(best)}",
        "",
        "| 算法 | 问题-运行记录数 |",
        "|------|----------------|",
    ]
    for a, c in sorted(algo_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {a} | {c} |")
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"提取完成: {len(records)} 条 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
