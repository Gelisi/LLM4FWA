#!/usr/bin/env python3
"""统计检验对比报告：本算法 vs 12 个最强基线，逐算法/逐问题。

比较方式（与 CEC 竞赛报告一致）：
  - 逐问题、逐对手：Wilcoxon 秩和检验（双侧，alpha=0.05）
      '+'  本算法显著更优
      '-'  本算法显著更差
      '='  无显著差异
  - 综合实力：每个问题上按 mean error 对（本算法+12 对手）排名，
    取所有问题的平均排名（Friedman 式），输出各 benchmark 的综合名次，
    并对照 Top-K 目标（默认 2）给出 achieved 判定。

产出（results/<algo>_<tag>/ 下）：
  report.md      人读报告（对手胜负汇总 + 逐问题明细 + 综合排名）
  pairwise.csv   逐问题×逐对手的检验明细
  status.json    agent 可读的迭代目标状态（top2 达成与否）

用法：
  python3 experiments/summarize.py --algo fwa_rl --tag real_tfwa_c22d10
  python3 experiments/summarize.py --algo fwa_rl --tag xxx --target-rank 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ranksums

ROOT = Path(__file__).resolve().parent.parent


def load_algo_results(algo: str, tag: str):
    d = ROOT / "results" / f"{algo}_{tag}"
    by_problem = defaultdict(list)
    for run_json in d.glob("*/*/run_*.json"):
        with open(run_json) as f:
            r = json.load(f)
        if r.get("status") != "success":
            continue
        by_problem[(r["benchmark"], r["func_num"], r["dim"])].append(float(r["error"]))
    return by_problem


def mark(ours, theirs, alpha: float) -> str:
    """'+': we better (p<alpha); '-': we worse; '=': no sig diff."""
    if len(ours) < 3 or len(theirs) < 3:
        return "="
    stat, p = ranksums(ours, theirs)
    if p >= alpha:
        return "="
    return "+" if np.mean(ours) < np.mean(theirs) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="fwa_rl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-rank", type=int, default=2,
                    help="综合实力目标名次（agent 迭代目标，默认 Top-2）")
    args = ap.parse_args()

    samples_path = ROOT / "baselines" / "samples.json"
    if not samples_path.exists():
        print("缺少 baselines/samples.json，请先运行: python3 experiments/sync_baselines.py")
        return 1
    with open(samples_path) as f:
        samples = json.load(f)

    ours = load_algo_results(args.algo, args.tag)
    if not ours:
        print(f"没有结果: results/{args.algo}_{args.tag}/")
        return 1

    all_opponents = sorted({a for b in samples.values() for a in b})

    # ---------------- 逐问题：检验 + 排名 ----------------
    pairwise_rows = []               # 逐问题×对手明细
    per_opp = {o: {"+": 0, "-": 0, "=": 0} for o in all_opponents}
    bench_ranks = defaultdict(list)  # bench -> [our_rank_per_problem]
    detail_lines = []

    for key in sorted(ours):
        bench, fn, dim = key
        my_errs = ours[key]
        my_mean = float(np.mean(my_errs))
        pkey = f"F{fn}_D{dim}"
        opp = samples.get(bench, {}).get(args.algo, {})
        opp = {a: probs.get(pkey, []) for a, probs in samples.get(bench, {}).items()
               if pkey in probs}
        if not opp:
            continue

        # 检验
        marks = {}
        for a, errs in opp.items():
            m = mark(my_errs, errs, args.alpha)
            marks[a] = m
            per_opp[a][m] += 1
            pairwise_rows.append({
                "bench": bench, "func": fn, "dim": dim, "opponent": a,
                "my_mean": my_mean, "opp_mean": float(np.mean(errs)),
                "mark": m,
                "p_value": float(ranksums(my_errs, errs)[1]) if len(errs) >= 3 else "",
            })

        # 综合排名（按 mean error；并列取最优名次，即 competition ranking）
        means = {a: float(np.mean(e)) for a, e in opp.items()}
        means[args.algo] = my_mean
        tol = 1e-12 * max(1.0, abs(my_mean))
        better = sum(1 for v in means.values() if v < my_mean - tol)
        n_tied = sum(1 for v in means.values() if abs(v - my_mean) <= tol)
        my_rank = better + 1  # 并列者同名次
        bench_ranks[bench].append(my_rank)

        w = sum(1 for m in marks.values() if m == "+")
        l = sum(1 for m in marks.values() if m == "-")
        t = len(marks) - w - l
        detail_lines.append(
            f"| {bench} {pkey} | {my_mean:.4e} | {len(my_errs)} | {my_rank} ({n_tied} 并列) "
            f"| {w}+ / {l}- / {t}= |"
        )

    # ---------------- 汇总 ----------------
    lines = [
        f"# {args.algo} ({args.tag}) 统计检验报告 (Wilcoxon rank-sum, alpha={args.alpha})",
        "",
        f"- 覆盖问题数: {len(detail_lines)}；对手: {len(per_opp)} 个",
        "",
        "## 1. 逐对手胜负汇总（全部问题累计）",
        "",
        "| 对手 | + (我们优) | = | - (我们劣) | 胜率 |",
        "|------|-----------|---|-----------|------|",
    ]
    for a in sorted(per_opp):
        c = per_opp[a]
        n = sum(c.values())
        lines.append(f"| {a} | {c['+']} | {c['=']} | {c['-']} | {c['+']/n:.0%} |")

    lines += ["", "## 2. 综合实力（Friedman 式平均排名，越小越强）", ""]
    lines += ["| Benchmark | 我们的平均排名 | 参与算法数 | 达成 Top-%d |" % args.target_rank,
              "|-----------|---------------|-----------|------------|"]
    status = {"algo": args.algo, "tag": args.tag, "target_rank": args.target_rank,
              "bench_ranks": {}, "achieved": {}}
    overall = []
    for bench, ranks in sorted(bench_ranks.items()):
        mr = float(np.mean(ranks))
        n_algos = len(all_opponents) + 1
        ok = mr <= args.target_rank
        status["bench_ranks"][bench] = round(mr, 3)
        status["achieved"][bench] = ok
        overall.extend(ranks)
        lines.append(f"| {bench} | {mr:.2f} | {n_algos} | {'YES' if ok else 'NO'} |")
    if overall:
        omr = float(np.mean(overall))
        status["overall_rank"] = round(omr, 3)
        status["achieved_overall"] = omr <= args.target_rank
        lines += ["", f"**全部 benchmark 平均排名: {omr:.2f} —— Top-{args.target_rank} "
                      f"{'已达成' if omr <= args.target_rank else '未达成'}**"]

    lines += ["", "## 3. 逐问题明细", "",
              "| 问题 | 本算法 mean | runs | 排名 | 与 12 对手 Wilcoxon |", "|------|------------|------|------|---------------------|"]
    lines += detail_lines

    out_dir = ROOT / "results" / f"{args.algo}_{args.tag}"
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    with open(out_dir / "pairwise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "func", "dim", "opponent",
                                          "my_mean", "opp_mean", "mark", "p_value"])
        w.writeheader()
        w.writerows(pairwise_rows)
    with open(out_dir / "status.json", "w") as f:
        json.dump(status, f, indent=1)

    print(f"报告: {out_dir/'report.md'}")
    print(f"状态: {out_dir/'status.json'}")
    if overall:
        print(f"平均排名 {status.get('overall_rank')} / Top-{args.target_rank}: "
              f"{'达成' if status['achieved_overall'] else '未达成'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
