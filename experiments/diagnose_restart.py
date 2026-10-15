#!/usr/bin/env python3
"""TFWA 重启机制诊断器：量化"无效重启 / 死预算"。

在给定问题上插桩运行原版 TFWA，记录：
  - 条件重启（improves×rest_iter < gap）与兜底重启（100 代无改进）的
    触发时刻、触发时 scale/span、与当前最优的差距
  - 死预算占比 = 1 - 末次全局最优改进位置 / MaxFES
  - 重启次数、重启方式分布

用途：验证"重启时机/落点不当导致搜索算力浪费"假设（iter_001 的证据来源）。

用法：
  python3 experiments/diagnose_restart.py --bench cec2022 --func 9 --dim 10 --seeds 7 14 21
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fwarl.paths import MPOPT_DIR, ensure_syspath  # noqa: E402

ensure_syspath()

from mpopt.algorithms.TFWA import TFWA  # noqa: E402
from fwarl.bench import get_problem  # noqa: E402
from fwarl.tfwa import SafeEvaluator  # noqa: E402
from mpopt.tools.objective import ObjFunction  # noqa: E402

EPS = 1e-8


class RestartProbe(TFWA):
    """原版 TFWA 主循环 + 重启事件插桩（逻辑与原版一致，仅记录）。"""

    def optimize(self, e):
        self.init(e)
        self.restart_events = []      # dict(evals, kind, scale_ratio, gap)
        self._last_best_eval = 0
        best_y = min(self.fits)
        n_cond = n_fallback = 0

        while not e.terminate():
            n_fit = np.array([0.0 for _ in range(self.fw_size)])
            for i in range(self.fw_size):
                self.pops[i].evolve(e)
                n_fit[i] = e(self.pops[i].dist.shift.reshape(1, self.dim))[0]

            improves = self.fits - n_fit
            rest_iter = (e.max_eval - e.num_eval) / (self.sp_size + self.fw_size)
            restart_ = (improves > EPS) * (improves * rest_iter < (n_fit - e.cur_y))

            for i in range(self.fw_size):
                if self.use_restart and restart_[i] == True:  # noqa: E712
                    self.restart_events.append({
                        "evals": e.num_eval, "kind": "conditional",
                        "scale_ratio": float(self.pops[i].dist.scale / np.mean(self.ub - self.lb)),
                        "gap": float(n_fit[i] - e.cur_y),
                    })
                    self.restart(i, e)
                    n_cond += 1
                    if getattr(self, "repair_control_state", False):
                        n_fit[i] = self.fits[i]
                if self.pops[i].no_improve_count >= 100:
                    self.restart_events.append({
                        "evals": e.num_eval, "kind": "fallback_keep_shift",
                        "scale_ratio": float(self.pops[i].dist.scale / np.mean(self.ub - self.lb)),
                        "gap": float(n_fit[i] - e.cur_y),
                    })
                    self.restart(i, e, shift=self.pops[i].dist.shift)
                    n_fallback += 1
                    if getattr(self, "repair_control_state", False):
                        n_fit[i] = self.fits[i]
            self.fits = n_fit
            if min(self.fits.min(), float(e.cur_y)) < best_y:
                best_y = min(self.fits.min(), float(e.cur_y))
                self._last_best_eval = e.num_eval

        self.final_best = best_y
        self.total_restarts = n_cond + n_fallback
        self.dead_budget = 1.0 - self._last_best_eval / e.max_eval
        return e.best_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="cec2022")
    ap.add_argument("--func", type=int, required=True)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 14, 21])
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    problem = get_problem(args.bench, args.func, dim=args.dim)
    obj = ObjFunction(problem.objective, dim=problem.dim, lb=problem.lb, ub=problem.ub)

    results = []
    print(f"诊断 {problem.name}, MaxFES={problem.max_evals}, seeds={args.seeds}")
    for seed in args.seeds:
        ev = SafeEvaluator(obj, max_eval=problem.max_evals)
        algo = RestartProbe()
        pa = algo.default_params()
        pa["seed"] = seed
        algo.set_params(pa)
        best = algo.optimize(ev)

        evs = [x["evals"] for x in algo.restart_events]
        kinds = [x["kind"] for x in algo.restart_events]
        rec = {
            "seed": seed,
            "error": float(best) - problem.optimal,
            "restarts_total": algo.total_restarts,
            "restarts_conditional": kinds.count("conditional"),
            "restarts_fallback": kinds.count("fallback_keep_shift"),
            "restart_evals_first10": evs[:10],
            "scale_ratio_first10": [round(x["scale_ratio"], 6) for x in algo.restart_events[:10]],
            "last_improve_evals": algo._last_best_eval,
            "dead_budget_ratio": round(algo.dead_budget, 4),
        }
        results.append(rec)
        print(f"  seed{seed}: err={rec['error']:.3e}  重启={rec['restarts_total']}"
              f"(条件{rec['restarts_conditional']}/兜底{rec['restarts_fallback']})"
              f"  末次改进@{rec['last_improve_evals']}  死预算={rec['dead_budget_ratio']:.1%}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"problem": problem.name, "runs": results}, f, indent=1)
        print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
