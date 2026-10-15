"""CEC benchmark 统一封装。

直接复用 reference_algorithms 的 cec_benchmark（C++ .so 加速），
把 (benchmark, 函数, 维度) 封装成统一 Problem：
    objective(x) -> float,  x in [lb, ub]^dim
评测协议与 reference_algorithms/experiment_results 保持一致，保证基线可比。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .paths import REF_DIR, ensure_syspath

# reference_algorithms 路径（默认与本仓库同级，可用 FWARL_REF_DIR 覆盖；
# .so 依赖其目录结构，整体拷贝即可）
ensure_syspath()

from cec_benchmark import (  # noqa: E402
    CEC2013,
    CEC2017,
    CEC2019,
    CEC2021,
    CEC2022,
)

# 与 reference_algorithms/experiment_results 的评测协议一致
BENCH_SPECS: dict[str, dict] = {
    "cec2013": dict(num_functions=28, dims=[10, 30, 50], eval_factor=10000, skip_functions=()),
    "cec2017": dict(num_functions=30, dims=[10, 30, 50, 100], eval_factor=10000, skip_functions=()),
    "cec2019": dict(num_functions=10, dims=[10], eval_factor=10000, skip_functions=()),
    "cec2021": dict(num_functions=10, dims=[10, 20], eval_factor=10000, skip_functions=()),
    "cec2022": dict(num_functions=12, dims=[10, 20], eval_factor=20000, skip_functions=()),
}

CEC2019_FUNC_DIMS = {1: 9, 2: 16, 3: 18}
CEC2019_FUNC_BOUNDS = {1: (-8192.0, 8192.0), 2: (-16384.0, 16384.0), 3: (-4.0, 4.0)}


@dataclass
class Problem:
    name: str                 # 例如 "cec2013_F1_D10"
    bench_name: str
    func_num: int
    dim: int
    lb: np.ndarray
    ub: np.ndarray
    optimal: float
    max_evals: int
    objective: Callable[[np.ndarray], float]


@lru_cache(maxsize=None)
def _get_bench_class(bench_name: str):
    return {
        "cec2013": CEC2013,
        "cec2017": CEC2017,
        "cec2019": CEC2019,
        "cec2021": CEC2021,
        "cec2022": CEC2022,
    }[bench_name]


def get_problem(bench_name: str, func_num: int, dim: Optional[int] = None,
                eval_factor: Optional[int] = None) -> Problem:
    """构造一个 Problem 实例。

    CEC2019 的维度/边界由函数编号决定，忽略 dim 参数。
    """
    spec = BENCH_SPECS[bench_name]
    if bench_name == "cec2019":
        bench = _get_bench_class(bench_name)(func_num=func_num)
        dim = bench.FUNC_DIMS[func_num]
        lo, hi = bench.FUNC_BOUNDS[func_num]
        lb, ub = np.full(dim, lo), np.full(dim, hi)
        optimal = bench.OPTIMAL_VALUES[func_num]
        max_evals = (eval_factor or spec["eval_factor"]) * dim
    else:
        if dim is None:
            dim = spec["dims"][0]
        bench = _get_bench_class(bench_name)(dim=dim)
        lo, hi = bench.SEARCH_RANGE
        lb, ub = np.full(dim, lo), np.full(dim, hi)
        optimal = bench.get_optimal_value(func_num)
        factor = eval_factor if eval_factor is not None else spec["eval_factor"]
        max_evals = factor * dim

    def objective(x: np.ndarray) -> float:
        return float(bench.evaluate(func_num, np.asarray(x, dtype=float)))

    return Problem(
        name=f"{bench_name}_F{func_num}_D{dim}",
        bench_name=bench_name,
        func_num=func_num,
        dim=dim,
        lb=lb,
        ub=ub,
        optimal=float(optimal),
        max_evals=int(max_evals),
        objective=objective,
    )


def iter_problems(bench_names, dims=None, func_nums=None):
    """枚举 (bench, func, dim) 组合，跳过各 benchmark 声明不稳定的函数。"""
    for bn in bench_names:
        spec = BENCH_SPECS[bn]
        fns = func_nums or [f for f in range(1, spec["num_functions"] + 1)
                            if f not in spec["skip_functions"]]
        for fn in fns:
            if bn == "cec2019":
                yield get_problem(bn, fn)
            else:
                for d in (dims or spec["dims"]):
                    yield get_problem(bn, fn, dim=d)
