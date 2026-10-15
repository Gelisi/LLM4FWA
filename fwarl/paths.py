"""中心化路径配置——全项目唯一的绝对路径来源。

布局约定（相对路径，随仓库迁移）：
    <parent>/LLM4FWA|autoresearch/   <- 本仓库
    <parent>/reference_algorithms/   <- CEC benchmark .so 与基线数据（同级目录）
    <parent>/MPOPT/                  <- TFWA 原实现（同级目录）

服务器部署时若无同级目录，用环境变量覆盖（无需改代码）：
    export FWARL_REF_DIR=/path/to/reference_algorithms
    export FWARL_MPOPT_DIR=/path/to/MPOPT
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录


def _resolve(env_key: str, default: Path) -> Path:
    v = os.environ.get(env_key)
    return (Path(v).expanduser().resolve() if v else default).resolve()


REF_DIR = _resolve("FWARL_REF_DIR", PROJECT_ROOT.parent / "reference_algorithms")
REF_RESULTS = _resolve("FWARL_REF_RESULTS", REF_DIR / "experiment_results")
MPOPT_DIR = _resolve("FWARL_MPOPT_DIR", PROJECT_ROOT.parent / "MPOPT")


def ensure_syspath():
    """把 REF_DIR / MPOPT_DIR 加入 sys.path（cec_benchmark 与 mpopt 的 import 需要）。"""
    import sys
    for p in (str(REF_DIR), str(MPOPT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
