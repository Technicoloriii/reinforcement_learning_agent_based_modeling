#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
套餐1（6维）参数遍历脚本：一次性跑完 N 次后再输出汇总结果（不逐次打印/不逐次出图）。

- 默认模式：sweep（仅遍历并汇总分布；不做“拟合筛选”）
- 可选模式：calibrate（提供 --targets 后，按 targets 打分并选 top-K）

输出（out_dir 下）：
- candidates.csv              每次运行的参数 + summary 指标 + score（若有）
- quantiles_all.json          所有 runs 的各指标分位数（p05/p25/p50/p75/p95）
- topk.jsonl                  top-K 参数集（calibrate 模式）
- quantiles_topk.json         top-K runs 的各指标分位数（calibrate 模式）
- best_run.json               最佳（最低 score）那次的参数 + summary（calibrate 模式）

用法示例：
  # 仅遍历并看输出分布（推荐你现在先看这个）
  python calibrate_ensemble_pack1.py --n 100 --out outputs_pack1

  # 需要按 targets 筛选 top-K（以后你有 targets_S1.json 再用）
  python calibrate_ensemble_pack1.py --mode calibrate --targets params/targets_S1.json --n 300 --k 50 --out outputs_pack1_calib
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# 确保能 import 到同目录下的 summary_utils.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from summary_utils import summarize_module


# ----------------------------
# 默认 6 维 search space（你刚给的范围）
# ----------------------------
DEFAULT_SEARCH_SPACE = {
    "ALPHA_Q_UPD": [0.2, 0.3],
    "TAU_A_START": [0.25, 0.35],
    "TAU_A_END": [0.03, 0.07],
    "PATIENT_HABIT_STRENGTH": [0.2, 0.4],
    "PATIENT_INFO_ASYMMETRY": [0.05, 0.15],
    "WAIT_TOLERANCE_A0": [6.0, 10.0],
}

PARAM_KEYS_PACK1 = [
    "ALPHA_Q_UPD",
    "TAU_A_START",
    "TAU_A_END",
    "PATIENT_HABIT_STRENGTH",
    "PATIENT_INFO_ASYMMETRY",
    "WAIT_TOLERANCE_A0",
]


# ----------------------------
# 目标函数：可选 “区间惩罚” 距离（calibrate 模式）
# targets JSON 格式示例：
# {
#   "flow_share_county_mean": {"low": 0.40, "high": 0.50, "weight": 2.0},
#   "insurance_total_mean": {"low": 180000, "high": 190000, "weight": 1.0}
# }
# ----------------------------
def interval_penalty(summary: Dict[str, Any], targets: Dict[str, Any]) -> float:
    """
    区间内罚分 0；区间外按“距离最近边界 / 区间宽度”归一化后平方罚分。
    """
    total = 0.0
    used = 0
    for k, conf in targets.items():
        if k not in summary:
            continue
        v = summary.get(k, None)
        if v is None:
            continue
        low = conf.get("low", None)
        high = conf.get("high", None)
        if low is None or high is None:
            continue
        w = float(conf.get("weight", 1.0))
        width = float(high - low)
        if width <= 0:
            continue

        if v < low:
            d = (low - v) / width
        elif v > high:
            d = (v - high) / width
        else:
            d = 0.0
        total += w * (d * d)
        used += 1

    if used == 0:
        return float("inf")
    return total / used


# ----------------------------
# code patch：把 S1 脚本里的常量赋值替换成候选参数
# ----------------------------
def patch_assignments(code_text: str, params: Dict[str, float]) -> str:
    """
    将形如：PARAM = 0.25 的赋值行替换为新的数值。
    仅替换行首赋值（避免误伤字典 key、注释等）。
    """
    for key, val in params.items():
        # 数值格式：保持足够精度
        repl = f"{key} = {val:.8f}"
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[-+0-9.eE]+(\s*(#.*)?)$"
        if re.search(pattern, code_text) is None:
            # 没找到就跳过（不抛错，便于未来扩展）
            continue
        code_text = re.sub(pattern, repl + r"\1", code_text, count=1)
    return code_text


def inject_headless_and_seed(code_text: str, seed: int) -> str:
    """
    注入：
    - matplotlib headless（避免 plt.show() 阻塞）
    - 每次运行的随机种子（可复现）
    """
    header = f"""
# ---- injected by calibrate_ensemble_pack1.py ----
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")
import random as _random
import numpy as _np
_random.seed({seed})
_np.random.seed({seed})
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    _plt.show = lambda *args, **kwargs: None
except Exception:
    pass
# ---- end injection ----
"""
    return header + "\n" + code_text


def load_s1_code(s1_path: str) -> str:
    with open(s1_path, "r", encoding="utf-8") as f:
        return f.read()


def exec_module_from_text(module_name: str, code_text: str):
    """
    在独立 module namespace 中执行脚本，返回 module 对象（带 __dict__）。
    """
    import importlib.util
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(code_text, module.__dict__)
    return module


def uniform_sample(low: float, high: float) -> float:
    return low + (high - low) * random.random()


def sample_params(search_space: Dict[str, List[float]]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for k in PARAM_KEYS_PACK1:
        lo, hi = float(search_space[k][0]), float(search_space[k][1])
        params[k] = uniform_sample(lo, hi)
    return params


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if math.isfinite(float(x)):
                return float(x)
            return None
        return float(x)
    except Exception:
        return None


def compute_quantiles(rows: List[Dict[str, Any]], keys: List[str], ps=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, Any]:
    import numpy as np
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [safe_float(r.get(k, None)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[k] = {f"p{int(p*100):02d}": float(np.quantile(arr, p)) for p in ps}
        out[k]["n"] = int(arr.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "calibrate"], default="sweep",
                    help="sweep=只遍历并汇总；calibrate=按 targets 打分并选 top-K")
    ap.add_argument("--n", type=int, default=100, help="遍历次数（候选参数组数量）")
    ap.add_argument("--k", type=int, default=20, help="top-K（仅 calibrate 模式使用）")
    ap.add_argument("--eval-window", type=int, default=100, help="summary_utils 的最后窗口长度")
    ap.add_argument("--seed", type=int, default=12345, help="随机种子（每次运行会自动 seed+i）")
    ap.add_argument("--out", type=str, default="outputs_pack1", help="输出目录")
    ap.add_argument("--targets", type=str, default=None, help="targets JSON 路径（calibrate 模式需要）")
    ap.add_argument("--search-space", type=str, default=None, help="search_space.json 路径（可选）")
    ap.add_argument("--s1-path", type=str, default=None, help="S1 脚本路径（默认尝试 legacy/ 或当前目录）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    ensure_dir(out_dir)

    # resolve S1 path
    if args.s1_path is None:
        cand1 = os.path.join(THIS_DIR, "legacy", "S1_10_15_17_7_nol2.py")
        cand2 = os.path.join(THIS_DIR, "S1_10_15_17_7_nol2.py")
        if os.path.exists(cand1):
            s1_path = cand1
        elif os.path.exists(cand2):
            s1_path = cand2
        else:
            raise FileNotFoundError("找不到 S1 脚本。请用 --s1-path 指定。")
    else:
        s1_path = os.path.abspath(args.s1_path)
        if not os.path.exists(s1_path):
            raise FileNotFoundError(f"S1 脚本不存在: {s1_path}")

    # load search space
    if args.search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    else:
        with open(args.search_space, "r", encoding="utf-8") as f:
            search_space = json.load(f)

    # load targets if needed
    targets = None
    if args.mode == "calibrate":
        if args.targets is None:
            raise ValueError("calibrate 模式必须提供 --targets")
        with open(args.targets, "r", encoding="utf-8") as f:
            targets = json.load(f)

    s1_code_raw = load_s1_code(s1_path)

    all_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # 主循环：不逐次打印（只在最后汇总）
    for i in range(args.n):
        params = sample_params(search_space)
        run_seed = args.seed + i

        code = s1_code_raw
        code = patch_assignments(code, params)
        code = inject_headless_and_seed(code, run_seed)

        module_name = f"_s1_run_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{i}"
        try:
            mod = exec_module_from_text(module_name, code)
            summary = summarize_module(mod, eval_window=args.eval_window)
            row = {**params, **summary}
            if args.mode == "calibrate" and targets is not None:
                row["score"] = interval_penalty(summary, targets)
            else:
                row["score"] = None
            row["run_id"] = i
            all_rows.append(row)
        except Exception as e:
            err = {"run_id": i, **params, "error": repr(e)}
            errors.append(err)
            # 失败也记录一行，方便你排查“某些参数组合会炸”
            row = {**params, "run_id": i, "score": float("inf"), "error": repr(e)}
            all_rows.append(row)

    # 选取要做分位数汇总的 key：自动从 rows 里抓数值列（排除 params、run_id、error）
    metric_keys = []
    sample_row = all_rows[0] if all_rows else {}
    for k, v in sample_row.items():
        if k in PARAM_KEYS_PACK1 or k in ("run_id", "score", "error"):
            continue
        if safe_float(v) is not None:
            metric_keys.append(k)

    # 写 candidates.csv（包含所有 runs 的 summary + score；你不看也没关系）
    cand_csv = os.path.join(out_dir, "candidates.csv")
    fieldnames = ["run_id"] + PARAM_KEYS_PACK1 + sorted([k for k in sample_row.keys() if k not in PARAM_KEYS_PACK1 and k != "run_id"])
    # 去重保持顺序
    seen = set()
    ordered_fields = []
    for f in fieldnames:
        if f not in seen:
            ordered_fields.append(f); seen.add(f)
    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # quantiles across all
    q_all = compute_quantiles(all_rows, metric_keys)
    with open(os.path.join(out_dir, "quantiles_all.json"), "w", encoding="utf-8") as f:
        json.dump(q_all, f, ensure_ascii=False, indent=2)

    # calibrate: top-K
    if args.mode == "calibrate":
        valid = [r for r in all_rows if safe_float(r.get("score")) is not None and math.isfinite(float(r["score"])) and "error" not in r]
        valid.sort(key=lambda r: float(r["score"]))
        topk = valid[: max(1, min(args.k, len(valid)))] if valid else []

        # best run
        if topk:
            best = topk[0]
            with open(os.path.join(out_dir, "best_run.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

        # topk jsonl
        topk_path = os.path.join(out_dir, "topk.jsonl")
        with open(topk_path, "w", encoding="utf-8") as f:
            for r in topk:
                slim = {k: r.get(k) for k in (PARAM_KEYS_PACK1 + ["score"])}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")

        # topk quantiles
        q_topk = compute_quantiles(topk, metric_keys)
        with open(os.path.join(out_dir, "quantiles_topk.json"), "w", encoding="utf-8") as f:
            json.dump(q_topk, f, ensure_ascii=False, indent=2)

    # 最终只打印一个“总览”
    print("\n=== DONE ===")
    print(f"mode={args.mode}  n={args.n}  eval_window={args.eval_window}  out={out_dir}")
    if errors:
        print(f"runs with error: {len(errors)}/{args.n}  (详见 candidates.csv 的 error 列)")
    # 打印几个最常用指标的分位数（如果存在）
    for k in ["flow_share_county_mean", "flow_share_primary_mean", "insurance_total_mean",
              "insurance_county_share_mean", "severity_county_mean", "severity_primary_mean",
              "capacity_county_last", "capacity_township_last", "capacity_secondary_last",
              "capacity_county_share_last", "capacity_township_share_last"]:
        if k in q_all:
            q = q_all[k]
            print(f"{k}: p05={q['p05']:.4g}  p25={q['p25']:.4g}  p50={q['p50']:.4g}  p75={q['p75']:.4g}  p95={q['p95']:.4g}  (n={q['n']})")
    print("files written:")
    print(" - candidates.csv")
    print(" - quantiles_all.json")
    if args.mode == "calibrate":
        print(" - topk.jsonl")
        print(" - quantiles_topk.json")
        print(" - best_run.json")


if __name__ == "__main__":
    main()
