#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_ensemble_pack1_scored.py

在原 calibrate_ensemble_pack1.py 基础上，增加“基于 targets_S1.json 的参数评价 + 选 top-K”。

你现在最常用的是两种场景：

A) 你已经有上一轮 sweep 的 candidates.csv，不想重跑 S1：
   python calibrate_ensemble_pack1_scored.py --mode select --candidates candidates.csv --targets targets_S1.json --k 20 --out outputs_s1_top20

B) 你想边跑 S1 边打分筛选 top-K（未来跑更大 N 时用）：
   python calibrate_ensemble_pack1_scored.py --mode calibrate --n 100 --k 20 --targets targets_S1.json --out outputs_s1_calib

输出（out_dir 下）：
- scored_candidates.csv        candidates + score + 命中情况（pass_all 等）
- topk.csv                     top-K runs（含参数+summary+score）
- topk_params.json             top-K 参数（便于后续跑 S2/S3）
- quantiles_topk.json          top-K 的分位数（p05/p25/p50/p75/p95）
- best_run.json                score 最小的那次（参数+summary+score）

可选：
- --plot 生成 targets 指标的 UI 区间图（median/IQR/90%UI）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import contextlib
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# 确保能 import 到同目录下的 summary_utils.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from summary_utils import summarize_module


# ----------------------------
# 默认 6 维 search space（套餐1）
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
# 工具函数
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            if math.isfinite(v):
                return v
            return None
        v = float(str(x))
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def uniform_sample(low: float, high: float) -> float:
    return low + (high - low) * random.random()


def sample_params(search_space: Dict[str, List[float]]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for k in PARAM_KEYS_PACK1:
        lo, hi = float(search_space[k][0]), float(search_space[k][1])
        params[k] = uniform_sample(lo, hi)
    return params


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_default_targets_path() -> Optional[str]:
    # 你说 targets_S1.json 已放在根目录；这里做一个默认查找
    cand = [
        os.path.join(THIS_DIR, "targets_S1.json"),
        os.path.join(THIS_DIR, "params", "targets_S1.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def resolve_default_s1_path() -> str:
    cand1 = os.path.join(THIS_DIR, "legacy", "S1_10_15_17_7_nol2.py")
    cand2 = os.path.join(THIS_DIR, "S1_10_15_17_7_nol2.py")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError("找不到 S1 脚本。请用 --s1-path 指定。")


def load_s1_code(s1_path: str) -> str:
    with open(s1_path, "r", encoding="utf-8") as f:
        return f.read()


def exec_module_from_text(module_name: str, code_text: str):
    import importlib.util
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(code_text, module.__dict__)
    return module


# ----------------------------
# patch：替换 S1 脚本里的常量赋值为采样参数
# ----------------------------
def patch_assignments(code_text: str, params: Dict[str, float]) -> str:
    """
    将形如：PARAM = 0.25 的赋值行替换为新的数值。
    仅替换行首赋值（避免误伤 dict key、注释等）。
    """
    for key, val in params.items():
        repl = f"{key} = {val:.8f}"
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[-+0-9.eE]+(\s*(#.*)?)$"
        if re.search(pattern, code_text) is None:
            continue
        code_text = re.sub(pattern, repl + r"\1", code_text, count=1)
    return code_text


def inject_headless_and_seed(code_text: str, seed: int) -> str:
    header = f"""
# ---- injected by calibrate_ensemble_pack1_scored.py ----
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


# ----------------------------
# 打分：区间惩罚（合理区间）
# ----------------------------
def score_interval_penalty(summary: Dict[str, Any], targets: Dict[str, Any]) -> Tuple[float, int, int]:
    """
    你的 targets 是“合理区间”：
    - 如果 summary 值落在 [low, high]：该指标罚分 = 0
    - 如果落在区间外：罚分 = ((distance to nearest bound) / width)^2 * weight
      其中 weight 可选（targets 里不写则默认 1）

    返回：
      score, n_used, n_in
    """
    total = 0.0
    used = 0
    n_in = 0
    for k, conf in targets.items():
        if k not in summary:
            continue
        v = safe_float(summary.get(k))
        if v is None:
            continue

        low = conf.get("low", None)
        high = conf.get("high", None)
        if low is None or high is None:
            continue
        low = float(low); high = float(high)
        width = high - low
        if width <= 0:
            continue

        w = float(conf.get("weight", 1.0))

        if low <= v <= high:
            d = 0.0
            n_in += 1
        elif v < low:
            d = (low - v) / width
        else:
            d = (v - high) / width

        total += w * (d * d)
        used += 1

    if used == 0:
        return float("inf"), 0, 0

    return total / used, used, n_in


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


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def plot_ui_intervals_topk(out_png: str, quantiles: Dict[str, Any], title: str):
    """
    画一个横向 UI 区间图：median(dot)、IQR(thick)、90%UI(thin)
    只画 quantiles 里可用的指标。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 优先画你 targets 的那些 key（更像“校准结果展示”）
    keys = list(quantiles.keys())
    # 过滤掉非 quantile dict 项
    keys = [k for k in keys if isinstance(quantiles.get(k), dict) and "p50" in quantiles[k]]

    if not keys:
        return

    labels = keys  # 先用 key；你后续如果想映射成中文名我可以再帮你美化

    fig = plt.figure(figsize=(10, max(3.5, 0.55 * len(keys) + 1.2)))
    ax = fig.add_subplot(111)

    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys[::-1]):
        q = quantiles[k]
        p05, p25, p50, p75, p95 = q["p05"], q["p25"], q["p50"], q["p75"], q["p95"]
        ax.plot([p05, p95], [yi, yi], linewidth=2)
        ax.plot([p25, p75], [yi, yi], linewidth=6)
        ax.plot(p50, yi, marker="o", markersize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_and_score_s1(args, targets: Dict[str, Any], search_space: Dict[str, Any], s1_path: str) -> List[Dict[str, Any]]:
    s1_code_raw = load_s1_code(s1_path)

    all_rows: List[Dict[str, Any]] = []

    for i in range(args.n):
        params = sample_params(search_space)
        run_seed = args.seed + i

        code = patch_assignments(s1_code_raw, params)
        code = inject_headless_and_seed(code, run_seed)

        module_name = f"_s1_run_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{i}"

        try:
            if args.verbose:
                mod = exec_module_from_text(module_name, code)
            else:
                _buf_out = io.StringIO()
                _buf_err = io.StringIO()
                with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
                    mod = exec_module_from_text(module_name, code)

            summary = summarize_module(mod, eval_window=args.eval_window)
            score, used, n_in = score_interval_penalty(summary, targets)

            row = {**params, **summary}
            row["score"] = score
            row["n_targets_used"] = used
            row["n_targets_in"] = n_in
            row["pass_all"] = (used > 0 and n_in == used)
            row["run_id"] = i
            all_rows.append(row)

        except Exception as e:
            row = {**params, "run_id": i, "score": float("inf"), "error": repr(e),
                   "n_targets_used": 0, "n_targets_in": 0, "pass_all": False}
            all_rows.append(row)

    return all_rows


def score_existing_candidates(args, targets: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 保留原始行（字符串）+ 追加 score
            # 同时把 params/metrics 里可转数值的字段保持原样（写回 CSV 时仍是 str/float 都可）
            summary_like = dict(r)
            # score 需要数值
            score, used, n_in = score_interval_penalty(summary_like, targets)
            r["score"] = score
            r["n_targets_used"] = used
            r["n_targets_in"] = n_in
            r["pass_all"] = (used > 0 and n_in == used)
            rows.append(r)
    return rows


def select_topk(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    # 过滤掉 error / score=inf 的
    valid = []
    for r in rows:
        if "error" in r and r["error"]:
            continue
        s = safe_float(r.get("score"))
        if s is None or not math.isfinite(float(s)):
            continue
        valid.append(r)
    valid.sort(key=lambda r: float(r["score"]))
    return valid[: max(1, min(k, len(valid)))] if valid else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "calibrate", "select"], default="select",
                    help="select=只对已有 candidates.csv 打分选 top-K；calibrate=边跑S1边打分；sweep=只跑不打分")
    ap.add_argument("--n", type=int, default=100, help="运行次数（仅 sweep / calibrate）")
    ap.add_argument("--k", type=int, default=20, help="top-K")
    ap.add_argument("--eval-window", type=int, default=100, help="summary_utils 的最后窗口长度")
    ap.add_argument("--seed", type=int, default=12345, help="随机种子（每次运行 seed+i）")
    ap.add_argument("--out", type=str, default="outputs_s1_scored", help="输出目录")
    ap.add_argument("--targets", type=str, default=None, help="targets JSON 路径（合理区间）")
    ap.add_argument("--search-space", type=str, default=None, help="search_space.json 路径（可选）")
    ap.add_argument("--s1-path", type=str, default=None, help="S1 脚本路径（calibrate / sweep 用）")
    ap.add_argument("--candidates", type=str, default=None, help="已有 candidates.csv 路径（select 模式必填）")
    ap.add_argument("--verbose", action="store_true", help="显示每次运行脚本的输出（默认静默）")
    ap.add_argument("--plot", action="store_true", help="生成 top-K 的 UI 区间图（PNG）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    ensure_dir(out_dir)

    # targets
    targets_path = args.targets
    if targets_path is None:
        targets_path = resolve_default_targets_path()
        if targets_path is None:
            raise FileNotFoundError("未提供 --targets，且未在根目录找到 targets_S1.json/params/targets_S1.json")
    targets = load_json(targets_path)

    # search space
    if args.search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    else:
        search_space = load_json(args.search_space)

    # Mode handling
    rows: List[Dict[str, Any]] = []

    if args.mode == "select":
        if args.candidates is None:
            raise ValueError("select 模式必须提供 --candidates candidates.csv")
        rows = score_existing_candidates(args, targets)

    elif args.mode == "calibrate":
        s1_path = resolve_default_s1_path() if args.s1_path is None else os.path.abspath(args.s1_path)
        rows = run_and_score_s1(args, targets, search_space, s1_path)

    elif args.mode == "sweep":
        # 只跑不打分：仍然保存 candidates.csv + quantiles_all.json，便于你后续 select
        s1_path = resolve_default_s1_path() if args.s1_path is None else os.path.abspath(args.s1_path)
        s1_code_raw = load_s1_code(s1_path)

        for i in range(args.n):
            params = sample_params(search_space)
            run_seed = args.seed + i
            code = inject_headless_and_seed(patch_assignments(s1_code_raw, params), run_seed)
            module_name = f"_s1_run_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{i}"
            try:
                if args.verbose:
                    mod = exec_module_from_text(module_name, code)
                else:
                    _buf_out = io.StringIO()
                    _buf_err = io.StringIO()
                    with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
                        mod = exec_module_from_text(module_name, code)
                summary = summarize_module(mod, eval_window=args.eval_window)
                row = {**params, **summary, "run_id": i}
                rows.append(row)
            except Exception as e:
                rows.append({**params, "run_id": i, "error": repr(e)})

    # 统一写出 scored_candidates / candidates
    # 决定 fieldnames：从所有 rows 合并 key
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # 把 run_id + 参数放前面
    preferred = ["run_id"] + PARAM_KEYS_PACK1 + ["score", "n_targets_used", "n_targets_in", "pass_all"]
    rest = [k for k in sorted(all_keys) if k not in preferred]
    fieldnames = [k for k in preferred if k in all_keys] + rest

    if args.mode == "sweep":
        cand_path = os.path.join(out_dir, "candidates.csv")
    else:
        cand_path = os.path.join(out_dir, "scored_candidates.csv")
    write_csv(cand_path, rows, fieldnames)

    # 做分位数汇总（all rows / topk rows）
    # 自动抓数值列（排除 params/run_id/error）
    metric_keys = []
    for k in fieldnames:
        if k in PARAM_KEYS_PACK1 or k in ("run_id", "score", "n_targets_used", "n_targets_in", "pass_all", "error"):
            continue
        # 看看至少有一行能转成 float
        if any(safe_float(r.get(k)) is not None for r in rows):
            metric_keys.append(k)

    if args.mode == "sweep":
        q_all = compute_quantiles(rows, metric_keys)
        with open(os.path.join(out_dir, "quantiles_all.json"), "w", encoding="utf-8") as f:
            json.dump(q_all, f, ensure_ascii=False, indent=2)
    else:
        topk = select_topk(rows, args.k)
        # topk.csv
        topk_path = os.path.join(out_dir, "topk.csv")
        write_csv(topk_path, topk, fieldnames)

        # best_run.json
        if topk:
            best = topk[0]
            with open(os.path.join(out_dir, "best_run.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

        # topk_params.json（只保留参数 + score）
        topk_params = []
        for r in topk:
            topk_params.append({k: safe_float(r.get(k)) for k in PARAM_KEYS_PACK1} | {"score": safe_float(r.get("score"))})
        with open(os.path.join(out_dir, "topk_params.json"), "w", encoding="utf-8") as f:
            json.dump(topk_params, f, ensure_ascii=False, indent=2)

        # quantiles_topk.json
        q_topk = compute_quantiles(topk, metric_keys)
        with open(os.path.join(out_dir, "quantiles_topk.json"), "w", encoding="utf-8") as f:
            json.dump(q_topk, f, ensure_ascii=False, indent=2)

        # 也顺便输出 targets 对应指标的 quantiles（便于写校准图）
        # 如果 targets 的 key 不在 metric_keys 里，也没关系
        q_targets = compute_quantiles(topk, list(targets.keys()))
        with open(os.path.join(out_dir, "quantiles_topk_targets.json"), "w", encoding="utf-8") as f:
            json.dump(q_targets, f, ensure_ascii=False, indent=2)

        # 可选：画一张图
        if args.plot:
            out_png = os.path.join(out_dir, "ui_topk_targets.png")
            plot_ui_intervals_topk(
                out_png,
                q_targets,
                title=f"S1 (FFS) — Top-{len(topk)} good-fitting parameter sets\nMedian(dot), IQR(thick), 90% UI(thin)"
            )

    # 控制台只给总览
    print("\n=== DONE ===")
    print(f"mode={args.mode}  out={out_dir}")
    print(f"targets={targets_path}")
    if args.mode in ("calibrate", "sweep"):
        print(f"n={args.n}  eval_window={args.eval_window}")
    if args.mode != "sweep":
        print(f"k={args.k}")
        print("outputs: scored_candidates.csv, topk.csv, topk_params.json, quantiles_topk.json, best_run.json")
        if args.plot:
            print("         ui_topk_targets.png")
    else:
        print("outputs: candidates.csv, quantiles_all.json")


if __name__ == "__main__":
    main()
