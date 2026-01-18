#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_s2_ui_band.py

目的：
- 使用 S1 校准筛选出的 plausible 参数集合（topk_params.json）
- 在 S2 场景下做“参数不确定性传播”（不再校准，不再打分）
- 生成一个与原来风格接近的 share 轨迹图，并叠加不确定性区间（UI band）

输出（默认）：
- outputs_uncertainty_s2/
    - s2_flow_share_ui_band.png
    - s2_flow_share_quantiles.csv   (每期的分位数，便于后续复用/做S3/做Δ分布)
    - s2_flow_share_runs.npz        (可选：缓存每次run的MA序列，方便重复画图)

运行方式（PyCharm 直接 Run 即可）：
- 默认不需要任何参数：会读取
    - topk_params.json
    - S2_10_15_17_7_nol2.py
- 如果你的文件名不同，在“脚本形参”里传参即可：
    --params params/topk_params.json --s2 legacy/S2_xxx.py

说明：
- 我们会对 S2 脚本做“最小侵入式”的 patch：
    1) 把 ALPHA_Q_UPD 等参数赋值行替换为本次参数集数值
    2) 在进入 S2 脚本的“Enhanced Plotting”段之前提前退出（避免弹窗/大量图/CSV）
- 因为我们只需要 flow_share_lvl 的时间序列做 UI band。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Patch helpers
# -------------------------

def patch_assignments(code_text: str, params: Dict[str, float]) -> str:
    """
    将形如：PARAM = 0.25 的赋值行替换为新的数值。
    仅替换行首赋值（避免误伤字典 key、注释等）。
    """
    for key, val in params.items():
        if key == "score":
            continue
        # 以 repr(val) 保留合理精度（避免 0.3000000004 之类）
        repl = f"{key} = {repr(float(val))}"
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[-+0-9.eE]+(\s*(#.*)?)$"
        if re.search(pattern, code_text) is None:
            # 没找到就跳过（便于未来扩展/不同版本脚本）
            continue
        code_text = re.sub(pattern, repl + r"\1", code_text, count=1)
    return code_text


def inject_ensemble_early_exit(code_text: str) -> str:
    """
    在 S2 脚本进入大段绘图/导出前提前退出，避免：
    - 大量 plt.figure / plt.show
    - export_metrics_to_csv / heatmap / CSV 写盘
    只要数据收集（flow_share_lvl 等）已经完成即可。

    我们使用一个“marker”定位：S2 脚本中应存在：
        # =============== Enhanced Plotting ===============

    在 marker 后插入：
        if globals().get("__ENSEMBLE__", False): raise SystemExit
    """
    marker = "# =============== Enhanced Plotting ==============="
    if marker not in code_text:
        raise RuntimeError(
            "在 S2 脚本中找不到 marker: '# =============== Enhanced Plotting ==============='\n"
            "说明你的 S2 脚本版本和我们当前假设不一致。你可以把 S2 脚本开头/结尾发我，我来适配。"
        )
    inject = marker + "\n" + "if globals().get('__ENSEMBLE__', False):\n    raise SystemExit\n"
    return code_text.replace(marker, inject, 1)


def exec_s2_with_params(s2_path: Path, params: Dict[str, float], ensemble_mode: bool = True) -> Dict:
    """
    执行一次 S2，并返回其 globals dict（包含 flow_share_lvl 等）。
    """
    code = s2_path.read_text(encoding="utf-8", errors="ignore")
    code = patch_assignments(code, params)
    if ensemble_mode:
        code = inject_ensemble_early_exit(code)

    # headless: 避免 GUI backend（保险起见）
    os.environ.setdefault("MPLBACKEND", "Agg")

    g: Dict = {
        "__name__": "__main__",
        "__file__": str(s2_path),
        "__ENSEMBLE__": bool(ensemble_mode),
    }

    try:
        exec(compile(code, str(s2_path), "exec"), g, g)
    except SystemExit:
        # 这是我们主动提前退出
        pass
    return g


# -------------------------
# Core computation
# -------------------------

def as_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def safe_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    简化版 moving average（same），并对 NaN 做简单填充。
    目的：与原脚本效果足够接近（并不追求一模一样）。
    """
    if x.size == 0:
        return x
    x2 = x.astype(float, copy=True)
    if np.isnan(x2).any():
        # forward fill
        last = None
        for i in range(len(x2)):
            if np.isfinite(x2[i]):
                last = x2[i]
            elif last is not None:
                x2[i] = last
        # backward fill
        nxt = None
        for i in range(len(x2) - 1, -1, -1):
            if np.isfinite(x2[i]):
                nxt = x2[i]
            elif nxt is not None:
                x2[i] = nxt
        x2 = np.nan_to_num(x2, nan=0.0)

    w = int(max(1, w))
    kernel = np.ones(w, dtype=float) / w
    out = np.convolve(x2, kernel, mode="same")

    # edge correction (partial window)
    half = w // 2
    for i in range(min(half, len(x2))):
        out[i] = float(np.mean(x2[: i * 2 + 1]))
    for i in range(max(0, len(x2) - half), len(x2)):
        start = max(0, i - half)
        out[i] = float(np.mean(x2[start : i + 1]))
    return out


def quantiles_over_runs(mat: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> np.ndarray:
    """
    mat: shape (n_runs, T)
    return: shape (len(qs), T)
    """
    return np.quantile(mat, qs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="topk_params.json", help="S1 筛选出的参数集合（list of dict）")
    ap.add_argument("--s2", type=str, default="S2_10_15_17_7_nol2.py", help="S2 场景脚本路径")
    ap.add_argument("--outdir", type=str, default="outputs_uncertainty_s2", help="输出目录")
    ap.add_argument("--movavg", type=int, default=None, help="移动平均窗口（默认读 S2 脚本里的 REPORT_MOVAVG，如无则用 24）")
    ap.add_argument("--max_n", type=int, default=None, help="只跑前 N 组参数（调试用）")
    ap.add_argument("--cache_npz", action="store_true", help="缓存每次run的 MA 序列到 npz，便于重复画图")
    ap.add_argument("--no_plot_show", action="store_true", help="不弹出窗口（只保存 png）")
    args = ap.parse_args()

    root = Path.cwd()
    params_path = (root / args.params).resolve()
    s2_path = (root / args.s2).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not params_path.exists():
        raise FileNotFoundError(f"找不到参数文件: {params_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"找不到 S2 脚本: {s2_path}")

    param_sets: List[Dict] = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(param_sets, list) or not param_sets:
        raise ValueError("params 文件格式应为 list[dict]，且不能为空。")

    if args.max_n is not None:
        param_sets = param_sets[: int(args.max_n)]

    n = len(param_sets)
    print(f"[S2 ensemble] parameter sets = {n}")

    # 先跑一次获取 T 与颜色/默认 movavg
    t0 = time.time()
    g0 = exec_s2_with_params(s2_path, param_sets[0], ensemble_mode=True)

    flow0 = g0.get("flow_share_lvl", None)
    if not isinstance(flow0, dict) or "county" not in flow0:
        raise RuntimeError("S2 脚本执行后没有找到 flow_share_lvl['county']，请检查脚本版本。")

    T = int(g0.get("T", len(flow0["county"])))
    county_color = None
    primary_color = None
    colors = g0.get("COLORS", {})
    if isinstance(colors, dict):
        county_color = colors.get("county", None)
        # primary 用 township/secondary 的色系（脚本里 township 和 secondary 都是同一个红）
        primary_color = colors.get("township", colors.get("secondary", None))
    county_color = county_color or "#2C458D"
    primary_color = primary_color or "#EF5950"

    movavg = args.movavg
    if movavg is None:
        movavg = int(g0.get("REPORT_MOVAVG", 24))
    print(f"[S2 ensemble] T={T}, movavg={movavg}")

    # 收集所有 run 的 raw 与 MA
    county_raw_runs = np.zeros((n, T), dtype=float)
    primary_raw_runs = np.zeros((n, T), dtype=float)
    county_ma_runs = np.zeros((n, T), dtype=float)
    primary_ma_runs = np.zeros((n, T), dtype=float)

    def extract_shares(g: Dict) -> Tuple[np.ndarray, np.ndarray]:
        flow = g["flow_share_lvl"]
        county = as_float_array(flow["county"])
        sec = as_float_array(flow.get("secondary", np.zeros_like(county)))
        town = as_float_array(flow.get("township", np.zeros_like(county)))
        # primary = secondary + township（与原图“towns + large towns”一致）
        primary = sec + town
        # 对齐长度
        if county.size < T:
            county = np.pad(county, (0, T - county.size), constant_values=np.nan)
        if primary.size < T:
            primary = np.pad(primary, (0, T - primary.size), constant_values=np.nan)
        return county[:T], primary[:T]

    # 第一组（已经跑过）
    county0, primary0 = extract_shares(g0)
    county_raw_runs[0, :] = county0
    primary_raw_runs[0, :] = primary0
    county_ma_runs[0, :] = safe_moving_average(county0, movavg)
    primary_ma_runs[0, :] = safe_moving_average(primary0, movavg)

    # 其余参数集
    for i in range(1, n):
        if i % 5 == 0 or i == n - 1:
            print(f"  running {i+1}/{n} ...")
        gi = exec_s2_with_params(s2_path, param_sets[i], ensemble_mode=True)
        c, p = extract_shares(gi)
        county_raw_runs[i, :] = c
        primary_raw_runs[i, :] = p
        county_ma_runs[i, :] = safe_moving_average(c, movavg)
        primary_ma_runs[i, :] = safe_moving_average(p, movavg)

    print(f"[S2 ensemble] done in {time.time()-t0:.1f}s")

    # 计算分位数（raw + MA）
    X = np.arange(T)
    qs = (0.05, 0.25, 0.50, 0.75, 0.95)

    q_county_raw = quantiles_over_runs(county_raw_runs, qs)
    q_primary_raw = quantiles_over_runs(primary_raw_runs, qs)
    q_county_ma = quantiles_over_runs(county_ma_runs, qs)
    q_primary_ma = quantiles_over_runs(primary_ma_runs, qs)

    # 输出 quantiles CSV（后续做 S3 / Δ 分布会用到）
    def qdf(name: str, qmat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"t": X})
        for qi, q in enumerate(qs):
            df[f"{name}_q{int(q*100):02d}"] = qmat[qi, :]
        return df

    df_out = pd.concat(
        [
            qdf("county_raw", q_county_raw),
            qdf("primary_raw", q_primary_raw).drop(columns=["t"]),
            qdf("county_ma", q_county_ma).drop(columns=["t"]),
            qdf("primary_ma", q_primary_ma).drop(columns=["t"]),
        ],
        axis=1,
    )
    csv_path = outdir / "s2_flow_share_quantiles.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if args.cache_npz:
        np.savez_compressed(
            outdir / "s2_flow_share_runs.npz",
            county_ma=county_ma_runs,
            primary_ma=primary_ma_runs,
            county_raw=county_raw_runs,
            primary_raw=primary_raw_runs,
            t=X,
        )

    # -------------------------
    # Plot (style close to your original)
    # -------------------------
    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(111)

    # UI bands (基于 MA)
    q05, q25, q50, q75, q95 = 0, 1, 2, 3, 4
    ax.fill_between(X, q_county_ma[q05], q_county_ma[q95], color=county_color, alpha=0.10, linewidth=0)
    ax.fill_between(X, q_county_ma[q25], q_county_ma[q75], color=county_color, alpha=0.18, linewidth=0)
    ax.fill_between(X, q_primary_ma[q05], q_primary_ma[q95], color=primary_color, alpha=0.10, linewidth=0)
    ax.fill_between(X, q_primary_ma[q25], q_primary_ma[q75], color=primary_color, alpha=0.18, linewidth=0)

    # median raw (dashed) + median MA (solid) — keep legend similar to your original
    ax.plot(X, q_county_raw[q50], linestyle="--", linewidth=1.0, color=county_color, alpha=0.30, label="County Hospital (raw)")
    ax.plot(X, q_county_ma[q50], linestyle="-", linewidth=2.5, color=county_color, label="County Hospital (12m MA)")

    ax.plot(X, q_primary_raw[q50], linestyle="--", linewidth=1.0, color=primary_color, alpha=0.30, label="Primary Care (Towns + Large Towns) (raw)")
    ax.plot(X, q_primary_ma[q50], linestyle="-", linewidth=2.5, color=primary_color, label="Primary Care (Towns + Large Towns) (12m MA)")

    ax.set_ylabel("Share")
    ax.set_xlabel("Time step")
    ax.set_ylim(0.0, 0.9)
    ax.grid(True, alpha=0.35)
    ax.set_title(f"S2 — Share with uncertainty bands (n={n} plausible parameter sets from S1)\n"
                 f"Band: IQR (dark) + 90% UI (light); center line: median")
    ax.legend(frameon=True, fontsize=9, loc="center right")

    out_png = outdir / "s2_flow_share_ui_band.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"Saved: {out_png}")
    print(f"Saved: {csv_path}")

    if not args.no_plot_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
