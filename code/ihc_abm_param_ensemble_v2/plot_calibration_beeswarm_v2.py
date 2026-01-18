#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_calibration_beeswarm_v2.py

改进点（针对你说的“看起来怪”）：
1) 修复“中线特别粗”：之前在每一行循环里重复画 axvline(0)，叠加成粗线；现在只画一次。
2) 加入边界提示：在 x=-0.5 和 x=+0.5 画虚线（对应 target 区间上下界），更直观。
3) 支持只画“全部命中 targets 的参数集”：--pass_all_only
4) 颜色更像 SHAP：用区间位置 p=(value-low)/(high-low) 并 clip 到 [0,1]，越接近 low 越蓝、越接近 high 越红。

横轴定义（与你现在图一致）：
    x = (value - mid) / (high - low)
    mid=(low+high)/2
区间内必在 [-0.5, +0.5]。

默认输入/输出（相对工作目录）：
- targets_S1.json
- outputs_s1_top20/topk.csv
- outputs_s1_top20/calibration_beeswarm_v2.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_swarm_offsets(x: np.ndarray, bin_count: int = 28, step: float = 0.085) -> np.ndarray:
    """
    轻量 beeswarm：按 x 分箱，在箱内把点沿 y 对称展开，减少遮挡。
    """
    if x.size == 0:
        return np.array([])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    pad = 0.02 * (xmax - xmin + 1e-9)
    bins = np.linspace(xmin - pad, xmax + pad, bin_count + 1)
    b = np.digitize(x, bins)
    offsets = np.zeros_like(x, dtype=float)

    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        if idx.size <= 1:
            continue
        seq = np.zeros(idx.size, dtype=float)
        for t in range(idx.size):
            if t == 0:
                seq[t] = 0
            else:
                k = (t + 1) // 2
                seq[t] = k if (t % 2 == 1) else -k
        seq = seq * step
        perm = np.random.permutation(idx.size)
        offsets[idx[perm]] = seq
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="targets_S1.json")
    ap.add_argument("--input", type=str, default="outputs_s1_top20_hardflow/topk.csv")
    ap.add_argument("--out", type=str, default="outputs_s1_top20_hardflow/calibration_beeswarm_v2.png")
    ap.add_argument("--title", type=str, default="S1 calibration diagnostics (beeswarm)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--pass_all_only", action="store_true",
                    help="只绘制 pass_all==True 的参数集（如果输入CSV含该列）")
    ap.add_argument("--sort", choices=["none", "mad"], default="mad",
                    help="mad=按|偏离|的中位数排序（更像SHAP）；none=按targets顺序")
    args = ap.parse_args()

    np.random.seed(args.seed)

    targets = load_json(Path(args.targets))
    df = pd.read_csv(Path(args.input))

    if args.pass_all_only and "pass_all" in df.columns:
        df = df[df["pass_all"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    keys = [k for k in targets.keys() if k in df.columns]
    if not keys:
        raise RuntimeError("在输入CSV中找不到任何 targets 指标列；请检查 targets_S1.json key 与 CSV 列名。")

    rows = []
    for k in keys:
        low = float(targets[k]["low"]); high = float(targets[k]["high"])
        width = high - low
        if width <= 0:
            continue
        mid = 0.5 * (low + high)

        v = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue

        x = (v - mid) / width
        p = (v - low) / width
        p_clip = np.clip(p, 0.0, 1.0)
        rows.append((k, x, p_clip))

    if not rows:
        raise RuntimeError("targets 指标列读取失败（可能都是空/不可转换为数值）。")

    if args.sort == "mad":
        rows.sort(key=lambda t: float(np.median(np.abs(t[1]))), reverse=True)

    n = len(rows)
    fig_h = max(4.0, 0.60 * n + 1.1)
    fig = plt.figure(figsize=(12, fig_h))
    ax = fig.add_subplot(111)

    # 区间内带 & 边界线（只画一次）
    ax.axvspan(-0.5, 0.5, alpha=0.10)
    ax.axvline(-0.5, linestyle="--", linewidth=1)
    ax.axvline(+0.5, linestyle="--", linewidth=1)
    ax.axvline(0.0, linewidth=1)

    # 从上到下画
    yticks = []
    ylabels = []
    for i, (k, x, p_clip) in enumerate(rows):
        base_y = n - 1 - i
        offsets = compute_swarm_offsets(x)
        y = base_y + offsets

        if args.no_color:
            ax.scatter(x, y, s=14, alpha=0.85, edgecolors="none")
        else:
            ax.scatter(x, y, c=p_clip, cmap="coolwarm", s=14, alpha=0.85, edgecolors="none")

        yticks.append(base_y)
        ylabels.append(k)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(
        "Normalized deviation from target interval center:  x=(value-mid)/(high-low)\n"
        "In-target region is [-0.5, +0.5] (shaded); dashed lines are bounds"
    )
    suffix = ""
    if args.pass_all_only and "pass_all" in df.columns:
        suffix = f" (pass_all_only, n={len(df)})"
    ax.set_title(args.title + suffix)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
