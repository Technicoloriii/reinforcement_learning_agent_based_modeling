#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_calibration_beeswarm_v3.py

在 v2 的基础上，专门优化“竖着一列”的视觉呈现（ties / 离散取值）：

新增功能：
1) x 方向的“微抖动”（只对重复取值的点）：--x-jitter
   - 这不会改变统计意义，只是把完全重合的点轻微摊开，避免看到一根“钉子”
   - 默认开启 0.010，建议 0.006~0.015（单位是 normalized x）
2) 可选：点更小 + 更透明：--s, --alpha
3) 可选：每行叠加中位数/四分位：--summary（更像 BMJ 的诊断风格）

横轴定义：
    x = (value - mid) / (high - low)
区间内必在 [-0.5, +0.5]（阴影）。

默认输入/输出：
- targets_S1.json
- outputs_s1_top20_hardflow/topk.csv
- outputs_s1_top20_hardflow/calibration_beeswarm_v3.png
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
    """y 方向 beeswarm（按 x 分箱，y 方向对称展开）"""
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


def jitter_ties(x: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """
    仅对重复取值的 x 做微抖动：
    - 对每个“相同 x”组，添加均值为 0 的小噪声（范围 ~ [-jitter, +jitter]）
    - 用 rounding 聚合，避免浮点误差导致识别不到 ties
    """
    if jitter <= 0:
        return x
    x2 = x.copy()
    key = np.round(x2, 6)
    uniq, counts = np.unique(key, return_counts=True)
    tie_vals = set(uniq[counts > 2])  # 出现>=3次才抖动，避免无意义抖动
    if not tie_vals:
        return x2
    for v in tie_vals:
        idx = np.where(key == v)[0]
        noise = rng.uniform(-jitter, jitter, size=idx.size)
        x2[idx] = x2[idx] + noise
    return x2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="targets_S1.json")
    ap.add_argument("--input", type=str, default="outputs_s1_top20_hardflow/topk.csv")
    ap.add_argument("--out", type=str, default="outputs_s1_top20_hardflow/calibration_beeswarm_v3.png")
    ap.add_argument("--title", type=str, default="S1 calibration diagnostics (beeswarm)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no_color", action="store_true")
    ap.add_argument("--sort", choices=["none", "mad"], default="mad")
    ap.add_argument("--x-jitter", type=float, default=0.040,
                    help="对重复 x 值做微抖动的幅度（normalized x 单位）。0=关闭。建议 0.006~0.015")
    ap.add_argument("--s", type=float, default=13.0, help="点大小")
    ap.add_argument("--alpha", type=float, default=0.78, help="点透明度")
    ap.add_argument("--summary", action="store_true",
                    help="叠加每行的中位数与IQR（短横线）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    targets = load_json(Path(args.targets))
    df = pd.read_csv(Path(args.input))

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

    if args.sort == "mad":
        rows.sort(key=lambda t: float(np.median(np.abs(t[1]))), reverse=True)

    n = len(rows)
    fig_h = max(4.0, 0.62 * n + 1.1)
    fig = plt.figure(figsize=(12, fig_h))
    ax = fig.add_subplot(111)

    ax.axvspan(-0.5, 0.5, alpha=0.10)
    ax.axvline(-0.5, linestyle="--", linewidth=1)
    ax.axvline(+0.5, linestyle="--", linewidth=1)
    ax.axvline(0.0, linewidth=1)

    yticks, ylabels = [], []
    for i, (k, x, p_clip) in enumerate(rows):
        base_y = n - 1 - i

        x_plot = jitter_ties(x, args.x_jitter, rng)
        offsets = compute_swarm_offsets(x_plot)
        y = base_y + offsets

        if args.no_color:
            ax.scatter(x_plot, y, s=args.s, alpha=args.alpha, edgecolors="none")
        else:
            ax.scatter(x_plot, y, c=p_clip, cmap="coolwarm", s=args.s, alpha=args.alpha, edgecolors="none")

        if args.summary:
            q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
            ax.plot([q25, q75], [base_y, base_y], linewidth=6)
            ax.plot(q50, base_y, marker="o", markersize=6)

        yticks.append(base_y)
        ylabels.append(k)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(
        "Normalized deviation from target interval center:  x=(value-mid)/(high-low)\n"
        "In-target region is [-0.5, +0.5] (shaded); dashed lines are bounds"
    )
    ax.set_title(args.title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
