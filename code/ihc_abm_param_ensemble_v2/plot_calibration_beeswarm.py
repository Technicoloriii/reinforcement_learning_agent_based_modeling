#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_calibration_beeswarm.py

做一张“校准诊断视角”的 beeswarm 图（风格类似 SHAP summary）：
- 每一行是一个 calibration target 指标（targets_S1.json 里定义的 keys）
- 每个点是一组参数（来自 topk.csv / scored_candidates.csv）
- 横轴是“相对 target 区间的归一化偏离”（默认以区间中心为 0）：
    x = (value - mid) / (high - low)
  其中 mid=(low+high)/2
  => 落在 target 区间内的点必然落在 [-0.5, +0.5]
- 背景阴影标出 “在区间内”的区域（[-0.5, 0.5]）
- 点颜色用 0..1 的区间位置（(value-low)/(high-low)）映射，类似 SHAP 的 red/blue
  （你也可以关掉颜色映射，改成单色）

默认输入（相对工作目录）：
- targets_S1.json
- outputs_s1_top20/topk.csv   （你可以换成 scored_candidates.csv 看全体）

输出：
- outputs_s1_top20/calibration_beeswarm.png

PyCharm 里运行：
- 把本脚本放到项目根目录
- 右键 Run 即可（确保 Working directory 是项目根目录）
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


def compute_swarm_offsets(x: np.ndarray, bin_count: int = 28, step: float = 0.08) -> np.ndarray:
    """
    一个轻量 swarm：把 x 分箱，在每个箱内把点沿 y 方向对称展开，减少遮挡。
    """
    if x.size == 0:
        return np.array([])
    # 为了更稳定的布局，用固定 bins 覆盖范围
    xmin, xmax = float(np.min(x)), float(np.max(x))
    pad = 0.02 * (xmax - xmin + 1e-9)
    bins = np.linspace(xmin - pad, xmax + pad, bin_count + 1)
    b = np.digitize(x, bins)  # 1..bin_count+1
    offsets = np.zeros_like(x, dtype=float)

    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        if idx.size <= 1:
            continue
        # 为该 bin 的点生成对称 offset：0, +1, -1, +2, -2, ...
        order = np.arange(idx.size)
        seq = np.zeros(idx.size, dtype=float)
        k = 0
        for t in range(idx.size):
            if t == 0:
                seq[t] = 0
            else:
                k = (t + 1) // 2
                seq[t] = k if (t % 2 == 1) else -k
        seq = seq * step
        # 为了不让 x 顺序影响图形，随机打散后再赋值
        perm = np.random.permutation(idx.size)
        offsets[idx[perm]] = seq
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="targets_S1.json", help="targets_S1.json 路径")
    ap.add_argument("--input", type=str, default="outputs_s1_top20/topk.csv",
                    help="输入CSV：topk.csv 或 scored_candidates.csv")
    ap.add_argument("--out", type=str, default="outputs_s1_top20/calibration_beeswarm.png", help="输出PNG路径")
    ap.add_argument("--title", type=str, default="S1 calibration diagnostics (beeswarm)",
                    help="图标题")
    ap.add_argument("--seed", type=int, default=123, help="随机种子（用于 swarm 打散）")
    ap.add_argument("--no_color", action="store_true", help="不使用颜色映射（点为单色）")
    ap.add_argument("--sort", choices=["none", "mad", "score"], default="mad",
                    help="y轴排序：mad=按|偏离|的中位数排序；score=如果CSV里有score则按score排序；none=按targets顺序")
    args = ap.parse_args()

    np.random.seed(args.seed)

    t_path = Path(args.targets)
    in_path = Path(args.input)
    out_path = Path(args.out)

    targets = load_json(t_path)
    df = pd.read_csv(in_path)

    # 只使用 targets 中存在的指标
    keys = [k for k in targets.keys() if k in df.columns]
    if not keys:
        raise RuntimeError("在输入CSV中找不到任何 targets 指标列，请检查 targets_S1.json 的 key 是否与 CSV 列名一致。")

    # 为每个 target 计算 x（中心化偏离）与 color（区间位置）
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

        x = (v - mid) / width              # 0 表示刚好在区间中心
        c = (v - low) / width              # 0..1 表示在区间里的位置（<0 / >1 表示越界）

        rows.append((k, x, c, low, high))

    if not rows:
        raise RuntimeError("targets 指标列读取失败（可能都是空/不可转换为数值）。")

    # 排序（更像 SHAP：贡献大的排上面）
    if args.sort == "mad":
        rows.sort(key=lambda t: float(np.median(np.abs(t[1]))), reverse=True)
    elif args.sort == "score" and "score" in df.columns:
        # score 排序：这里按“与 score 最相关的指标”不严格，所以只是保持 targets 顺序更好
        # 但用户如果想按 score，本质上应该先选 top-k，这里就不硬做复杂逻辑了
        pass

    # 绘图
    fig_h = max(4.0, 0.55 * len(rows) + 1.2)
    fig = plt.figure(figsize=(12, fig_h))
    ax = fig.add_subplot(111)

    # 区间内带：[-0.5, 0.5]
    ax.axvspan(-0.5, 0.5, alpha=0.12)

    for i, (k, x, c, low, high) in enumerate(rows):
        base_y = len(rows) - 1 - i
        offsets = compute_swarm_offsets(x, bin_count=28, step=0.09)
        y = base_y + offsets

        if args.no_color:
            ax.scatter(x, y, s=14, alpha=0.85, edgecolors="none")
        else:
            # 使用蓝-红渐变（类似 SHAP），不在图上写 colorbar，保持干净
            ax.scatter(x, y, c=c, cmap="coolwarm", s=14, alpha=0.85, edgecolors="none")

        # 画 target 中心线
        ax.axvline(0.0, linewidth=1)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Normalized deviation from target interval center  x=(value-mid)/(high-low)\n"
                  "In-target region is [-0.5, +0.5] (shaded)")
    ax.set_title(args.title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
