#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_ui_topk_targets_scaled.py

目的：
- 解决你看到的“保险总支出(10^5量级) vs share(0-1) vs capacity(10^3)”混在一张图导致“都挤在0附近”的问题。
- 读取 outputs_s1_top20/quantiles_topk_targets.json + targets_S1.json
- 自动按量级分组，画成多面板（每个面板一个 x 轴尺度）。

默认读取路径（相对于工作目录）：
- outputs_s1_top20/quantiles_topk_targets.json
- targets_S1.json

输出：
- outputs_s1_top20/ui_topk_targets_scaled.png
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_group(key: str, low: float, high: float) -> str:
    # 用 targets 的量级来分组，优先保证“同尺度”在同一面板
    if high <= 1.5 and low >= 0.0:
        return "share_or_index"
    if high <= 10000:
        return "capacity"
    return "large_scale"


def nice_xlim(vals, low, high, pad_ratio=0.15):
    vmin = min(vals + [low])
    vmax = max(vals + [high])
    span = vmax - vmin
    if span <= 0:
        span = max(1.0, abs(vmax) if vmax != 0 else 1.0)
    pad = span * pad_ratio
    return vmin - pad, vmax + pad


def plot_group(ax, keys, q, targets, title):
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys[::-1]):
        qq = q[k]
        p05, p25, p50, p75, p95 = qq["p05"], qq["p25"], qq["p50"], qq["p75"], qq["p95"]
        ax.plot([p05, p95], [yi, yi], linewidth=2)
        ax.plot([p25, p75], [yi, yi], linewidth=6)
        ax.plot(p50, yi, marker="o", markersize=6)

        low = float(targets[k]["low"])
        high = float(targets[k]["high"])
        ax.axvline(low, linestyle="--", linewidth=1)
        ax.axvline(high, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(keys[::-1])
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs_s1_top20",
                    help="包含 quantiles_topk_targets.json 的输出目录（默认 outputs_s1_top20）")
    ap.add_argument("--quantiles", type=str, default=None,
                    help="quantiles_topk_targets.json 的路径（可选；默认 outdir/quantiles_topk_targets.json）")
    ap.add_argument("--targets", type=str, default="targets_S1.json",
                    help="targets_S1.json 的路径（默认根目录 targets_S1.json）")
    ap.add_argument("--outfile", type=str, default=None,
                    help="输出 PNG 路径（可选；默认 outdir/ui_topk_targets_scaled.png）")
    ap.add_argument("--title", type=str, default="S1 (FFS) — good-fitting parameter sets (scaled panels)",
                    help="图标题前缀")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    q_path = Path(args.quantiles) if args.quantiles else outdir / "quantiles_topk_targets.json"
    # 兼容：如果 quantiles_topk_targets.json 不存在，就退化使用 quantiles_topk.json（里面通常包含更多指标）
    if not q_path.exists():
        alt = outdir / "quantiles_topk.json"
        if alt.exists():
            q_path = alt

    t_path = Path(args.targets)
    out_png = Path(args.outfile) if args.outfile else outdir / "ui_topk_targets_scaled.png"

    q = load_json(q_path)
    targets = load_json(t_path)

    keys = [k for k in targets.keys() if k in q and isinstance(q[k], dict) and "p50" in q[k]]
    if not keys:
        raise RuntimeError("没有找到可绘制的 keys：请检查 quantiles/targets 文件内容与路径。")

    groups = {"share_or_index": [], "capacity": [], "large_scale": []}
    for k in keys:
        low = float(targets[k]["low"])
        high = float(targets[k]["high"])
        groups[infer_group(k, low, high)].append(k)

    groups = {g: ks for g, ks in groups.items() if ks}
    nrows = len(groups)

    fig = plt.figure(figsize=(12, 3.6 * nrows))
    axes = [fig.add_subplot(nrows, 1, i + 1) for i in range(nrows)]

    for ax, (g, ks) in zip(axes, groups.items()):
        all_vals, lows, highs = [], [], []
        for k in ks:
            qq = q[k]
            all_vals += [qq["p05"], qq["p25"], qq["p50"], qq["p75"], qq["p95"]]
            lows.append(float(targets[k]["low"]))
            highs.append(float(targets[k]["high"]))

        xmin, xmax = nice_xlim(all_vals, min(lows), max(highs), pad_ratio=0.15)

        subtitle = {
            "share_or_index": f"{args.title} — shares / indices (0–1-ish)\nMedian(dot), IQR(thick), 90% UI(thin); dashed=target bounds",
            "capacity": f"{args.title} — capacity (10^3-ish)\nMedian(dot), IQR(thick), 90% UI(thin); dashed=target bounds",
            "large_scale": f"{args.title} — large-scale (e.g., spending)\nMedian(dot), IQR(thick), 90% UI(thin); dashed=target bounds",
        }.get(g, args.title)

        plot_group(ax, ks, q, targets, subtitle)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Value")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png.resolve()}")


if __name__ == "__main__":
    main()
