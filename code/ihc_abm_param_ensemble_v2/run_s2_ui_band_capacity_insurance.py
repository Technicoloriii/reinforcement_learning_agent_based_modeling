#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_s2_ui_band_capacity_insurance.py

一次性生成 S2 的：
1) Capacity share（County vs Primary）带不确定区间（band from RAW runs; center line = MA of q50）
2) Insurance total（总医保支出）带不确定区间（band from RAW runs; center line = MA of q50）
3) Insurance share（County share vs Primary share）带不确定区间（可选，默认也会出）

你只需要：
- topk_params.json（S1筛选出的 plausible 参数集合）
- S2_*.py（你的S2场景脚本；默认 S2_10_15_17_7_nol2.py 在同一目录）

运行方式（PyCharm）：
- 直接 Run 本脚本即可（不需要命令行参数）
"""

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 用户可调的“配置区”
# =========================
HERE = Path(__file__).resolve().parent
TOPK_PARAMS_PATH = HERE / "topk_params.json"
S2_SCRIPT_PATH = HERE / "legacy" / "S2_10_15_17_7_nol2.py"

OUTDIR = HERE / "outputs_uncertainty_s2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# UI 区间：浅色 outer（10-90），深色 inner（25-75）
Q_OUTER = (10, 90)
Q_INNER = (25, 75)

# 线条
MOVAVG_MONTHS = 24          # 24m MA
LINEWIDTH_SOLID = 2.0       # 你想“窄一点”就调小，比如 1.8
LINEWIDTH_DASH = 1.0

# 纯视觉：把“中心线”往 IQR 中心拉一点（0=严格median；1=严格IQR中心）
# 只影响实线（center MA），不影响band
VISUAL_BLEND = 0.6         # 例如 0.3 / 0.5 看看效果

# 颜色（保持你之前的风格）
COUNTY_COLOR = "#2C458D"
PRIMARY_COLOR = "#EF5950"

# 是否保存每次run的原始序列缓存，便于后续重复画图
SAVE_CACHE_NPZ = True


# -------------------------
# Utilities: patch & early-exit
# -------------------------

def patch_assignments(code: str, params: Dict[str, float]) -> str:
    """
    把脚本里形如  PARAM = 0.123  的常量赋值替换成采样值。
    只改“顶层直接赋值”的常量行（不会改函数内部）。
    """
    for key, val in params.items():
        # 只匹配：行首/空格 + key + = + number
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[-+0-9.eE]+(\s*(#.*)?)$"
        repl = f"{key} = {float(val)}\\1"
        code, n = re.subn(pattern, repl, code)
        # 如果没匹配到，就不强求（有的参数可能不在该脚本）
    return code


def inject_ensemble_early_exit(code: str) -> str:
    """
    在脚本末尾注入 early-exit：
    - 让脚本在完成核心模拟与关键数组写入后提前退出
    - 避免大量 plot / 保存 / print（减少时间）
    依赖 S2 脚本里使用 __ENSEMBLE__ 这个开关（我们会在 globals 里提供）
    """
    hook = "\n\n# --- injected by run_s2_ui_band_capacity_insurance.py ---\n" \
           "if globals().get('__ENSEMBLE__', False):\n" \
           "    raise SystemExit\n"
    if hook.strip() in code:
        return code
    return code + hook


def exec_s2_with_params(s2_path: Path, params: Dict[str, float], ensemble_mode: bool = True) -> Dict:
    """
    执行一次 S2，并返回其 globals dict（包含 flow_share_lvl / capacity_trends / insurance_spending 等）。
    """
    code = s2_path.read_text(encoding="utf-8", errors="ignore")
    code = patch_assignments(code, params)
    if ensemble_mode:
        code = inject_ensemble_early_exit(code)

    # headless
    os.environ.setdefault("MPLBACKEND", "Agg")

    g: Dict = {
        "__name__": "__main__",
        "__file__": str(s2_path),
        "__ENSEMBLE__": bool(ensemble_mode),
    }

    try:
        exec(compile(code, str(s2_path), "exec"), g, g)
    except SystemExit:
        pass
    return g


# -------------------------
# Series helpers
# -------------------------

def as_float_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).copy()
    if a.ndim != 1:
        a = a.reshape(-1)
    # NaN/Inf guard
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def safe_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    x2 = x.astype(float, copy=True)
    # forward/back fill for NaN just in case
    if np.isnan(x2).any():
        last = None
        for i in range(len(x2)):
            if np.isfinite(x2[i]):
                last = x2[i]
            elif last is not None:
                x2[i] = last
        nxt = None
        for i in range(len(x2) - 1, -1, -1):
            if np.isfinite(x2[i]):
                nxt = x2[i]
            elif nxt is not None:
                x2[i] = nxt
        x2 = np.nan_to_num(x2, nan=0.0)

    w = int(max(1, w))
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(x2, kernel, mode="same")
    # edges: shrink window
    half = w // 2
    for i in range(half):
        y[i] = x2[: i + half + 1].mean()
        y[-(i + 1)] = x2[-(i + half + 1):].mean()
    return y


def quantiles_over_runs(runs: List[np.ndarray], q_list: List[int]) -> Dict[int, np.ndarray]:
    """
    runs: list of shape (T,) arrays
    returns: {q: (T,) array}
    """
    X = np.stack(runs, axis=0)  # (N, T)
    out = {}
    for q in q_list:
        out[q] = np.quantile(X, q / 100.0, axis=0)
    return out


# -------------------------
# Extract required series from S2 globals
# -------------------------

def extract_capacity_share(g: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns (county_share, primary_share) each shape (T,)
    capacity_trends is bed-days by level (county/secondary/township)
    """
    cap = g.get("capacity_trends", None)
    if not isinstance(cap, dict):
        raise KeyError("capacity_trends not found")
    county = as_float_array(cap.get("county", []))
    secondary = as_float_array(cap.get("secondary", []))
    township = as_float_array(cap.get("township", []))
    T = min(len(county), len(secondary), len(township))
    if T <= 0:
        raise ValueError("capacity_trends series empty")
    county = county[:T]
    primary = secondary[:T] + township[:T]
    denom = county + primary
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    county_share = county / denom
    primary_share = 1.0 - county_share
    return county_share, primary_share


def extract_insurance_total(g: Dict) -> np.ndarray:
    ins = g.get("insurance_spending", None)
    if not isinstance(ins, dict):
        raise KeyError("insurance_spending not found")
    total = as_float_array(ins.get("total", []))
    if len(total) <= 0:
        raise ValueError("insurance_spending['total'] empty")
    return total


def extract_insurance_share(g: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns (county_share, primary_share) each shape (T,)
    """
    ins = g.get("insurance_spending", None)
    if not isinstance(ins, dict):
        raise KeyError("insurance_spending not found")
    county_sh = as_float_array(ins.get("county_share", []))
    secondary_sh = as_float_array(ins.get("secondary_share", []))
    township_sh = as_float_array(ins.get("township_share", []))
    T = min(len(county_sh), len(secondary_sh), len(township_sh))
    if T <= 0:
        raise ValueError("insurance share series empty")
    county_sh = county_sh[:T]
    primary_sh = secondary_sh[:T] + township_sh[:T]
    # 保险share理论上 county+secondary+township=1；这里 primary = 1-county 更稳（避免微小误差）
    primary_sh = 1.0 - county_sh
    return county_sh, primary_sh


# -------------------------
# Plotting
# -------------------------

def plot_share_ui(
    title: str,
    X: np.ndarray,
    q_county: Dict[int, np.ndarray],
    out_png: Path,
    legend_county: str = "County Hospital",
    legend_primary: str = "Primary Care (Towns + Large Towns)",
):
    q_lo_o, q_hi_o = Q_OUTER
    q_lo_i, q_hi_i = Q_INNER
    q_med = 50

    county_med_raw = q_county[q_med]
    county_iqr_mid = 0.5 * (q_county[q_lo_i] + q_county[q_hi_i])
    county_center_raw = (1 - VISUAL_BLEND) * county_med_raw + VISUAL_BLEND * county_iqr_mid
    county_center_ma = safe_moving_average(county_center_raw, MOVAVG_MONTHS)

    # primary = 1 - county (保持完全互补，视觉上更一致)
    primary_q = {q: 1.0 - q_county[q] for q in q_county.keys()}
    primary_med_raw = primary_q[q_med]
    primary_iqr_mid = 0.5 * (primary_q[q_lo_i] + primary_q[q_hi_i])
    primary_center_raw = (1 - VISUAL_BLEND) * primary_med_raw + VISUAL_BLEND * primary_iqr_mid
    primary_center_ma = safe_moving_average(primary_center_raw, MOVAVG_MONTHS)

    fig, ax = plt.subplots(figsize=(13, 6))

    # bands (outer then inner, so inner looks darker)
    ax.fill_between(X, q_county[q_lo_o], q_county[q_hi_o], color=COUNTY_COLOR, alpha=0.10, linewidth=0)
    ax.fill_between(X, primary_q[q_lo_o], primary_q[q_hi_o], color=PRIMARY_COLOR, alpha=0.10, linewidth=0)
    ax.fill_between(X, q_county[q_lo_i], q_county[q_hi_i], color=COUNTY_COLOR, alpha=0.18, linewidth=0)
    ax.fill_between(X, primary_q[q_lo_i], primary_q[q_hi_i], color=PRIMARY_COLOR, alpha=0.18, linewidth=0)

    # lines
    ax.plot(X, county_med_raw, linestyle="--", linewidth=LINEWIDTH_DASH, color=COUNTY_COLOR, alpha=0.30,
            label=f"{legend_county} (raw)")
    ax.plot(X, county_center_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=COUNTY_COLOR,
            label=f"{legend_county} ({MOVAVG_MONTHS}m MA)")

    ax.plot(X, primary_med_raw, linestyle="--", linewidth=LINEWIDTH_DASH, color=PRIMARY_COLOR, alpha=0.30,
            label=f"{legend_primary} (raw)")
    ax.plot(X, primary_center_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=PRIMARY_COLOR,
            label=f"{legend_primary} ({MOVAVG_MONTHS}m MA)")

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_total_ui(
    title: str,
    X: np.ndarray,
    q_total: Dict[int, np.ndarray],
    out_png: Path,
    ylabel: str = "Insurance spending (total)",
):
    q_lo_o, q_hi_o = Q_OUTER
    q_lo_i, q_hi_i = Q_INNER
    q_med = 50

    med_raw = q_total[q_med]
    iqr_mid = 0.5 * (q_total[q_lo_i] + q_total[q_hi_i])
    center_raw = (1 - VISUAL_BLEND) * med_raw + VISUAL_BLEND * iqr_mid
    center_ma = safe_moving_average(center_raw, MOVAVG_MONTHS)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(X, q_total[q_lo_o], q_total[q_hi_o], color=PRIMARY_COLOR, alpha=0.10, linewidth=0)
    ax.fill_between(X, q_total[q_lo_i], q_total[q_hi_i], color=PRIMARY_COLOR, alpha=0.18, linewidth=0)

    ax.plot(X, med_raw, linestyle="--", linewidth=LINEWIDTH_DASH, color=PRIMARY_COLOR, alpha=0.30, label="Median (raw)")
    ax.plot(X, center_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=PRIMARY_COLOR, label=f"Center ({MOVAVG_MONTHS}m MA)")

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    if not TOPK_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing {TOPK_PARAMS_PATH}")
    if not S2_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing {S2_SCRIPT_PATH}")

    topk = json.loads(TOPK_PARAMS_PATH.read_text(encoding="utf-8"))
    if isinstance(topk, dict) and "params" in topk:
        params_list = topk["params"]
    elif isinstance(topk, list):
        params_list = topk
    else:
        raise ValueError("topk_params.json format not recognized (expect list[dict] or {params:[...]})")

    # Collect runs
    cap_share_runs: List[np.ndarray] = []
    ins_total_runs: List[np.ndarray] = []
    ins_share_runs: List[np.ndarray] = []

    ok = 0
    for i, p in enumerate(params_list):
        try:
            g = exec_s2_with_params(S2_SCRIPT_PATH, p, ensemble_mode=True)

            # capacity share
            cap_county_sh, _cap_primary_sh = extract_capacity_share(g)
            # insurance total
            ins_total = extract_insurance_total(g)
            # insurance share (county)
            ins_county_sh, _ins_primary_sh = extract_insurance_share(g)

            # Align lengths (keep common T)
            T = min(len(cap_county_sh), len(ins_total), len(ins_county_sh))
            if T <= 10:
                raise ValueError("too short series")
            cap_share_runs.append(cap_county_sh[:T])
            ins_total_runs.append(ins_total[:T])
            ins_share_runs.append(ins_county_sh[:T])

            ok += 1
        except Exception as e:
            # skip failed run
            continue

    if ok < 5:
        raise RuntimeError(f"Too few successful runs: {ok}")

    # Ensure all runs same length by trimming to minimal T
    T = min(arr.shape[0] for arr in cap_share_runs)
    cap_share_runs = [a[:T] for a in cap_share_runs]
    ins_total_runs = [a[:T] for a in ins_total_runs]
    ins_share_runs = [a[:T] for a in ins_share_runs]

    X = np.arange(T)

    q_list = sorted({Q_OUTER[0], Q_OUTER[1], Q_INNER[0], Q_INNER[1], 50})

    # Quantiles
    q_cap = quantiles_over_runs(cap_share_runs, q_list)
    q_ins_total = quantiles_over_runs(ins_total_runs, q_list)
    q_ins_share = quantiles_over_runs(ins_share_runs, q_list)

    # Save quantiles for reuse
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / "s2_capacity_insurance_quantiles.npz",
             q_list=np.array(q_list),
             cap_share=np.stack([q_cap[q] for q in q_list], axis=0),
             ins_total=np.stack([q_ins_total[q] for q in q_list], axis=0),
             ins_county_share=np.stack([q_ins_share[q] for q in q_list], axis=0),
             T=T)

    if SAVE_CACHE_NPZ:
        np.savez(OUTDIR / "s2_capacity_insurance_runs.npz",
                 cap_share=np.stack(cap_share_runs, axis=0),
                 ins_total=np.stack(ins_total_runs, axis=0),
                 ins_county_share=np.stack(ins_share_runs, axis=0))

    # Plots
    plot_share_ui(
        title=f"S2 — Capacity share with uncertainty bands (n={ok} plausible parameter sets from S1)\n"
              f"Band: {Q_INNER[0]}–{Q_INNER[1]}% (dark) + {Q_OUTER[0]}–{Q_OUTER[1]}% UI (light); center line: MA of q50",
        X=X,
        q_county=q_cap,
        out_png=OUTDIR / "s2_capacity_share_ui_band.png",
        legend_county="County capacity share",
        legend_primary="Primary capacity share",
    )

    plot_total_ui(
        title=f"S2 — Insurance total with uncertainty bands (n={ok} plausible parameter sets from S1)\n"
              f"Band: {Q_INNER[0]}–{Q_INNER[1]}% (dark) + {Q_OUTER[0]}–{Q_OUTER[1]}% UI (light); center line: MA of q50",
        X=X,
        q_total=q_ins_total,
        out_png=OUTDIR / "s2_insurance_total_ui_band.png",
        ylabel="Insurance spending (total)",
    )

    plot_share_ui(
        title=f"S2 — Insurance county share with uncertainty bands (n={ok} plausible parameter sets from S1)\n"
              f"Band: {Q_INNER[0]}–{Q_INNER[1]}% (dark) + {Q_OUTER[0]}–{Q_OUTER[1]}% UI (light); center line: MA of q50",
        X=X,
        q_county=q_ins_share,
        out_png=OUTDIR / "s2_insurance_share_ui_band.png",
        legend_county="County insurance share",
        legend_primary="Primary insurance share",
    )

    print("=== DONE ===")
    print(f"ok_runs={ok}  T={T}")
    print(f"outdir={OUTDIR}")
    print("outputs: s2_capacity_share_ui_band.png, s2_insurance_total_ui_band.png, s2_insurance_share_ui_band.png")


if __name__ == "__main__":
    main()
