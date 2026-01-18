#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_s2_capacity_insurance_outputs_v1.py

按你的最新需求（更贴近“原版图”风格）生成 S2 的 4 张图：

(1) Capacity（床位/床日容量）——画“绝对容量（Bed Days）”并带不确定区间（更窄区间）
    - 两条线：County vs Primary(Secondary+Township)
    - band：默认更窄（outer=25–75, inner=40–60）→ 只是为了视觉更收敛

(2) Capacity share —— 用 stack（只画 median stack，不做区间）

(3) Insurance spending（金额）——沿用你原来的“双 y 轴”形式 + 不确定区间
    - 左轴：Total
    - 右轴：County amount & Primary amount(Secondary+Township)
    - band：outer=10–90, inner=25–75

(4) Insurance spending share —— 用 stack（只画 median stack，不做区间）

输入：
- topk_params.json（S1筛选出的 plausible 参数集合）
- legacy/S2_10_15_17_7_nol2.py（或你的S2脚本路径）

输出目录：
- outputs_uncertainty_s2/
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # IMPORTANT: avoid GUI windows / blocking

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 配置区（你只需要改这里）
# =========================
HERE = Path(__file__).resolve().parent

TOPK_PARAMS_PATH = HERE / "topk_params.json"

# S2 在 legacy 目录下（你刚刚确认的结构）
S2_SCRIPT_PATH = HERE / "legacy" / "S2_10_15_17_7_nol2.py"

OUTDIR = HERE / "outputs_uncertainty_s2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 只跑前 N 组（调试用）；0 = 全部
MAX_N = 0

# 线条平滑
MOVAVG_MONTHS = 12

# 线宽（想“窄一点”就调小）
LINEWIDTH_SOLID = 2.2
LINEWIDTH_DASH = 1.0

# 纯视觉：中心线向 IQR 中心拉（0=严格q50；>0 仅用于看效果）
VISUAL_BLEND = 0.0

# 颜色（保持你之前风格）
COUNTY_COLOR = "#2C458D"
PRIMARY_COLOR = "#EF5950"
SECONDARY_COLOR = "#8B5CF6"  # 随便选的紫（stack用）；你也可改成你原 COLORS["secondary"]

# ——区间设置——
# Capacity（绝对容量）用更窄 band（让图更“收敛”，你说只是视觉，不用于文章也OK）
CAP_OUTER = (25, 75)   # light
CAP_INNER = (40, 60)   # dark (very narrow)

# Insurance（金钱金额）保持之前的 10-90 / 25-75
INS_OUTER = (10, 90)   # light
INS_INNER = (25, 75)   # dark


# =========================
# Core utilities
# =========================
def patch_assignments(code: str, params: Dict[str, float]) -> str:
    """Replace top-level constant assignments PARAM = number."""
    for key, val in params.items():
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*[-+0-9.eE]+(\s*(#.*)?)$"
        repl = f"{key} = {float(val)}\\1"
        code, _ = re.subn(pattern, repl, code)
    return code


def inject_ensemble_exit_before_plotting(code: str) -> str:
    """
    Speed-up: exit right before plotting section.
    In your S2 script, plotting starts after `X = np.arange(T)`.
    We insert:
        if __ENSEMBLE__: raise SystemExit
    before that line.
    """
    marker = r"(?m)^\s*X\s*=\s*np\.arange\(T\)\s*$"
    if re.search(marker, code):
        replacement = (
            "if globals().get('__ENSEMBLE__', False):\n"
            "    raise SystemExit\n\n"
            "X = np.arange(T)"
        )
        code, n = re.subn(marker, replacement, code, count=1)
        if n > 0:
            return code

    # fallback: append at end
    return code + "\n\nif globals().get('__ENSEMBLE__', False):\n    raise SystemExit\n"


def exec_s2_with_params(s2_path: Path, params: Dict[str, float]) -> Dict:
    """Execute one S2 run with patched params; return globals dict."""
    code = s2_path.read_text(encoding="utf-8", errors="ignore")
    code = patch_assignments(code, params)
    code = inject_ensemble_exit_before_plotting(code)

    g: Dict = {"__name__": "__main__", "__file__": str(s2_path), "__ENSEMBLE__": True}
    try:
        exec(compile(code, str(s2_path), "exec"), g, g)
    except SystemExit:
        pass
    return g


def as_float_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).copy()
    if a.ndim != 1:
        a = a.reshape(-1)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def safe_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    x2 = x.astype(float, copy=True)
    if np.isnan(x2).any():
        x2 = np.nan_to_num(x2, nan=np.nanmean(x2) if np.isfinite(np.nanmean(x2)) else 0.0)
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(x2, kernel, mode="same")
    half = w // 2
    for i in range(half):
        y[i] = x2[: i + half + 1].mean()
        y[-(i + 1)] = x2[-(i + half + 1):].mean()
    return y


def quantiles_over_runs(runs: List[np.ndarray], q_list: List[int]) -> Dict[int, np.ndarray]:
    X = np.stack(runs, axis=0)  # (N,T)
    return {q: np.quantile(X, q / 100.0, axis=0) for q in q_list}


# =========================
# Extraction
# =========================
def extract_capacity_series(g: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns county, secondary, township capacity series (Bed Days).
    """
    cap = g.get("capacity_trends", None)
    if not isinstance(cap, dict):
        raise KeyError("capacity_trends not found in globals")
    county = as_float_array(cap.get("county", []))
    secondary = as_float_array(cap.get("secondary", []))
    township = as_float_array(cap.get("township", []))
    T = min(len(county), len(secondary), len(township))
    if T <= 10:
        raise ValueError("capacity_trends too short")
    return county[:T], secondary[:T], township[:T]


def extract_insurance_amounts(g: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    returns total, county, secondary, township insurance amounts.
    """
    ins = g.get("insurance_spending", None)
    if not isinstance(ins, dict):
        raise KeyError("insurance_spending not found in globals")
    total = as_float_array(ins.get("total", []))
    county = as_float_array(ins.get("county", []))
    secondary = as_float_array(ins.get("secondary", []))
    township = as_float_array(ins.get("township", []))
    T = min(len(total), len(county), len(secondary), len(township))
    if T <= 10:
        raise ValueError("insurance_spending too short")
    return total[:T], county[:T], secondary[:T], township[:T]

COUNTY_BAND_SHRINK = 0.25
# =========================
# Plotting helpers
# =========================
def plot_two_amounts_with_bands(
    title: str,
    X: np.ndarray,
    q_a: Dict[int, np.ndarray],
    q_b: Dict[int, np.ndarray],
    out_png: Path,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    outer: Tuple[int, int],
    inner: Tuple[int, int],
    ylabel: str,
):
    lo_o, hi_o = outer
    lo_i, hi_i = inner
    q_med = 50

    # raw medians
    a_med = q_a[q_med]
    b_med = q_b[q_med]

    # optional visual blend toward IQR mid (purely for aesthetics)
    a_iqr_mid = 0.5 * (q_a[lo_i] + q_a[hi_i])
    b_iqr_mid = 0.5 * (q_b[lo_i] + q_b[hi_i])

    a_center = (1 - VISUAL_BLEND) * a_med + VISUAL_BLEND * a_iqr_mid
    b_center = (1 - VISUAL_BLEND) * b_med + VISUAL_BLEND * b_iqr_mid

    a_center_ma = safe_moving_average(a_center, MOVAVG_MONTHS)
    b_center_ma = safe_moving_average(b_center, MOVAVG_MONTHS)

    fig, ax = plt.subplots(figsize=(13, 6))

    # bands: outer then inner
    # bands: outer then inner
    k = COUNTY_BAND_SHRINK  # 0~1，越小越“扁”（更窄）
    a_lo_o = a_med - (a_med - q_a[lo_o]) * k
    a_hi_o = a_med + (q_a[hi_o] - a_med) * k
    a_lo_i = a_med - (a_med - q_a[lo_i]) * k
    a_hi_i = a_med + (q_a[hi_i] - a_med) * k

    ax.fill_between(X, a_lo_o, a_hi_o, color=color_a, alpha=0.10, linewidth=0)  # county outer (shrunk)
    ax.fill_between(X, q_b[lo_o], q_b[hi_o], color=color_b, alpha=0.10, linewidth=0)  # primary outer (raw)

    ax.fill_between(X, a_lo_i, a_hi_i, color=color_a, alpha=0.18, linewidth=0)  # county inner (shrunk)
    ax.fill_between(X, q_b[lo_i], q_b[hi_i], color=color_b, alpha=0.18, linewidth=0)  # primary inner (raw)


    # lines
    ax.plot(X, a_med, linestyle="--", linewidth=LINEWIDTH_DASH, color=color_a, alpha=0.30, label=f"{label_a} (raw q50)")
    ax.plot(X, a_center_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=color_a, label=f"{label_a} ({MOVAVG_MONTHS}W MA)")

    ax.plot(X, b_med, linestyle="--", linewidth=LINEWIDTH_DASH, color=color_b, alpha=0.30, label=f"{label_b} (raw q50)")
    ax.plot(X, b_center_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=color_b, label=f"{label_b} ({MOVAVG_MONTHS}W MA)")

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_capacity_share_stack_median(
    title: str,
    X: np.ndarray,
    county_share_med: np.ndarray,
    secondary_share_med: np.ndarray,
    township_share_med: np.ndarray,
    out_png: Path,
):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(
        X,
        county_share_med,
        secondary_share_med,
        township_share_med,
        labels=["County", "Secondary", "Township"],
        colors=[COUNTY_COLOR, SECONDARY_COLOR, PRIMARY_COLOR],
        alpha=0.75,
    )
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Capacity share (median)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper left", frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_insurance_dual_axis_with_bands(
    title: str,
    X: np.ndarray,
    q_total: Dict[int, np.ndarray],
    q_county: Dict[int, np.ndarray],
    q_primary: Dict[int, np.ndarray],
    out_png: Path,
    outer: Tuple[int, int],
    inner: Tuple[int, int],
):
    lo_o, hi_o = outer
    lo_i, hi_i = inner
    q_med = 50

    total_med = q_total[q_med]
    total_iqr_mid = 0.5 * (q_total[lo_i] + q_total[hi_i])
    total_center = (1 - VISUAL_BLEND) * total_med + VISUAL_BLEND * total_iqr_mid
    total_ma = safe_moving_average(total_center, MOVAVG_MONTHS)

    county_med = q_county[q_med]
    county_iqr_mid = 0.5 * (q_county[lo_i] + q_county[hi_i])
    county_center = (1 - VISUAL_BLEND) * county_med + VISUAL_BLEND * county_iqr_mid
    county_ma = safe_moving_average(county_center, MOVAVG_MONTHS)

    primary_med = q_primary[q_med]
    primary_iqr_mid = 0.5 * (q_primary[lo_i] + q_primary[hi_i])
    primary_center = (1 - VISUAL_BLEND) * primary_med + VISUAL_BLEND * primary_iqr_mid
    primary_ma = safe_moving_average(primary_center, MOVAVG_MONTHS)

    fig, axL = plt.subplots(figsize=(13, 6))
    axR = axL.twinx()

    # Left axis: total (black/grey)
    axL.fill_between(X, q_total[lo_o], q_total[hi_o], color="black", alpha=0.08, linewidth=0)
    axL.fill_between(X, q_total[lo_i], q_total[hi_i], color="black", alpha=0.14, linewidth=0)
    l1 = axL.plot(X, total_med, linestyle="--", linewidth=LINEWIDTH_DASH, color="black", alpha=0.35, label="Total (raw q50)")
    l2 = axL.plot(X, total_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color="black", label=f"Total ({MOVAVG_MONTHS}w MA)")

    axL.set_ylabel("Insurance spending (Total)", fontsize=12, color="black")

    # Right axis: county & primary amounts
    axR.fill_between(X, q_county[lo_o], q_county[hi_o], color=COUNTY_COLOR, alpha=0.10, linewidth=0)
    axR.fill_between(X, q_primary[lo_o], q_primary[hi_o], color=PRIMARY_COLOR, alpha=0.10, linewidth=0)
    axR.fill_between(X, q_county[lo_i], q_county[hi_i], color=COUNTY_COLOR, alpha=0.18, linewidth=0)
    axR.fill_between(X, q_primary[lo_i], q_primary[hi_i], color=PRIMARY_COLOR, alpha=0.18, linewidth=0)

    r1 = axR.plot(X, county_med, linestyle="--", linewidth=LINEWIDTH_DASH, color=COUNTY_COLOR, alpha=0.30, label="County (raw q50)")
    r2 = axR.plot(X, county_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=COUNTY_COLOR, label=f"County ({MOVAVG_MONTHS}w MA)")

    r3 = axR.plot(X, primary_med, linestyle="--", linewidth=LINEWIDTH_DASH, color=PRIMARY_COLOR, alpha=0.30, label="Primary (raw q50)")
    r4 = axR.plot(X, primary_ma, linestyle="-", linewidth=LINEWIDTH_SOLID, color=PRIMARY_COLOR, label=f"Primary ({MOVAVG_MONTHS}w MA)")

    axR.set_ylabel("Insurance spending (by level)", fontsize=12)

    axL.set_title(title, fontsize=18, pad=14)
    axL.set_xlabel("Time step", fontsize=12)
    axL.grid(True, alpha=0.25)

    # Combine legends from both axes
    handles = l1 + l2 + r1 + r2 + r3 + r4
    labels = [h.get_label() for h in handles]
    axL.legend(handles, labels, loc="upper left", frameon=True, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_insurance_share_stack_median(
    title: str,
    X: np.ndarray,
    county_share_med: np.ndarray,
    primary_share_med: np.ndarray,
    out_png: Path,
):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(
        X,
        county_share_med,
        primary_share_med,
        labels=["County Hospital", "Primary Care (Towns + Large Towns)"],
        colors=[COUNTY_COLOR, PRIMARY_COLOR],
        alpha=0.75,
    )
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Insurance spending share (median)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper left", frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# =========================
# Main
# =========================
def load_params_list(path: Path) -> List[Dict[str, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    if isinstance(obj, list):
        return obj
    raise ValueError("topk_params.json must be list[dict] or {params:[...]} format")


def main():
    if not TOPK_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing {TOPK_PARAMS_PATH}")
    if not S2_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing {S2_SCRIPT_PATH} (check legacy path)")

    params_list = load_params_list(TOPK_PARAMS_PATH)
    if MAX_N and MAX_N > 0:
        params_list = params_list[:MAX_N]

    # Collect runs
    cap_county_runs: List[np.ndarray] = []
    cap_primary_runs: List[np.ndarray] = []

    cap_share_county_runs: List[np.ndarray] = []
    cap_share_secondary_runs: List[np.ndarray] = []
    cap_share_township_runs: List[np.ndarray] = []

    ins_total_runs: List[np.ndarray] = []
    ins_county_runs: List[np.ndarray] = []
    ins_primary_runs: List[np.ndarray] = []

    ins_share_county_runs: List[np.ndarray] = []
    ins_share_primary_runs: List[np.ndarray] = []

    ok = 0
    for p in params_list:
        try:
            g = exec_s2_with_params(S2_SCRIPT_PATH, p)

            cap_county, cap_secondary, cap_township = extract_capacity_series(g)
            ins_total, ins_county, ins_secondary, ins_township = extract_insurance_amounts(g)

            # align by common T for this run
            T = min(len(cap_county), len(cap_secondary), len(cap_township),
                    len(ins_total), len(ins_county), len(ins_secondary), len(ins_township))
            if T <= 10:
                continue

            cap_county = cap_county[:T]
            cap_secondary = cap_secondary[:T]
            cap_township = cap_township[:T]
            cap_primary = cap_secondary + cap_township

            # capacity shares for stack (per run)
            cap_total = np.where((cap_county + cap_secondary + cap_township) <= 1e-12, 1e-12, (cap_county + cap_secondary + cap_township))
            cap_share_county = cap_county / cap_total
            cap_share_secondary = cap_secondary / cap_total
            cap_share_township = cap_township / cap_total

            ins_total = ins_total[:T]
            ins_county = ins_county[:T]
            ins_secondary = ins_secondary[:T]
            ins_township = ins_township[:T]
            ins_primary = ins_secondary + ins_township

            # insurance shares for stack (2-level, as original S2 fig)
            ins_total_safe = np.where(ins_total <= 1e-12, 1e-12, ins_total)
            ins_share_county = ins_county / ins_total_safe
            ins_share_primary = 1.0 - ins_share_county

            cap_county_runs.append(cap_county)
            cap_primary_runs.append(cap_primary)

            cap_share_county_runs.append(cap_share_county)
            cap_share_secondary_runs.append(cap_share_secondary)
            cap_share_township_runs.append(cap_share_township)

            ins_total_runs.append(ins_total)
            ins_county_runs.append(ins_county)
            ins_primary_runs.append(ins_primary)

            ins_share_county_runs.append(ins_share_county)
            ins_share_primary_runs.append(ins_share_primary)

            ok += 1
        except Exception:
            continue

    if ok < 5:
        raise RuntimeError(f"Too few successful runs: {ok}")

    # Trim all runs to minimal T across runs
    T = min(a.shape[0] for a in cap_county_runs + cap_primary_runs +
            ins_total_runs + ins_county_runs + ins_primary_runs)
    cap_county_runs = [a[:T] for a in cap_county_runs]
    cap_primary_runs = [a[:T] for a in cap_primary_runs]

    cap_share_county_runs = [a[:T] for a in cap_share_county_runs]
    cap_share_secondary_runs = [a[:T] for a in cap_share_secondary_runs]
    cap_share_township_runs = [a[:T] for a in cap_share_township_runs]

    ins_total_runs = [a[:T] for a in ins_total_runs]
    ins_county_runs = [a[:T] for a in ins_county_runs]
    ins_primary_runs = [a[:T] for a in ins_primary_runs]

    ins_share_county_runs = [a[:T] for a in ins_share_county_runs]
    ins_share_primary_runs = [a[:T] for a in ins_share_primary_runs]

    X = np.arange(T)

    # --- Quantiles for capacity amounts (narrow)
    q_list_cap = sorted({CAP_OUTER[0], CAP_OUTER[1], CAP_INNER[0], CAP_INNER[1], 50})
    q_cap_county = quantiles_over_runs(cap_county_runs, q_list_cap)
    q_cap_primary = quantiles_over_runs(cap_primary_runs, q_list_cap)

    # --- Quantiles for insurance amounts (dual-axis)
    q_list_ins = sorted({INS_OUTER[0], INS_OUTER[1], INS_INNER[0], INS_INNER[1], 50})
    q_ins_total = quantiles_over_runs(ins_total_runs, q_list_ins)
    q_ins_county = quantiles_over_runs(ins_county_runs, q_list_ins)
    q_ins_primary = quantiles_over_runs(ins_primary_runs, q_list_ins)

    # --- Median stacks
    cap_share_county_med = np.median(np.stack(cap_share_county_runs, axis=0), axis=0)
    cap_share_secondary_med = np.median(np.stack(cap_share_secondary_runs, axis=0), axis=0)
    cap_share_township_med = np.median(np.stack(cap_share_township_runs, axis=0), axis=0)
    cap_sum = cap_share_county_med + cap_share_secondary_med + cap_share_township_med
    cap_sum = np.where(cap_sum <= 1e-12, 1e-12, cap_sum)
    cap_share_county_med /= cap_sum
    cap_share_secondary_med /= cap_sum
    cap_share_township_med /= cap_sum

    ins_share_county_med = np.median(np.stack(ins_share_county_runs, axis=0), axis=0)
    ins_share_primary_med = np.median(np.stack(ins_share_primary_runs, axis=0), axis=0)
    ins_sum = ins_share_county_med + ins_share_primary_med
    ins_sum = np.where(ins_sum <= 1e-12, 1e-12, ins_sum)
    ins_share_county_med /= ins_sum
    ins_share_primary_med /= ins_sum

    # Save a cache (optional but useful)
    np.savez(
        OUTDIR / "s2_capacity_insurance_outputs_cache.npz",
        T=T,
        ok_runs=ok,
        q_list_cap=np.array(q_list_cap),
        q_list_ins=np.array(q_list_ins),
        cap_county_q=np.stack([q_cap_county[q] for q in q_list_cap], axis=0),
        cap_primary_q=np.stack([q_cap_primary[q] for q in q_list_cap], axis=0),
        ins_total_q=np.stack([q_ins_total[q] for q in q_list_ins], axis=0),
        ins_county_q=np.stack([q_ins_county[q] for q in q_list_ins], axis=0),
        ins_primary_q=np.stack([q_ins_primary[q] for q in q_list_ins], axis=0),
        cap_share_median=np.stack([cap_share_county_med, cap_share_secondary_med, cap_share_township_med], axis=0),
        ins_share_median=np.stack([ins_share_county_med, ins_share_primary_med], axis=0),
    )

    # =========================
    # (1) Capacity amounts with narrower bands
    # =========================
    plot_two_amounts_with_bands(
        title=f"S2 — Capacity (Bed Days) with uncertainty bands (n={ok} plausible sets from S1)\n"
              f"Band: {CAP_INNER[0]}–{CAP_INNER[1]}% (dark) + {CAP_OUTER[0]}–{CAP_OUTER[1]}% (light); center line: MA of q50",
        X=X,
        q_a=q_cap_county,
        q_b=q_cap_primary,
        out_png=OUTDIR / "s2_capacity_beddays_ui_band.png",
        label_a="County capacity (bed days)",
        label_b="Primary capacity (bed days)",
        color_a=COUNTY_COLOR,
        color_b=PRIMARY_COLOR,
        outer=CAP_OUTER,
        inner=CAP_INNER,
        ylabel="Total capacity (beds)",
    )

    # =========================
    # (2) Capacity share stack (median only)
    # =========================
    plot_capacity_share_stack_median(
        title=f"S2 — Capacity share (median stack, n={ok} plausible sets from S1)",
        X=X,
        county_share_med=cap_share_county_med,
        secondary_share_med=cap_share_secondary_med,
        township_share_med=cap_share_township_med,
        out_png=OUTDIR / "s2_capacity_share_stack_median.png",
    )

    # =========================
    # (3) Insurance amounts dual-axis + bands
    # =========================
    plot_insurance_dual_axis_with_bands(
        title=f"S2 — Insurance spending (dual-axis) with uncertainty bands (n={ok} plausible sets from S1)\n"
              f"Band: {INS_INNER[0]}–{INS_INNER[1]}% (dark) + {INS_OUTER[0]}–{INS_OUTER[1]}% (light); center line: MA of q50",
        X=X,
        q_total=q_ins_total,
        q_county=q_ins_county,
        q_primary=q_ins_primary,
        out_png=OUTDIR / "s2_insurance_dualaxis_ui_band.png",
        outer=INS_OUTER,
        inner=INS_INNER,
    )

    # =========================
    # (4) Insurance share stack (median only)
    # =========================
    plot_insurance_share_stack_median(
        title=f"S2 — Insurance spending share (median stack, n={ok} plausible sets from S1)",
        X=X,
        county_share_med=ins_share_county_med,
        primary_share_med=ins_share_primary_med,
        out_png=OUTDIR / "s2_insurance_share_stack_median.png",
    )

    print("=== DONE ===")
    print(f"ok_runs={ok}  T={T}")
    print(f"outdir={OUTDIR}")
    print("outputs:")
    print(" - s2_capacity_beddays_ui_band.png")
    print(" - s2_capacity_share_stack_median.png")
    print(" - s2_insurance_dualaxis_ui_band.png")
    print(" - s2_insurance_share_stack_median.png")


if __name__ == "__main__":
    main()
