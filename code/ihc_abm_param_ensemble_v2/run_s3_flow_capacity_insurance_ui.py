# -*- coding: utf-8 -*-
"""
Run S3 (legacy) with an ensemble of plausible parameter sets (from S1 top-k),
then generate UI-band plots for:
  1) Flow share (County vs Primary=Secondary+Township)  [bands]
  2) Capacity (bed days) (County vs Primary)            [bands]
  3) Capacity share stack (median only)                 [stack, no bands]
  4) Insurance amounts (dual y-axis)                    [bands]
  5) Insurance share stack (median only)                [stack, no bands]

Bands default to:
  - dark: 40–60%
  - light: 25–75%
Center line: moving-average of q50 (median)

Moving average window: 24 (weeks) by default.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# =========================
# Defaults (edit here)
# =========================
HERE = Path(__file__).resolve().parent

# Your S3 legacy script (note: inside legacy/ under the project root)
S3_SCRIPT_PATH = HERE / "legacy" / "S3_10_17_02_20.py"

# Plausible parameter sets from S1
TOPK_PARAMS_PATH = HERE / "topk_params.json"

# Output folder for plots
OUT_DIR = HERE / "outputs_s3_ui"

# Use first N parameter sets (None = use all)
MAX_RUNS = None

# Moving average window (weeks)
MOVAVG_W = 24

# UI bands
DARK_Q = (0.40, 0.60)
LIGHT_Q = (0.25, 0.75)

# Style
COUNTY_COLOR = "#1f3b82"   # deep blue
PRIMARY_COLOR = "#e74c3c"  # red
RAW_ALPHA = 0.35

# Band opacity (light should be lighter than dark)
LIGHT_ALPHA = 0.12
DARK_ALPHA = 0.22

# If a band looks too “wide” visually, you can shrink it (visual-only):
#   1.0 = no shrink; 0.6 makes it 40% narrower around the center.
# Leave 1.0 for honest uncertainty; decrease for *visual exploration only*.
VISUAL_SHRINK_BAND = 1.0


# =========================
# Utilities: patch + early exit
# =========================
_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*=\s*([^\n#]+)(\s*(#.*)?)?$")

def patch_assignments(code_text: str, params: Dict[str, Any]) -> str:
    """
    Replace simple 'NAME = value' assignments in the target script with values from params.
    Only patches top-level scalar/array literals that match this simple assignment pattern.
    """
    lines = code_text.splitlines()
    out = []
    for line in lines:
        m = _ASSIGN_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent, name, rhs, tail = m.group(1), m.group(2), m.group(3), (m.group(4) or "")
        if name in params:
            v = params[name]
            # Serialize value in a python-literal-ish way
            if isinstance(v, bool):
                new_rhs = "True" if v else "False"
            elif v is None:
                new_rhs = "None"
            elif isinstance(v, (int, float)):
                # keep enough precision
                new_rhs = repr(float(v)) if isinstance(v, float) else str(v)
            elif isinstance(v, (list, tuple, np.ndarray)):
                new_rhs = repr(list(v))
            else:
                # fallback: repr
                new_rhs = repr(v)
            out.append(f"{indent}{name} = {new_rhs}{tail}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def inject_ensemble_exit_before_plotting(code_text: str) -> str:
    """
    Insert a SystemExit right before the plotting block.
    We look for a common marker used in your legacy scripts: 'X = np.arange(T)'.
    """
    lines = code_text.splitlines()
    out = []
    inserted = False
    for line in lines:
        # This marker exists in your legacy S2/S3 scripts
        if (not inserted) and re.match(r"^\s*X\s*=\s*np\.arange\(T\)\s*$", line):
            out.append("if True:\n    # Ensemble wrapper: stop before plotting\n    raise SystemExit\n")
            inserted = True
        out.append(line)
    return "\n".join(out) + "\n"


def exec_s3_with_params(script_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the S3 script in-process with patched params.
    Returns the globals dict after execution (until SystemExit).
    """
    code_text = script_path.read_text(encoding="utf-8")
    code_text = patch_assignments(code_text, params)
    code_text = inject_ensemble_exit_before_plotting(code_text)

    g: Dict[str, Any] = {
        "__file__": str(script_path),
        "__name__": "__main__",
    }
    try:
        exec(compile(code_text, str(script_path), "exec"), g, g)
    except SystemExit:
        pass
    return g


# =========================
# Extraction from S3 globals
# =========================
@dataclass
class S3Series:
    county_flow_share: np.ndarray
    primary_flow_share: np.ndarray  # secondary+township

    county_capacity: np.ndarray     # bed days
    primary_capacity: np.ndarray    # bed days

    county_capacity_share: np.ndarray
    primary_capacity_share: np.ndarray

    insurance_total: np.ndarray
    insurance_county: np.ndarray
    insurance_primary: np.ndarray   # secondary+township sum

    insurance_county_share: np.ndarray
    insurance_primary_share: np.ndarray

def _as_float_array(x):
    if x is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return np.array([], dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1,)
    return arr


def extract_s3_series(g: Dict[str, Any]) -> S3Series:
    """
    Expected variables present in S3 legacy script (based on your S3_10_17_02_20.py):
      - combined_primary_flow_share (list)
      - capacity_trends (dict): total/county/secondary/township + *_share
      - insurance_spending (dict): total/county/secondary/township/primary + *_share

    If any of these are missing, we raise a clear error.
    """
    if "combined_primary_flow_share" not in g:
        raise KeyError("S3 script did not produce 'combined_primary_flow_share'.")
    primary_flow = _as_float_array(g["combined_primary_flow_share"])
    county_flow = 1.0 - primary_flow

    if "capacity_trends" not in g:
        raise KeyError("S3 script did not produce 'capacity_trends'.")
    cap = g["capacity_trends"]
    # bed days
    cap_county = _as_float_array(cap.get("county"))
    cap_secondary = _as_float_array(cap.get("secondary"))
    cap_township = _as_float_array(cap.get("township"))
    cap_primary = cap_secondary + cap_township

    cap_county_share = _as_float_array(cap.get("county_share"))
    cap_secondary_share = _as_float_array(cap.get("secondary_share"))
    cap_township_share = _as_float_array(cap.get("township_share"))
    cap_primary_share = cap_secondary_share + cap_township_share

    if "insurance_spending" not in g:
        raise KeyError("S3 script did not produce 'insurance_spending'.")
    ins = g["insurance_spending"]

    ins_total = _as_float_array(ins.get("total"))
    ins_county = _as_float_array(ins.get("county"))
    # In S3 legacy script, 'primary' is already (secondary+township)
    ins_primary = _as_float_array(ins.get("primary"))
    if ins_primary.size == 0 or np.all(np.isnan(ins_primary)):
        sec = _as_float_array(ins.get("secondary"))
        town = _as_float_array(ins.get("township"))
        if sec.size and town.size and sec.size == town.size:
            ins_primary = sec + town
        else:
            ins_primary = np.array([], dtype=float)

    ins_county_share = _as_float_array(ins.get("county_share"))
    ins_primary_share = _as_float_array(ins.get("primary_share"))
    if ins_primary_share.size == 0:
        ins_primary_share = _as_float_array(ins.get("secondary_share")) + _as_float_array(ins.get("township_share"))

    return S3Series(
        county_flow_share=county_flow,
        primary_flow_share=primary_flow,
        county_capacity=cap_county,
        primary_capacity=cap_primary,
        county_capacity_share=cap_county_share,
        primary_capacity_share=cap_primary_share,
        insurance_total=ins_total,
        insurance_county=ins_county,
        insurance_primary=ins_primary,
        insurance_county_share=ins_county_share,
        insurance_primary_share=ins_primary_share,
    )


# =========================
# Stats helpers
# =========================
def safe_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    if y.size < window:
        return y.copy()
    # Simple centered MA with edge padding (same length)
    w = int(window)
    kernel = np.ones(w) / w
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    sm = np.convolve(ypad, kernel, mode="valid")
    return sm[: y.size]


def quantiles_over_runs(runs: List[np.ndarray], qs: Tuple[float, ...]) -> np.ndarray:
    """
    runs: list of arrays with same length T
    returns: array shape (len(qs), T)
    """
    M = np.stack(runs, axis=0)  # (n, T)
    return np.quantile(M, q=qs, axis=0)


def shrink_band_around_center(q_lo: np.ndarray, q_hi: np.ndarray, center: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visual-only shrink: move lo/hi toward center by 'factor' (0..1).
      factor=1 -> no shrink.
      factor=0 -> collapse to center.
    """
    factor = float(factor)
    factor = max(0.0, min(1.0, factor))
    lo = center - (center - q_lo) * factor
    hi = center + (q_hi - center) * factor
    return lo, hi


# =========================
# Plot helpers
# =========================
def _setup_ax(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_two_series_with_bands(
    X: np.ndarray,
    county_runs: List[np.ndarray],
    primary_runs: List[np.ndarray],
    title: str,
    ylabel: str,
    out_path: Path,
    movavg: int,
    dark_q: Tuple[float, float],
    light_q: Tuple[float, float],
    visual_shrink: float,
    legend_labels: Tuple[str, str],
):
    # Quantiles for county
    q_levels = (light_q[0], dark_q[0], 0.50, dark_q[1], light_q[1])
    qc = quantiles_over_runs(county_runs, q_levels)
    qp = quantiles_over_runs(primary_runs, q_levels)

    # unpack
    c_lo_light, c_lo_dark, c_mid, c_hi_dark, c_hi_light = qc
    p_lo_light, p_lo_dark, p_mid, p_hi_dark, p_hi_light = qp

    # Optional: visual shrink
    if visual_shrink != 1.0:
        c_lo_light, c_hi_light = shrink_band_around_center(c_lo_light, c_hi_light, c_mid, visual_shrink)
        c_lo_dark,  c_hi_dark  = shrink_band_around_center(c_lo_dark,  c_hi_dark,  c_mid, visual_shrink)
        p_lo_light, p_hi_light = shrink_band_around_center(p_lo_light, p_hi_light, p_mid, visual_shrink)
        p_lo_dark,  p_hi_dark  = shrink_band_around_center(p_lo_dark,  p_hi_dark,  p_mid, visual_shrink)

    # Center line is MA of q50 (median)
    c_center = safe_moving_average(c_mid, movavg)
    p_center = safe_moving_average(p_mid, movavg)

    fig, ax = _setup_ax(title, "Time step", ylabel)

    # Light bands
    ax.fill_between(X, c_lo_light, c_hi_light, alpha=LIGHT_ALPHA, color=COUNTY_COLOR, linewidth=0)
    ax.fill_between(X, p_lo_light, p_hi_light, alpha=LIGHT_ALPHA, color=PRIMARY_COLOR, linewidth=0)

    # Dark bands
    ax.fill_between(X, c_lo_dark, c_hi_dark, alpha=DARK_ALPHA, color=COUNTY_COLOR, linewidth=0)
    ax.fill_between(X, p_lo_dark, p_hi_dark, alpha=DARK_ALPHA, color=PRIMARY_COLOR, linewidth=0)

    # Median (raw) as dashed
    ax.plot(X, c_mid, linestyle="--", linewidth=1.2, alpha=RAW_ALPHA, color=COUNTY_COLOR, label=f"{legend_labels[0]} (raw q50)")
    ax.plot(X, p_mid, linestyle="--", linewidth=1.2, alpha=RAW_ALPHA, color=PRIMARY_COLOR, label=f"{legend_labels[1]} (raw q50)")

    # Center line (MA of q50)
    ax.plot(X, c_center, linestyle="-", linewidth=2.8, color=COUNTY_COLOR, label=f"{legend_labels[0]} ({movavg}w MA of q50)")
    ax.plot(X, p_center, linestyle="-", linewidth=2.8, color=PRIMARY_COLOR, label=f"{legend_labels[1]} ({movavg}w MA of q50)")

    ax.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stack_median(
    X: np.ndarray,
    county_share_runs: List[np.ndarray],
    primary_share_runs: List[np.ndarray],
    title: str,
    ylabel: str,
    out_path: Path,
):
    # Median only
    county_q50 = quantiles_over_runs(county_share_runs, (0.5,))[0]
    primary_q50 = quantiles_over_runs(primary_share_runs, (0.5,))[0]

    fig, ax = _setup_ax(title, "Time step", ylabel)
    ax.stackplot(
        X,
        county_q50,
        primary_q50,
        labels=["County (median)", "Primary (Secondary+Township) (median)"],
        alpha=0.55,
    )
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_insurance_dual_axis_with_bands(
    X: np.ndarray,
    total_runs: List[np.ndarray],
    county_runs: List[np.ndarray],
    primary_runs: List[np.ndarray],
    title: str,
    out_path: Path,
    movavg: int,
    dark_q: Tuple[float, float],
    light_q: Tuple[float, float],
    visual_shrink: float,
):
    """
    Two y-axes:
      left: Total insurance spending
      right: County + Primary insurance spending
    All series shown as q50 (raw dashed) + MA(q50) solid, with bands.
    """
    q_levels = (light_q[0], dark_q[0], 0.50, dark_q[1], light_q[1])
    qt = quantiles_over_runs(total_runs, q_levels)
    qc = quantiles_over_runs(county_runs, q_levels)
    qp = quantiles_over_runs(primary_runs, q_levels)

    t_lo_light, t_lo_dark, t_mid, t_hi_dark, t_hi_light = qt
    c_lo_light, c_lo_dark, c_mid, c_hi_dark, c_hi_light = qc
    p_lo_light, p_lo_dark, p_mid, p_hi_dark, p_hi_light = qp

    if visual_shrink != 1.0:
        t_lo_light, t_hi_light = shrink_band_around_center(t_lo_light, t_hi_light, t_mid, visual_shrink)
        t_lo_dark,  t_hi_dark  = shrink_band_around_center(t_lo_dark,  t_hi_dark,  t_mid, visual_shrink)
        c_lo_light, c_hi_light = shrink_band_around_center(c_lo_light, c_hi_light, c_mid, visual_shrink)
        c_lo_dark,  c_hi_dark  = shrink_band_around_center(c_lo_dark,  c_hi_dark,  c_mid, visual_shrink)
        p_lo_light, p_hi_light = shrink_band_around_center(p_lo_light, p_hi_light, p_mid, visual_shrink)
        p_lo_dark,  p_hi_dark  = shrink_band_around_center(p_lo_dark,  p_hi_dark,  p_mid, visual_shrink)

    t_center = safe_moving_average(t_mid, movavg)
    c_center = safe_moving_average(c_mid, movavg)
    p_center = safe_moving_average(p_mid, movavg)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(title, fontsize=18, pad=14)
    ax1.set_xlabel("Time step", fontsize=12)
    ax1.set_ylabel("Total insurance spending", fontsize=12)
    ax1.grid(True, alpha=0.25)

    # Total on left axis (red-ish)
    ax1.fill_between(X, t_lo_light, t_hi_light, alpha=LIGHT_ALPHA, color=PRIMARY_COLOR, linewidth=0)
    ax1.fill_between(X, t_lo_dark,  t_hi_dark,  alpha=DARK_ALPHA, color=PRIMARY_COLOR, linewidth=0)
    ax1.plot(X, t_mid,   linestyle="--", linewidth=1.2, alpha=RAW_ALPHA, color=PRIMARY_COLOR, label="Total (raw q50)")
    ax1.plot(X, t_center, linestyle="-", linewidth=2.8, color=PRIMARY_COLOR, label=f"Total ({movavg}w MA of q50)")

    # County/Primary on right axis (blue/red)
    ax2 = ax1.twinx()
    ax2.set_ylabel("County / Primary insurance spending", fontsize=12)

    ax2.fill_between(X, c_lo_light, c_hi_light, alpha=LIGHT_ALPHA, color=COUNTY_COLOR, linewidth=0)
    ax2.fill_between(X, c_lo_dark,  c_hi_dark,  alpha=DARK_ALPHA, color=COUNTY_COLOR, linewidth=0)
    ax2.plot(X, c_mid,   linestyle="--", linewidth=1.2, alpha=RAW_ALPHA, color=COUNTY_COLOR, label="County (raw q50)")
    ax2.plot(X, c_center, linestyle="-", linewidth=2.8, color=COUNTY_COLOR, label=f"County ({movavg}w MA of q50)")

    ax2.fill_between(X, p_lo_light, p_hi_light, alpha=LIGHT_ALPHA, color=PRIMARY_COLOR, linewidth=0)
    ax2.fill_between(X, p_lo_dark,  p_hi_dark,  alpha=DARK_ALPHA, color=PRIMARY_COLOR, linewidth=0)
    ax2.plot(X, p_mid,   linestyle="--", linewidth=1.2, alpha=RAW_ALPHA, color=PRIMARY_COLOR, label="Primary (raw q50)")
    ax2.plot(X, p_center, linestyle="-", linewidth=2.8, color=PRIMARY_COLOR, label=f"Primary ({movavg}w MA of q50)")

    # Single legend combining both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", type=str, default=str(S3_SCRIPT_PATH), help="Path to legacy S3 script")
    ap.add_argument("--topk", type=str, default=str(TOPK_PARAMS_PATH), help="Path to topk_params.json")
    ap.add_argument("--out", type=str, default=str(OUT_DIR), help="Output directory")
    ap.add_argument("--max_runs", type=int, default=-1, help="Use first N runs (-1 = all)")
    ap.add_argument("--movavg", type=int, default=MOVAVG_W, help="Moving average window (weeks)")
    ap.add_argument("--dark_q", type=str, default=f"{DARK_Q[0]},{DARK_Q[1]}", help="dark band quantiles, e.g. 0.4,0.6")
    ap.add_argument("--light_q", type=str, default=f"{LIGHT_Q[0]},{LIGHT_Q[1]}", help="light band quantiles, e.g. 0.25,0.75")
    ap.add_argument("--visual_shrink", type=float, default=VISUAL_SHRINK_BAND, help="visual-only band shrink factor (0..1)")
    args = ap.parse_args()

    s3_path = Path(args.s3).resolve()
    topk_path = Path(args.topk).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dark_q = tuple(float(x) for x in args.dark_q.split(","))
    light_q = tuple(float(x) for x in args.light_q.split(","))

    params_list = json.loads(topk_path.read_text(encoding="utf-8"))
    if not isinstance(params_list, list) or not params_list:
        raise ValueError("topk_params.json must be a non-empty list of param dicts.")

    if args.max_runs and args.max_runs > 0:
        params_list = params_list[: args.max_runs]
    elif MAX_RUNS is not None:
        params_list = params_list[: MAX_RUNS]

    # Collect runs
    flow_county_runs: List[np.ndarray] = []
    flow_primary_runs: List[np.ndarray] = []

    cap_county_runs: List[np.ndarray] = []
    cap_primary_runs: List[np.ndarray] = []

    cap_share_county_runs: List[np.ndarray] = []
    cap_share_primary_runs: List[np.ndarray] = []

    ins_total_runs: List[np.ndarray] = []
    ins_county_runs: List[np.ndarray] = []
    ins_primary_runs: List[np.ndarray] = []

    ins_share_county_runs: List[np.ndarray] = []
    ins_share_primary_runs: List[np.ndarray] = []

    n_ok = 0
    for idx, params in enumerate(params_list, start=1):
        try:
            g = exec_s3_with_params(s3_path, params)
            series = extract_s3_series(g)

            flow_county_runs.append(series.county_flow_share)
            flow_primary_runs.append(series.primary_flow_share)

            cap_county_runs.append(series.county_capacity)
            cap_primary_runs.append(series.primary_capacity)

            cap_share_county_runs.append(series.county_capacity_share)
            cap_share_primary_runs.append(series.primary_capacity_share)

            ins_total_runs.append(series.insurance_total)
            ins_county_runs.append(series.insurance_county)
            ins_primary_runs.append(series.insurance_primary)

            ins_share_county_runs.append(series.insurance_county_share)
            ins_share_primary_runs.append(series.insurance_primary_share)

            n_ok += 1
            print(f"[{idx}/{len(params_list)}] OK")
        except Exception as e:
            print(f"[{idx}/{len(params_list)}] FAIL: {e}")

    if n_ok == 0:
        raise RuntimeError("No successful S3 runs; cannot plot.")

    # Use T from first successful run
    T = int(flow_primary_runs[0].shape[0])
    X = np.arange(T)

    # ---- 1) Flow share with bands
    plot_two_series_with_bands(
        X=X,
        county_runs=flow_county_runs,
        primary_runs=flow_primary_runs,
        title=f"S3 — Flow share with uncertainty bands (n={n_ok} plausible sets from S1)\n"
              f"Band: {int(dark_q[0]*100)}–{int(dark_q[1]*100)}% (dark) + {int(light_q[0]*100)}–{int(light_q[1]*100)}% (light); "
              f"center line: MA of q50",
        ylabel="Share",
        out_path=out_dir / "s3_flow_share_ui_band.png",
        movavg=args.movavg,
        dark_q=dark_q,
        light_q=light_q,
        visual_shrink=args.visual_shrink,
        legend_labels=("County Hospital", "Primary Care (Secondary+Township)"),
    )

    # ---- 2) Capacity (bed days) with bands
    plot_two_series_with_bands(
        X=X,
        county_runs=cap_county_runs,
        primary_runs=cap_primary_runs,
        title=f"S3 — Capacity (Bed Days) with uncertainty bands (n={n_ok} plausible sets from S1)\n"
              f"Band: {int(dark_q[0]*100)}–{int(dark_q[1]*100)}% (dark) + {int(light_q[0]*100)}–{int(light_q[1]*100)}% (light); "
              f"center line: MA of q50",
        ylabel="Total capacity (bed days)",
        out_path=out_dir / "s3_capacity_bed_days_ui_band.png",
        movavg=args.movavg,
        dark_q=dark_q,
        light_q=light_q,
        visual_shrink=args.visual_shrink,
        legend_labels=("County capacity (bed days)", "Primary capacity (bed days)"),
    )

    # ---- 3) Capacity share stack (median only)
    plot_stack_median(
        X=X,
        county_share_runs=cap_share_county_runs,
        primary_share_runs=cap_share_primary_runs,
        title=f"S3 — Capacity share (median stack, n={n_ok})",
        ylabel="Share",
        out_path=out_dir / "s3_capacity_share_stack_median.png",
    )

    # ---- 4) Insurance amounts dual-axis with bands
    plot_insurance_dual_axis_with_bands(
        X=X,
        total_runs=ins_total_runs,
        county_runs=ins_county_runs,
        primary_runs=ins_primary_runs,
        title=f"S3 — Insurance spending with uncertainty bands (n={n_ok} plausible sets from S1)\n"
              f"Band: {int(dark_q[0]*100)}–{int(dark_q[1]*100)}% (dark) + {int(light_q[0]*100)}–{int(light_q[1]*100)}% (light); "
              f"center line: MA of q50",
        out_path=out_dir / "s3_insurance_dual_axis_ui_band.png",
        movavg=args.movavg,
        dark_q=dark_q,
        light_q=light_q,
        visual_shrink=args.visual_shrink,
    )

    # ---- 5) Insurance share stack (median only)
    plot_stack_median(
        X=X,
        county_share_runs=ins_share_county_runs,
        primary_share_runs=ins_share_primary_runs,
        title=f"S3 — Insurance share (median stack, n={n_ok})",
        ylabel="Share",
        out_path=out_dir / "s3_insurance_share_stack_median.png",
    )

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
