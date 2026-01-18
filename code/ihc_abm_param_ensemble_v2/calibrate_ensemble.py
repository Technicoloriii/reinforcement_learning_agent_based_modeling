
"""
Small ensemble calibration for scenario S1.

Idea:
- Sample N candidate parameter sets (ALPHA_Q_UPD, TAU_A_START, TAU_A_END)
  within ranges defined in search_space.json.
- For each candidate, run a patched version of S1 (text-level replacement),
  compute summary metrics, and compute a distance score vs. targets.
- Keep top-K parameter sets.

Usage:
    python calibrate_ensemble.py --n-candidates 20 --k-select 5 --eval-window 52 --out outputs

If --targets is not provided, we will:
    1) Run baseline S1 once, compute its summary.
    2) Use that as center of an auto-target file (auto_targets.json).

Outputs:
    - calibration/auto_targets.json (if auto-generated)
    - calibration/selected_params_<timestamp>.jsonl
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
from copy import deepcopy

from summary_utils import summarize_module

LEGACY_S1_MODULE = "legacy.S1_10_15_17_7_nol2"
LEGACY_S1_PATH = os.path.join(os.path.dirname(__file__), "legacy", "S1_10_15_17_7_nol2.py")
SEARCH_SPACE_PATH = os.path.join(os.path.dirname(__file__), "search_space.json")

# ---- helpers for dynamic loading ----

def load_module_from_text(module_name, code_text):
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(code_text, module.__dict__)
    return module

def load_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"Cannot find module {module_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# ---- targets / distance ----

def auto_build_targets(eval_window, out_dir):
    """
    Run baseline S1 once and use its summary as target center.
    """
    print("Auto-building targets from baseline S1 ...")
    base_mod = load_module(LEGACY_S1_MODULE)
    summary = summarize_module(base_mod, eval_window=eval_window)
    # Wrap as simple "center-only" targets
    targets = {k: {"center": v} for k, v in summary.items()}
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, "auto_targets.json")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    print(f"Auto targets written to {target_path}")
    return targets

def load_targets(path, eval_window, calib_dir):
    if path is None:
        return auto_build_targets(eval_window, calib_dir)
    with open(path, "r", encoding="utf-8") as f:
        targets = json.load(f)
    return targets

def load_search_space():
    if os.path.exists(SEARCH_SPACE_PATH):
        with open(SEARCH_SPACE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # default ranges
    return {
        "ALPHA_Q_UPD": [0.15, 0.35],
        "TAU_A_START": [0.2, 0.5],
        "TAU_A_END": [0.02, 0.15],
    }

def sample_params(search_space):
    return {
        "ALPHA_Q_UPD": random.uniform(*search_space["ALPHA_Q_UPD"]),
        "TAU_A_START": random.uniform(*search_space["TAU_A_START"]),
        "TAU_A_END": random.uniform(*search_space["TAU_A_END"]),
    }

def patch_code(code_text, params):
    # simple regex-free string replacement based on full assignment lines
    # we assume the originals are exactly:
    #   ALPHA_Q_UPD = 0.25
    #   TAU_A_START = 0.3
    #   TAU_A_END = 0.05
    lines = code_text.splitlines()
    for i, line in enumerate(lines):
        if "ALPHA_Q_UPD" in line and "=" in line:
            lines[i] = f"ALPHA_Q_UPD = {params['ALPHA_Q_UPD']:.5f}"
        elif "TAU_A_START" in line and "=" in line:
            lines[i] = f"TAU_A_START = {params['TAU_A_START']:.5f}"
        elif "TAU_A_END" in line and "=" in line:
            lines[i] = f"TAU_A_END = {params['TAU_A_END']:.5f}"
    return "\n".join(lines)

def distance(summary, targets):
    """
    Simple squared-distance over overlapping keys.
    If target is {"center": x, "weight": w}, use weight; default w=1.
    """
    d = 0.0
    count = 0
    for k, tconf in targets.items():
        if k not in summary:
            continue
        center = tconf.get("center", None)
        if center is None:
            continue
        w = tconf.get("weight", 1.0)
        v = summary[k]
        if v is None:
            continue
        diff = (v - center)
        d += w * diff * diff
        count += 1
    if count == 0:
        return float("inf")
    return d / count

# ---- main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--k-select", type=int, default=5)
    parser.add_argument("--eval-window", type=int, default=52)
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON file with targets; if omitted, auto-build from baseline.")
    parser.add_argument("--out", type=str, default="outputs",
                        help="Base output folder.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.append(base_dir)

    calib_dir = os.path.join(args.out, "calibration_S1")
    os.makedirs(calib_dir, exist_ok=True)

    targets = load_targets(args.targets, args.eval_window, calib_dir)
    search_space = load_search_space()

    with open(LEGACY_S1_PATH, "r", encoding="utf-8") as f:
        base_code = f.read()

    results = []
    for i in range(args.n_candidates):
        print(f"\n=== Candidate {i+1}/{args.n_candidates} ===")
        params = sample_params(search_space)
        patched = patch_code(base_code, params)
        # use a unique module name each time
        module_name = f"legacy_S1_patched_{i}"
        try:
            mod = load_module_from_text(module_name, patched)
            summary = summarize_module(mod, eval_window=args.eval_window)
            score = distance(summary, targets)
            print("params:", params)
            print("summary:", json.dumps(summary, ensure_ascii=False))
            print("score:", score)
            results.append({"params": params, "summary": summary, "score": score})
        except Exception as e:
            print("Error running candidate:", e)
            continue

    # sort by score and keep top-K
    results_sorted = sorted(results, key=lambda x: x["score"])
    top_k = results_sorted[: args.k_select]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(calib_dir, f"selected_params_{timestamp}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in top_k:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTop-{args.k_select} parameter sets written to {out_jsonl}")

if __name__ == "__main__":
    main()
