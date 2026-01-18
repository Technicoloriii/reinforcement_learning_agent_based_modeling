
"""
Run a single scenario script (legacy) and export a compact summary JSON.

Usage:
    python run_single.py --scenario S1 --eval-window 52 --out outputs

Scenarios:
    S1 -> legacy/S1_10_15_17_7_nol2.py
    S2 -> legacy/S2_10_15_17_7_nol2.py
    S3 -> legacy/S3_10_17_02_20.py
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

from summary_utils import summarize_module

SCENARIO_MAP = {
    "S1": "legacy.S1_10_15_17_7_nol2",
    "S2": "legacy.S2_10_15_17_7_nol2",
    "S3": "legacy.S3_10_17_02_20",
}

def load_and_run(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"Cannot find module {module_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["S1", "S2", "S3"], required=True)
    parser.add_argument("--eval-window", type=int, default=52,
                        help="Number of final periods to average over.")
    parser.add_argument("--out", type=str, default="outputs",
                        help="Base output folder.")
    args = parser.parse_args()

    module_name = SCENARIO_MAP[args.scenario]

    # Make sure legacy folder is importable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.append(base_dir)

    print(f"Running scenario {args.scenario} using {module_name} ...")
    mod = load_and_run(module_name)
    summary = summarize_module(mod, eval_window=args.eval_window)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{args.scenario}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary written to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
