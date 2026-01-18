#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_select_topk_hardflow.py

用途：
- 在 PyCharm 里“一键运行”，不用手动敲命令行参数
- 基于已有 candidates.csv + targets_S1.json 重新打分
- 先做硬门槛过滤：flow_share_county_mean 必须在 [low, high]
- 再在剩余集合里选 top-K，输出到 outputs_s1_top20_hardflow/

你只需要：
1) 确保项目根目录下有：
   - candidates.csv
   - targets_S1.json
   - calibrate_ensemble_pack1_scored_hard.py
2) 右键本文件 Run

如需改 K / 输出目录，直接改下面的常量即可。
"""

import subprocess
import sys
from pathlib import Path

K = 50
OUTDIR = "outputs_s1_top20_hardflow"
CANDIDATES = "candidates.csv"
TARGETS = "targets_S1.json"

def main():
    root = Path(__file__).resolve().parent
    script = root / "calibrate_ensemble_pack1_scored_hard.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 {script.name}，请确认已放在项目根目录。")

    cmd = [
        sys.executable, str(script),
        "--mode", "select",
        "--candidates", str(root / CANDIDATES),
        "--targets", str(root / TARGETS),
        "--k", str(K),
        "--out", str(root / OUTDIR),
        "--plot",
        "--hard-target", "flow_share_county_mean",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
