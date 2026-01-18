import subprocess
import sys

cmd = [
    sys.executable, "calibrate_ensemble_pack1_scored.py",
    "--mode", "select",
    "--candidates", "candidates.csv",
    "--targets", "targets_S1.json",
    "--k", "50",
    "--out", "outputs_s1_top20",
    "--plot"
]
subprocess.check_call(cmd)
