import os, sys, re, time, types, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 改成你的 S2 脚本路径（你也可以换成 S1 或 S3）
SCRIPT_PATH = r"D:\Job\PKUHSC\Submission\ABG\nature\python_project\ihc_abm_param_ensemble_v2\legacy\S2_10_15_17_7_nol2.py"

def build_code_with_plot_stub(seed=0, params=None):
    """
    关键点：
    - 不改模型机制
    - 但把 legacy 脚本里“画很多图”的 pyplot 替换成轻量 stub（否则跑ensemble会非常慢）
    """
    base = pathlib.Path(SCRIPT_PATH).read_text(encoding="utf-8", errors="ignore")
    params = params or {}

    def replace_assignment(txt, name, val):
        pattern = rf"^{name}\s*=\s*([0-9.]+)\s*$"
        return re.sub(pattern, f"{name} = {val}", txt, flags=re.MULTILINE)

    for k, v in params.items():
        base = replace_assignment(base, k, v)

    header = f"""
import sys, types
plt = types.ModuleType("matplotlib.pyplot")
def _noop(*args, **kwargs): return None
class _Dummy:
    def __getattr__(self, name): return _noop
def _subplots(*args, **kwargs): return _Dummy(), _Dummy()
plt.subplots = _subplots
plt.figure = lambda *a, **k: _Dummy()
plt.gca = lambda *a, **k: _Dummy()
plt.plot = plt.scatter = plt.imshow = plt.bar = plt.hist = _noop
plt.fill_between = plt.legend = plt.title = plt.xlabel = plt.ylabel = _noop
plt.xticks = plt.yticks = plt.grid = plt.tight_layout = plt.subplots_adjust = _noop
plt.savefig = plt.close = plt.show = _noop
def __getattr__(name): return _noop
plt.__getattr__ = __getattr__
sys.modules["matplotlib.pyplot"] = plt

import random, numpy as np
random.seed({seed})
np.random.seed({seed})
"""
    return header + "\n" + base

def run_once(params, seed, module_name):
    code = build_code_with_plot_stub(seed=seed, params=params)
    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod
    exec(code, mod.__dict__)

    # 你要画哪个指标，就从 mod 里取哪个 time series
    # 这里演示：县医院就诊占比（随时间）
    y = np.array(mod.flow_share_lvl["county"], dtype=float)
    return y

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(SCRIPT_PATH), "..", "outputs_uncertainty")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_params = {"ALPHA_Q_UPD": 0.25, "TAU_A_START": 0.30, "TAU_A_END": 0.05}

    runs_specs = []
    for seed in range(1, 21):  # 20次重复
        runs_specs.append((base_params, seed, f"seed_{seed}"))

    t0 = time.time()
    Y = []
    for params, seed, name in runs_specs:
        Y.append(run_once(params, seed, name))
    print("Finished runs. seconds =", round(time.time() - t0, 2))

    Y = np.vstack(Y)
    x = np.arange(Y.shape[1])
    y_med = np.median(Y, axis=0)
    q05 = np.quantile(Y, 0.05, axis=0)
    q95 = np.quantile(Y, 0.95, axis=0)
    q25 = np.quantile(Y, 0.25, axis=0)
    q75 = np.quantile(Y, 0.75, axis=0)

    # 存数据
    csv_path = os.path.join(out_dir, "county_flow_share_band_demo.csv")
    np.savetxt(csv_path, np.column_stack([x, y_med, q05, q95, q25, q75]),
               delimiter=",",
               header="t,median,q05,q95,q25,q75",
               comments="")

    print("Saved CSV:", csv_path)

    plt.figure(figsize=(10, 5))
    plt.fill_between(x, q05, q95, alpha=0.20, label="90% UI (5–95%)")
    plt.fill_between(x, q25, q75, alpha=0.35, label="IQR (25–75%)")
    plt.plot(x, y_med, linewidth=2, label="Median")
    plt.xlabel("Time (t)")
    plt.ylabel("County visit share")
    plt.title("Uncertainty band (seed uncertainty)")
    plt.legend()
    png_path = os.path.join(out_dir, "uncertainty_band_demo.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved PNG:", png_path)
