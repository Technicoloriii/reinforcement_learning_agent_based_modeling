
ihc_abm_param_ensemble_v2
=========================

目录结构
--------
- legacy/
    - S1_10_15_17_7_nol2.py
    - S2_10_15_17_7_nol2.py
    - S3_10_17_02_20.py
  （你的三份原始场景脚本，保持原样）
- summary_utils.py
  （从场景脚本的全局变量里提取“就诊占比、医保支出、转诊率、严重程度、床位”等指标的小工具）
- run_single.py
  （跑单个场景并输出 summary.json）
- calibrate_ensemble.py
  （对 S1 做一个“小规模参数集筛选”的示例）
- search_space.json
  （参数抽样范围）

基本用法
--------

1. 跑单个场景（不改任何参数，只提取 summary）

    ```bash
    cd ihc_abm_param_ensemble_v2

    # 跑 S2，取最后 52 期做平均
    python run_single.py --scenario S2 --eval-window 52 --out outputs
    ```

    运行结束后会得到：

    - `outputs/S2_YYYYMMDD_HHMMSS/summary.json`

    其中字段包括：
    - `flow_share_county_mean`
    - `flow_share_primary_mean`
    - `insurance_total_mean`
    - `insurance_county_share_mean`
    - `insurance_primary_share_mean`
    - `referral_success_rate_mean`
    - `severity_county_mean`
    - `severity_primary_mean`
    - `capacity_county_last` / `capacity_primary_last`
    - `capacity_county_mean` / `capacity_primary_mean`
    等（如果原脚本里有对应数据）。

2. 对 S1 做“小规模参数集筛选”（示例）

    先看一下 `search_space.json`，里面给了三个参数的范围：

    - `ALPHA_Q_UPD`
    - `TAU_A_START`
    - `TAU_A_END`

    然后执行：

    ```bash
    cd ihc_abm_param_ensemble_v2
    
    # 例如：随机 20 套候选参数，选出最接近目标的 5 套
    python calibrate_ensemble.py --n-candidates 20 --k-select 5 --eval-window 52 --out outputs
    ```

    如果没有指定 `--targets`，脚本会：
    1. 先用原始 S1 跑一次，算出 baseline 的 summary；
    2. 把 baseline summary 写成 `outputs/calibration_S1/auto_targets.json`；
    3. 在 `search_space.json` 的范围里采样参数，依次跑 S1，并根据 summary 与 auto_targets 的距离打分；
    4. 把得分最小的前 `k-select` 套参数写入：
       - `outputs/calibration_S1/selected_params_*.jsonl`

    每一行是一个 JSON，包含：
    - `params`：这一套参数的数值；
    - `summary`：对应的 summary 指标；
    - `score`：与目标的距离。

    注意：这里的示例只是演示“参数集 + summary + 距离函数”的工作流，参数范围和目标你可以根据文章需要再精细化。

注意事项
--------
- 这套代码不会改动 `legacy/` 里的原始脚本逻辑；所有仿真依然按原脚本运行。
- `summary_utils.py` 只是从脚本运行后留下的全局变量里抽取你关心的指标，不会影响仿真过程。
- `calibrate_ensemble.py` 是通过文本替换的方式修改 S1 中 `ALPHA_Q_UPD`、`TAU_A_START`、`TAU_A_END`，然后在内存里执行，不会改写磁盘上的原始脚本。





But that's different, being related to the outcome variable is not equal to being related to the error term. we'd like to see  IV and outcome variable being perfectly uncorrelated, but that's more like an "ideal" condition you'll never get.  thus, in most case, we take one step back, it can be related with the outcome,
