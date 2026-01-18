# plot_severity_boxplots_combined_from_stats.py
# 用“箱线图统计量(bxp)”把 S1/S2/S3 三张 severity boxplot 合到一张图，并上色
# 注意：这里的数值是根据你截图“目测近似”的；如需更精确，手动微调下面的 stats 即可。

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# 1) 在这里改数值（核心）
# =========================
# 每个 box 需要：q1, med, q3, whislo, whishi, fliers(list)
# 你可以把 fliers 列表删空 []，就不画离群点

STATS = {
    "S1": {
        "County Hospital": dict(
            q1=0.52, med=0.59, q3=0.67, whislo=0.25, whishi=0.93,
            fliers=[0.12, 0.15, 0.18, 0.20, 0.22, 0.24, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        ),
        "Primary Care": dict(
            q1=0.23, med=0.38, q3=0.41, whislo=0.00, whishi=0.55,
            fliers=[]
        ),
    },
    "S2": {
        "County Hospital": dict(
            q1=0.79, med=0.82, q3=0.85, whislo=0.67, whishi=0.99,
            fliers=[0.36, 0.38, 0.40, 0.42, 0.45, 0.50, 0.98, 0.99, 1.00]
        ),
        "Primary Care": dict(
            q1=0.34, med=0.42, q3=0.48, whislo=0.10, whishi=0.71,
            fliers=[0.00, 0.02, 0.03, 0.05, 0.07, 0.75, 0.78, 0.80, 0.85, 0.88]
        ),
    },
    "S3": {
        "County Hospital": dict(
            q1=0.50, med=0.58, q3=0.66, whislo=0.25, whishi=0.92,
            fliers=[0.15, 0.18, 0.20, 0.22, 0.24, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
        ),
        "Primary Care Network": dict(
            q1=0.25, med=0.34, q3=0.39, whislo=0.05, whishi=0.60,
            fliers=[0.00, 0.01, 0.02, 0.03]
        ),
    },
}

# 颜色：你可以直接改这里（让它更像你原来的配色）
COLOR_COUNTY = "#1f77b4"   # 蓝
COLOR_PRIMARY = "#ff7f0e"  # 橙

# 输出文件名
OUT_PNG = "severity_boxplots_S1S2S3_combined.png"

# =========================
# 2) 生成 bxp 所需结构
# =========================
def to_bxp_stats(label: str, d: dict):
    return {
        "label": label,
        "q1": d["q1"],
        "med": d["med"],
        "q3": d["q3"],
        "whislo": d["whislo"],
        "whishi": d["whishi"],
        "fliers": d.get("fliers", []),
    }

# 盒子顺序：S1(County,Primary) -> S2(County,Primary) -> S3(County,Primary)
scenario_order = ["S1", "S2", "S3"]
# 每个场景下的“两个类别”名字（S3 的 primary 名字不同，所以单独处理）
pair_labels = {
    "S1": ("County Hospital", "Primary Care"),
    "S2": ("County Hospital", "Primary Care"),
    "S3": ("County Hospital", "Primary Care Network"),
}

stats_all = []
positions = []

# 每个场景一个中心 x：1,2,3；两盒偏移
offset = 0.18
for i, sc in enumerate(scenario_order, start=1):
    a, b = pair_labels[sc]
    stats_all.append(to_bxp_stats(f"{sc}-{a}", STATS[sc][a]))
    positions.append(i - offset)
    stats_all.append(to_bxp_stats(f"{sc}-{b}", STATS[sc][b]))
    positions.append(i + offset)

# =========================
# 3) 画图
# =========================
fig, ax = plt.subplots(figsize=(10.5, 4.8))

bp = ax.bxp(
    stats_all,
    positions=positions,
    widths=0.18,
    patch_artist=True,
    showfliers=True,
    manage_ticks=False
)

# 上色：偶数 index(0,2,4) 视为 County；奇数视为 Primary
for idx, box in enumerate(bp["boxes"]):
    if idx % 2 == 0:
        box.set_facecolor(COLOR_COUNTY)
        box.set_alpha(0.45)
    else:
        box.set_facecolor(COLOR_PRIMARY)
        box.set_alpha(0.45)
    box.set_edgecolor("#2b2b2b")
    box.set_linewidth(1.2)

# whiskers/caps/medians/fliers 统一风格
for w in bp["whiskers"]:
    w.set_color("#2b2b2b")
    w.set_linewidth(1.1)
for c in bp["caps"]:
    c.set_color("#2b2b2b")
    c.set_linewidth(1.1)
for m in bp["medians"]:
    m.set_color("#2b2b2b")   # 你也可以改成白色/更粗
    m.set_linewidth(1.8)
for f in bp["fliers"]:
    f.set_marker("o")
    f.set_markersize(3.2)
    f.set_markerfacecolor("none")
    f.set_markeredgecolor("#222222")
    f.set_alpha(0.85)

# x 轴：只显示 S1/S2/S3
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["S1", "S2", "S3"])
ax.set_xlim(0.5, 3.5)

ax.set_ylabel("Severity Score")
ax.set_ylim(-0.02, 1.02)
ax.grid(axis="y", alpha=0.25)

ax.set_title("Severity distribution across scenarios (boxplots)")

# legend
handles = [
    Patch(facecolor=COLOR_COUNTY, edgecolor="#2b2b2b", alpha=0.45, label="County Hospital"),
    Patch(facecolor=COLOR_PRIMARY, edgecolor="#2b2b2b", alpha=0.45, label="Primary level"),
]
ax.legend(handles=handles, loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"[OK] saved: {OUT_PNG}")
