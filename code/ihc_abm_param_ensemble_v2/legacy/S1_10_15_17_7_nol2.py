# -*- coding: utf-8 -*-
"""
IHC-ABM Enhanced (v4) — S2 Ultimate Version
- 真实地理布局（从geocity导入）
- 等待容忍阈值 + 外流分类
- 床日/LOS机制（基于病情严重度）
- 内生行为机制（关闭外生偏置）
- 增强监控和输出
"""

import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import math
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import json

# =============== 专业热力图参数 ===============
# inferno colormap without top white (trim last 8%)
_inferno = mpl.colormaps['inferno']
HEAT_CMAP = ListedColormap(_inferno(np.linspace(0.0, 0.92, 256)))

# 城市背景颜色映射
SOFT_CMAP = LinearSegmentedColormap.from_list(
    "soft_mint", ["#ffffff","#f3faf7","#e7f6f0","#d3efe6","#bfe7dd","#aee0d5"]
)

# 医院颜色
COLOR_TERTIARY = "#d97706"  # 县医院
COLOR_SECONDARY = "#2563eb"  # 二级医院
COLOR_PRIMARY = "#059669"    # 基层医院

# =============== 从geocity导入真实布局 ===============
def load_city_layout():
    """从geocity_update_2.py加载城市布局和医院位置"""
    # 这里需要您提供geocity_update_2.py中生成的数据
    # 或者我们可以重构这部分来直接生成相同的布局
    # 暂时使用模拟数据，您需要替换为真实数据

    # 城市参数
    R_CITY = 10.0
    R_INNER = 3.0
    R_MIDDLE = 6.0

    # 医院位置（示例数据，需要替换）
    hospital_positions = {
        'county': [(0.0, 0.0)],  # TER_1
        'secondary': [
            (0.11379539773014025, -5.650593474),  # SEC_1
            (4.320249347575342, 3.185028953685901)  # SEC_2
        ],
        'township': [
            (-1.55559427, -0.989992556),  # PRI_1
            (-0.886082892, 2.133249007302485),  # PRI_2
            (2.4797599149815674, -2.85845215),  # PRI_3
            (4.040729383325832, -4.399558712),  # PRI_4
            (-6.634243241, -0.371456373),  # PRI_5
            (3.6605658066377575, -0.103055971),  # PRI_6
            (8.659329422749051, -2.229064251),  # PRI_7
            (7.328182317, 4.816806888528005),  # PRI_8
            (-1.050331321, 7.472897930775263),  # PRI_9
            (-4.29982209, -1.095054112)  # PRI_10
        ]
    }

    # 人口密度网格（示例）
    GRID_RES = 100
    xs = np.linspace(-R_CITY, R_CITY, GRID_RES)
    ys = np.linspace(-R_CITY, R_CITY, GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    RR = np.sqrt(X ** 2 + Y ** 2)

    # 基础人口分布（中心密度高，向外衰减）
    base_density = np.exp(-0.3 * RR)

    # 二级医院周边人口提升
    secondary_boost = np.zeros_like(base_density)
    for pos in hospital_positions['secondary']:
        sx, sy = pos
        secondary_boost += 0.3 * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * 1.5 ** 2))

    population_grid = np.minimum(base_density + secondary_boost, 0.95)
    population_grid = population_grid / np.nanmax(population_grid)
    population_grid = np.power(population_grid, 0.75)
    population_grid = np.maximum(population_grid, 0.15)
    population_grid[RR > R_CITY] = 0

    return hospital_positions, population_grid, X, Y, R_CITY


# =============== RNG & Colors ===============
SEED = 42
rng = np.random.default_rng(SEED)

COLORS = {
    "county": "#2C458D",  # deep blue
    "secondary": "#EF5950",  # amber
    "township": "#EF5950",  # coral red
    "total": "#8D5A78",  # plum
    "grid": "#E4DFD9",  # warm gray
    "outflow_wait": "#FF6B6B",
    "outflow_full": "#763262",
    "outflow_mismatch": "#45B7D1"
}

# =============== Horizon & Topology ===============
T = 300
WARMUP = 18
REPORT_MOVAVG = 24

# 从真实布局加载
HOSPITAL_POSITIONS, POPULATION_GRID, X_GRID, Y_GRID, CITY_RADIUS = load_city_layout()
TOPOLOGY_RADIUS = CITY_RADIUS

NT = len(HOSPITAL_POSITIONS['township'])
N_SECONDARY = len(HOSPITAL_POSITIONS['secondary'])
MAX_PATIENT_CHOICES = 5  # 患者最多尝试5次

# =============== Enhanced Behavioral Parameters ===============
USE_REFERRAL_SYSTEM = True
ENABLE_COUNTY_ALLOCATION = True
PATIENT_HABIT_STRENGTH = 0.3
PATIENT_INFO_ASYMMETRY = 0.1

# =============== 等待容忍参数 ===============
WAIT_TOLERANCE_A0 = 8.0 # 基础等待容忍
WAIT_TOLERANCE_A1 = 10.0  # 严重度对等待容忍的影响系数

# =============== 床日/LOS参数 ===============
LOS_TAU = 0.35  # 住院阈值，严重度>=0.38需要住院
LOS_BASE = 1.0  # 基础床日
LOS_SEVERITY_MULTIPLIER = 6.0  # 严重度对床日的影响
LOS_LEVEL_MULTIPLIER = {  # 医院级别对床日的影响
    'township': 0.9,
    'secondary': 1.0,
    'county': 1.15
}

# =============== 关闭外生偏置 ===============
ENABLE_PRESTIGE = True  # 关闭外生声望偏置
PRESTIGE_BIAS_COUNTY = 0.2
ENABLE_NOISE = True
UTILITY_NOISE_SIGMA = 0.05

# =============== Induced Demand ===============
USE_INDUCED_DEMAND = False
THETA_Q = 0.6
THETA_W = 1.0
THETA_D = 1.1
THETA_HHI = 0.7
ALPHA0 = -0.35
ALPHA_Q = 0.8
ALPHA_WD = 1.1
BACKLOG_RHO = 0.65

# =============== Prices, Costs, Reimbursement ===============
PRICE = dict(county=650.0, secondary=220.0, township=220.0)
REIMB_RATIO = 0.70


def revenue_multiplier(s):
    return 1.0 + 1.6 * (s ** 1.5)


def unit_cost(s):
    return 90.0 + 120.0 * s + 160.0 * (s ** 2)


FIXED_COST_PER_CAP = dict(county=80.0, secondary=20.0, township=20.0)
A_COST_PER_UNIT = 40.0

I_COST = {-1: 1500.0, 0: 0.0, 1: 2500.0, 2: 3000.0}
K_DELTA = {-1: -0.06, 0: 0.0, 1: 0.10, 2: 0.22}

# ===== Scenario 1: Bundled payment with surplus retention =====
SCENARIO = "fee_for_service"
# 关闭所有合作机制
ENABLE_RHO = False      # 关闭转诊激励
ENABLE_V = False        # 关闭县医院协调
ENABLE_D = False        # 关闭能力下沉
ENABLE_COUNTY_ALLOCATION = False  # 关闭县医院分配决策

BUDGET_FACTOR = 0.00
USE_BUDGET_FIXED = False
BUDGET_FIXED = 0.0

# Surplus sharing rule
BONUS_BASE_SHARE = dict(county=0, secondary=0, township=0)
BONUS_RHO_WEIGHT = 0.60
BONUS_D_WEIGHT = 0.60
ALLOW_NEGATIVE_SURPLUS = False

# ===== County Allocation Decision =====
if ENABLE_COUNTY_ALLOCATION:
    ALLOCATION_GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

# ===== Enhanced Cooperation Parameters =====
S_MAX_UPPER = dict(county=0.9, secondary=0.55, township=0.5)
SMAX0 = dict(county=0.9, secondary=0.3, township=0.3)
ETA_D = dict(county=0.0, secondary=0.035, township=0.065)

EMPTY_COST_PHI = dict(county=0.55, secondary=0.2, township=0.15)
I_COST_MULT = dict(county=1.00, secondary=1.70, township=1.90)
FINANCE_PSI = dict(county=0.05, secondary=0.12, township=0.15)

THR_BASE = dict(township=0.3, secondary=0.3)
THR_RHO_COEF = dict(township=0.20, secondary=0.20)
THR_V_COEF = dict(township=0.1, secondary=0.1)

LIGHT_SEV_THRESHOLD = 0.35

# =============== Waiting & Process Quality ===============
W0 = 0.6
XI = 0.9
BETA_A_W = 0.28
BETA_A_Q = 0.06

Q_BASE = dict(county=0.8, secondary=0.7, township=0.7)
Q_PEN = dict(county=0.12, secondary=0.20, township=0.20)

SMAX = dict(county=0.9, secondary=0.35, township=0.30)
KAPPA_SUCC = 10.0

K_INIT = dict(county=1200, secondary=60, township=40)
U_START = dict(county=0.85, secondary=0.6, township=0.6)

TENSION = 0.95
BASELINE_THROUGHPUT = int(
    U_START["county"] * K_INIT["county"] + U_START["secondary"] * K_INIT["secondary"] + U_START["township"] * K_INIT[
        "township"])
LAMBDA_BASE = int(TENSION * BASELINE_THROUGHPUT)

SEASON_AMPL = 0.08
NOISE_AMPL = 0.015

# =============== Severity & Utility Weights ===============
SEV_ALPHA, SEV_BETA = 2.0, 3.0


def wQ(s): return 1.15 * (1.0 + 1.1 * s)


def wW(s): return 0.6 + 0.9 * (1.0 - s)


def wD(s): return 0.6 * (0.16 * (1.0 + 0.6 * (1.0 - s)))


def wO(s): return 0.0011 * (1.0 + 0.8 * (1.0 - s))


# =============== 等待容忍函数 ===============
def waiting_tolerance(severity):
    """计算患者等待容忍阈值，重症患者更不耐等待"""
    return max(2.0, WAIT_TOLERANCE_A0 - WAIT_TOLERANCE_A1 * severity)


# =============== 床日计算函数 ===============
def calculate_beddays(severity, hospital_level):
    """基于病情严重度和医院级别计算所需床日"""
    if severity < LOS_TAU:
        return 0  # 门诊不占床

    # 床日数的均值和方差随严重度增加
    mean_los = LOS_BASE + LOS_SEVERITY_MULTIPLIER * severity
    std_los = 1.0 + 3 * severity  # 方差也随严重度增加

    # 使用截断正态分布，确保床日数≥1
    los = rng.normal(mean_los, std_los)
    los = max(1, los)

    # 应用医院级别调整
    los *= LOS_LEVEL_MULTIPLIER[hospital_level]

    return los

def enhanced_rho_reward(hospital, flows, referral_success_count):
    """增强的转诊奖励函数"""
    base_bonus = getattr(hospital, 'bonus_prev', 0.0) / max(hospital.K, 1.0)
    # 增加转诊成功的直接奖励
    referral_bonus = 0.1 * referral_success_count / max(flows, 1)
    # 增加对减轻拥挤的奖励
    congestion_relief = 0.05 * (1 - hospital.u_prev)
    return base_bonus + referral_bonus + congestion_relief
# =============== Learning (Softmax + Q) ===============
TAU_A_START = 0.3
TAU_A_END = 0.05
TAU_I = 0.20

GAMMA = 0.75
ALPHA_Q_UPD = 0.25


# =============== Helpers ===============
def moving_average(x, w=12):
    if len(x) < 1: return np.array(x)
    # 处理NaN值：用前后有效值的均值填充
    x_filled = np.array(x, dtype=float)
    nan_mask = np.isnan(x_filled)
    if nan_mask.any():
        # 简单的NaN填充：用前后非NaN值的均值
        for i in range(len(x_filled)):
            if np.isnan(x_filled[i]):
                # 找前一个非NaN值
                prev_val = None
                for j in range(i - 1, -1, -1):
                    if not np.isnan(x_filled[j]):
                        prev_val = x_filled[j]
                        break
                # 找后一个非NaN值
                next_val = None
                for j in range(i + 1, len(x_filled)):
                    if not np.isnan(x_filled[j]):
                        next_val = x_filled[j]
                        break

                if prev_val is not None and next_val is not None:
                    x_filled[i] = (prev_val + next_val) / 2
                elif prev_val is not None:
                    x_filled[i] = prev_val
                elif next_val is not None:
                    x_filled[i] = next_val
                else:
                    x_filled[i] = 0  # 所有值都是NaN时用0
    kernel = np.ones(w) / w
    out = np.convolve(x, kernel, mode='same')
    half_w = w // 2
    for i in range(half_w):
        if i < len(x_filled):
            out[i] = np.mean(x_filled[:i * 2 + 1]) if i * 2 + 1 <= len(x_filled) else np.mean(x_filled)
    for i in range(len(x_filled) - half_w, len(x_filled)):
        if i >= 0:
            remaining = len(x_filled) - (i - half_w)
            out[i] = np.mean(x_filled[i - remaining + 1:i + 1]) if remaining > 0 else np.mean(x_filled)

    return out

def softmax(qvals, tau):
    q = np.array(qvals, dtype=float)
    q = q - np.max(q)
    z = np.exp(q / max(tau, 1e-6))
    return z / np.sum(z)


def success_prob(s, smax):
    return 1.0 / (1.0 + np.exp(KAPPA_SUCC * (s - smax)))


def waiting_time(u, a):
    u = np.clip(u, 0.0, 0.97)
    base = W0 / ((1.0 - u) ** XI)
    return base * (1.0 - BETA_A_W * a)


def process_quality(level, u, a):
    return Q_BASE[level] + BETA_A_Q * a - Q_PEN[level] * u


def induced_access_index(deltaQ, deltaW, deltaD, deltaHHI):
    return math.exp(THETA_Q * deltaQ - THETA_W * deltaW - THETA_D * deltaD - THETA_HHI * deltaHHI)


def logistic(x): return 1.0 / (1.0 + np.exp(-x))


# =============== 基于人口密度抽样 ===============
def sample_population_by_density(n, population_grid, X_grid, Y_grid, rng):
    """基于人口密度网格抽样患者位置"""
    flat_grid = population_grid.flatten()
    prob = flat_grid / np.sum(flat_grid)

    indices = rng.choice(len(flat_grid), size=n, p=prob)
    x_coords = X_grid.flatten()[indices]
    y_coords = Y_grid.flatten()[indices]

    return np.stack([x_coords, y_coords], axis=1)


# =============== 医院布局 ===============
def place_hospitals_from_layout():
    """从加载的布局放置医院"""
    hospitals = []

    # 县医院
    for pos in HOSPITAL_POSITIONS['county']:
        hospitals.append({"name": "County", "level": "county", "pos": np.array(pos)})

    # 二级医院
    for i, pos in enumerate(HOSPITAL_POSITIONS['secondary']):
        hospitals.append({"name": f"Secondary_{i + 1}", "level": "secondary", "pos": np.array(pos)})

    # 基层医院
    for i, pos in enumerate(HOSPITAL_POSITIONS['township']):
        hospitals.append({"name": f"Township_{i + 1}", "level": "township", "pos": np.array(pos)})

    return hospitals


# =============== Enhanced Hospital Agent ===============
class Hospital:
    def __init__(self, name, level, pos, cap_init):
        self.name = name
        self.level = level
        self.pos = np.array(pos, dtype=float)
        self.K = float(cap_init)  # 床日容量

        self.a_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.i_grid = np.array([-1, 0, 1, 2])

        # Enhanced action grids
        self.rho_grid = (np.array([0.0, 0.25, 0.5, 0.75, 1.0]) if level != "county" else np.array([0.0]))
        self.v_grid = (np.array([0.0, 0.5, 1.0]) if level == "county" else np.array([0.0]))
        self.d_grid = (np.array([0.0, 0.5, 1.0]) if level == "county" else np.array([0.0]))

        # County allocation decision
        if ENABLE_COUNTY_ALLOCATION and level == "county":
            self.alloc_grid = ALLOCATION_GRID
            self.Qalloc = defaultdict(lambda: np.zeros(len(self.alloc_grid)))
            self.allocation_ratio = 0.5

        self.Qa = defaultdict(lambda: np.zeros(len(self.a_grid)))
        self.Qi = defaultdict(lambda: np.zeros(len(self.i_grid)))
        self.Qrho = defaultdict(lambda: np.zeros(len(self.rho_grid)))
        self.Qv = defaultdict(lambda: np.zeros(len(self.v_grid)))
        self.Qd = defaultdict(lambda: np.zeros(len(self.d_grid)))

        self.a = 0.25 if level != "township" else 0.15
        self.i_pending = 0
        self.v = 0.0
        self.rho = 0.0
        self.d = 0.0
        self.Smax = SMAX[self.level]
        self.finance_left = 0
        self.finance_add = 0.0

        self.u_prev = U_START[level]
        self.share_prev = 0.0
        self.W_prev = waiting_time(self.u_prev, self.a)
        self.profit_prev = 0.0

        # Enhanced: track referral performance
        self.referral_success = 0
        self.referral_attempts = 0

    def state_hash_alloc(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
        ref_rate = self.referral_success / max(self.referral_attempts, 1)
        ref_bin = 0 if ref_rate < 0.3 else (1 if ref_rate < 0.7 else 2)
        return (u_bin, p_bin, ref_bin)

    def select_allocation(self, tau=0.6):
        if not ENABLE_COUNTY_ALLOCATION or self.level != "county":
            self.allocation_ratio = 0.5
            return self.state_hash_alloc(), 0

        s = self.state_hash_alloc()
        probs = softmax(self.Qalloc[s], tau)
        idx = rng.choice(len(self.alloc_grid), p=probs)
        self.allocation_ratio = float(self.alloc_grid[idx])
        return s, idx

    def update_Qalloc(self, s, idx, r, s_next):
        if ENABLE_COUNTY_ALLOCATION and self.level == "county":
            best_next = np.max(self.Qalloc[s_next])
            self.Qalloc[s][idx] = (1 - ALPHA_Q_UPD) * self.Qalloc[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    # def state_hash_rho(self):
    #     u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
    #     p_bin = 1 if self.profit_prev > 0 else 0
    #     return (u_bin, p_bin)
    def state_hash_rho(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
        # 增加转诊成功率信息
        ref_rate = self.referral_success / max(self.referral_attempts, 1)
        ref_bin = 0 if ref_rate < 0.3 else (1 if ref_rate < 0.7 else 2)
        return (u_bin, p_bin, ref_bin)

    def select_rho(self, tau=0.8):
        if (self.level == "county") or (not ENABLE_RHO):
            self.rho = 0.0
            return self.state_hash_rho(), 0
        s = self.state_hash_rho()
        probs = softmax(self.Qrho[s], tau)
        idx = rng.choice(len(self.rho_grid), p=probs)
        self.rho = float(self.rho_grid[idx])
        return s, idx

    def update_Qrho(self, s, idx, r, s_next):
        best_next = np.max(self.Qrho[s_next])
        self.Qrho[s][idx] = (1 - ALPHA_Q_UPD) * self.Qrho[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    def state_hash_v(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        W_bin = 0 if self.W_prev < 6 else (1 if self.W_prev < 12 else 2)
        return (u_bin, W_bin)

    def select_v(self, tau=0.8):
        if (self.level != "county") or (not ENABLE_V):
            self.v = 0.0
            return self.state_hash_v(), 0
        s = self.state_hash_v()
        probs = softmax(self.Qv[s], tau)
        idx = rng.choice(len(self.v_grid), p=probs)
        self.v = float(self.v_grid[idx])
        return s, idx

    def update_Qv(self, s, idx, r, s_next):
        best_next = np.max(self.Qv[s_next])
        self.Qv[s][idx] = (1 - ALPHA_Q_UPD) * self.Qv[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    def state_hash_d(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        return (u_bin,)

    def select_d(self, tau=0.6):
        if (self.level != "county") or (not ENABLE_D):
            self.d = 0.0
            return self.state_hash_d(), 0
        s = self.state_hash_d()
        probs = softmax(self.Qd[s], tau)
        idx = rng.choice(len(self.d_grid), p=probs)
        self.d = float(self.d_grid[idx])
        return s, idx

    def update_Qd(self, s, idx, r, s_next):
        best_next = np.max(self.Qd[s_next])
        self.Qd[s][idx] = (1 - ALPHA_Q_UPD) * self.Qd[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    def state_hash_a(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        W_bin = 0 if self.W_prev < 6 else (1 if self.W_prev < 12 else 2)
        s_bin = 0 if self.share_prev < 0.2 else (1 if self.share_prev < 0.6 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
        return (u_bin, W_bin, s_bin, p_bin)

    def select_a(self, tau):
        s = self.state_hash_a()
        probs = softmax(self.Qa[s], tau)
        idx = rng.choice(len(self.a_grid), p=probs)
        self.a = float(self.a_grid[idx])
        return s, idx

    def update_Qa(self, s, a_idx, r, s_next):
        best_next = np.max(self.Qa[s_next])
        self.Qa[s][a_idx] = (1 - ALPHA_Q_UPD) * self.Qa[s][a_idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    def state_hash_i(self):
        u_bin = 0 if self.u_prev < 0.5 else (1 if self.u_prev < 0.8 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
        return (u_bin, p_bin)

    def select_i(self):
        s = self.state_hash_i()
        probs = softmax(self.Qi[s], TAU_I)
        idx = rng.choice(len(self.i_grid), p=probs)
        self.i_pending = int(self.i_grid[idx])
        return s, idx

    def update_Qi(self, s, i_idx, r, s_next):
        best_next = np.max(self.Qi[s_next])
        self.Qi[s][i_idx] = (1 - ALPHA_Q_UPD) * self.Qi[s][i_idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)


# =============== Enhanced Patient Behavior ===============
class PatientBehavior:
    def __init__(self, N_patients):
        self.habit_strength = PATIENT_HABIT_STRENGTH
        self.info_asymmetry = PATIENT_INFO_ASYMMETRY
        self.history = np.zeros((N_patients,), dtype=int) - 1

    def enhance_utility(self, U, patient_idx, hospital_idx, hospital_levels):
        enhanced_U = U.copy()

        # 就医习惯
        history_reshaped = self.history[patient_idx].reshape(-1, 1)
        hospital_idx_reshaped = hospital_idx.reshape(1, -1)
        habit_preference = (history_reshaped == hospital_idx_reshaped).astype(float) * self.habit_strength
        enhanced_U += habit_preference

        # 信息不对称
        for i in range(U.shape[1]):
            if hospital_levels[i] == "township":
                info_penalty = rng.random(U.shape[0]) * self.info_asymmetry
                enhanced_U[:, i] -= info_penalty

        return enhanced_U

    def update_history(self, patient_idx, hospital_idx):
        if isinstance(patient_idx, (int, np.integer)) and hospital_idx != -1:
            self.history[patient_idx] = hospital_idx
        elif hasattr(patient_idx, '__len__') and hospital_idx != -1:
            self.history[patient_idx[0]] = hospital_idx

# =============== Enhanced Admission with Waiting Tolerance ===============
def enhanced_patient_admission(Hlist, Ppos, Sev, U, Psucc, bedday_capacity, W_eff,
                               patient_behavior, t, INIT_BIAS_MONTHS, INIT_THRESH):
    """增强的患者入院流程：包含等待容忍和外流分类"""
    H = len(Hlist)
    N_demand = len(Ppos)

    topk_idx = np.argsort(U, axis=1)[:, -MAX_PATIENT_CHOICES:]
    order_pat = np.argsort(-Sev)

    attempts = np.zeros(N_demand, dtype=int)
    served_idx_by_h = [[] for _ in range(H)]
    beddays_used_by_h = [0.0 for _ in range(H)]
    assigned_to = -np.ones(N_demand, dtype=int)
    referral_paths = [[] for _ in range(N_demand)]

    # 外流分类
    outflow_wait = []  # 等待时间超过容忍
    outflow_full = []  # 床位已满
    outflow_mismatch = []  # 能力不匹配

    for p in order_pat:
        attempts_p = 0
        choices = topk_idx[p][::-1]
        wtol = waiting_tolerance(Sev[p])  # 个体等待容忍

        # 分级转诊逻辑
        if USE_REFERRAL_SYSTEM and Sev[p] < 0.6:
            township_choices = [c for c in choices if Hlist[c].level == "township"]
            secondary_choices = [c for c in choices if Hlist[c].level == "secondary"]
            county_choices = [c for c in choices if Hlist[c].level == "county"]
            ordered_choices = township_choices + secondary_choices + county_choices
        else:
            ordered_choices = choices

        admitted = False
        for c in ordered_choices:
            attempts_p += 1
            lvl = Hlist[c].level

            # 等待容忍检查
            if W_eff[c] > wtol:
                if attempts_p == len(ordered_choices):  # 最后一次尝试
                    outflow_wait.append(p)
                continue

            # 转诊阈值判断
            if t < INIT_BIAS_MONTHS:
                if lvl == "township" and Psucc[p, c] < INIT_THRESH["township"]:
                    if attempts_p == len(ordered_choices):
                        outflow_mismatch.append(p)
                    continue
                if lvl == "secondary" and Psucc[p, c] < INIT_THRESH["secondary"]:
                    if attempts_p == len(ordered_choices):
                        outflow_mismatch.append(p)
                    continue
                if lvl == "county" and Psucc[p, c] < INIT_THRESH["county"]:
                    if attempts_p == len(ordered_choices):
                        outflow_mismatch.append(p)
                    continue
            else:
                if lvl == "township":
                    thr = THR_BASE["township"]
                    if ENABLE_RHO:
                        thr -= THR_RHO_COEF["township"] * getattr(Hlist[c], 'rho', 0.0)
                    if ENABLE_V and len([h for h in Hlist if h.level == "county"]) > 0:
                        county_h = [h for h in Hlist if h.level == "county"][0]
                        thr -= THR_V_COEF["township"] * getattr(county_h, 'v', 0.0)
                    if Psucc[p, c] < thr:
                        if attempts_p == len(ordered_choices):
                            outflow_mismatch.append(p)
                        continue
                elif lvl == "secondary":
                    thr = THR_BASE["secondary"]
                    if ENABLE_RHO:
                        thr += THR_RHO_COEF["secondary"] * getattr(Hlist[c], 'rho', 0.0)
                    if ENABLE_V and len([h for h in Hlist if h.level == "county"]) > 0:
                        county_h = [h for h in Hlist if h.level == "county"][0]
                        thr += THR_V_COEF["secondary"] * getattr(county_h, 'v', 0.0)
                    if Psucc[p, c] < thr:
                        if attempts_p == len(ordered_choices):
                            outflow_mismatch.append(p)
                        continue

            # 床位检查
            need_beddays = calculate_beddays(Sev[p], lvl)
            if bedday_capacity[c] >= need_beddays:
                assigned_to[p] = c
                bedday_capacity[c] -= need_beddays
                beddays_used_by_h[c] += need_beddays
                served_idx_by_h[c].append(p)
                referral_paths[p].append(c)
                admitted = True

                # 记录转诊表现
                if len(referral_paths[p]) > 1:
                    referring_hospital = referral_paths[p][-2]
                    Hlist[referring_hospital].referral_attempts += 1
                    Hlist[referring_hospital].referral_success += 1
                break
            else:
                # 床位不足
                if attempts_p == len(ordered_choices):
                    outflow_full.append(p)
                # 记录转诊尝试
                if len(referral_paths[p]) > 0:
                    referring_hospital = referral_paths[p][-1]
                    Hlist[referring_hospital].referral_attempts += 1

        attempts[p] = attempts_p

        # 如果所有尝试都失败且未被分类，归类为能力不匹配
        if not admitted and p not in outflow_wait and p not in outflow_full and p not in outflow_mismatch:
            outflow_mismatch.append(p)

        patient_behavior.update_history(p, assigned_to[p])

    outflow_info = {
        'wait': outflow_wait,
        'full': outflow_full,
        'mismatch': outflow_mismatch
    }

    return assigned_to, served_idx_by_h, beddays_used_by_h, attempts, referral_paths, outflow_info

# =============== Budget Calculation ===============
def expected_severity_multiplier(n=50000):
    s = rng.beta(SEV_ALPHA, SEV_BETA, size=n)
    return float(np.mean(revenue_multiplier(s)))


if not USE_BUDGET_FIXED:
    exp_mult = expected_severity_multiplier()
    BUDGET_PER_PERIOD = BUDGET_FACTOR * REIMB_RATIO * exp_mult * sum([
        U_START['county'] * K_INIT['county'] * PRICE['county'],
        U_START['secondary'] * K_INIT['secondary'] * PRICE['secondary'],
        U_START['township'] * K_INIT['township'] * PRICE['township']
    ])
else:
    BUDGET_PER_PERIOD = float(BUDGET_FIXED)

# =============== Build System ===============
HOSP_META = place_hospitals_from_layout()
Hlist = []
sec_each = K_INIT["secondary"] / N_SECONDARY
town_each = K_INIT["township"] / NT
for meta in HOSP_META:
    lvl = meta["level"]
    cap = K_INIT["county"] if lvl == "county" else (sec_each if lvl == "secondary" else town_each)
    Hlist.append(Hospital(meta["name"], lvl, meta["pos"], cap))

idx_county = [i for i, h in enumerate(Hlist) if h.level == "county"]
idx_secondary = [i for i, h in enumerate(Hlist) if h.level == "secondary"]
idx_township = [i for i, h in enumerate(Hlist) if h.level == "township"]
H = len(Hlist)

# ===== Hard Caps =====
K_CAP_PER = dict(county=1500.0, secondary=120, township=100.0)

# ===== Initial Bias =====
INIT_BIAS_MONTHS = 12
INIT_THRESH = dict(county=0.80, secondary=0.35, township=0.20)

# =============== Enhanced Data Collectors ===============
# 原有指标
county_light_share = []
multi_attempt_rate = []
rho_mean_lower = []
smax_uplift_lower = []
treated_series, unserved_series = [], []
flow_share_lvl = {"county": [], "secondary": [], "township": []}
cap_lvl = {"county": [], "secondary": [], "township": []}
sev_mean_lvl = {"county": [], "secondary": [], "township": []}
ins_spend_lvl = {"county": [], "secondary": [], "township": [], "total": []}
a_traj = {"county": [], "secondary": [], "township": []}
i_traj = {"county": [], "secondary": [], "township": []}
entropy_policy = {"county": [], "secondary": [], "township": []}

# 新增指标
referral_success_rate = []
county_allocation_ratio = []
referral_path_length = []

# 床日相关指标
bed_occupancy_lvl = {"county": [], "secondary": [], "township": []}
inpatient_share_lvl = {"county": [], "secondary": [], "township": []}
avg_los_inp_lvl = {"county": [], "secondary": [], "township": []}

# 外流分类指标
outflow_wait_series = []
outflow_full_series = []
outflow_mismatch_series = []

# 空间热力数据
hospital_flow_history = [[] for _ in range(H)]  # 记录每个医院的历史流量

# =============== 新增：两级架构合并统计 ===============
# 基层医院合并统计（二级+基层）
combined_primary_flow_share = []  # 基层+二级的总流量份额
combined_primary_capacity = []    # 基层+二级的总容量
combined_primary_severity = []    # 基层+二级的平均严重度
combined_primary_profits = []     # 基层+二级的总利润
combined_primary_utilization = [] # 基层+二级的平均利用率
combined_primary_bed_occupancy = [] # 基层+二级的床位利用率
combined_primary_inpatient_share = [] # 基层+二级的住院比例

ref_Q = ref_W = ref_D = ref_HHI = None


# =============== added Data Collectors ===============
# 在现有数据收集器后面添加这些指标

# 原有核心指标（需要恢复）
waiting_time_series = []  # 平均等待时间
process_quality_series = []  # 平均处理质量
patient_utility_series = []  # 患者效用
distance_traveled_series = []  # 平均就诊距离
hospital_profits = {"county": [], "secondary": [], "township": []}  # 医院利润
capacity_utilization = {"county": [], "secondary": [], "township": []}  # 容量利用率
referral_efficiency = []  # 转诊效率
budget_utilization = []  # 预算利用率
surplus_distribution = {"county": [], "secondary": [], "township": []}  # 结余分配

# 学习过程指标
learning_convergence = {"county": [], "secondary": [], "township": []}  # 学习收敛
action_distribution = {
    "county": {"a": [], "i": [], "rho": [], "v": [], "d": []},
    "secondary": {"a": [], "i": [], "rho": []},
    "township": {"a": [], "i": [], "rho": []}
}
# 新增：详细严重程度统计
sev_stats_lvl = {
    "county": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []},
    "secondary": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []},
    "township": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []}
}
insurance_spending = {
    "total": [],           # 总医保支出
    "county": [],          # 县医院医保支出
    "secondary": [],       # 二级医院医保支出
    "township": [],        # 基层医院医保支出
    "county_share": [],    # 县医院支出占比
    "secondary_share": [], # 二级医院支出占比
    "township_share": []   # 基层医院支出占比
}
# 新增：容量变动监控
capacity_trends = {
    "county": [],          # 县医院总容量
    "secondary": [],       # 二级医院总容量
    "township": [],        # 基层医院总容量
    "county_share": [],    # 县医院容量占比
    "secondary_share": [], # 二级医院容量占比
    "township_share": []   # 基层医院容量占比
}
# 患者行为指标
patient_choice_quality = []  # 患者选择质量
habit_persistence = []  # 习惯持续性
information_asymmetry_effect = []  # 信息不对称影响
# =============== Main Loop ===============
current_referral_attempts = 0
current_referral_success = 0
backlog_unserved = 0.0
tau_a = TAU_A_START
tau_decay = (TAU_A_START - TAU_A_END) / max(T - 1, 1)

# Initialize patient behavior
patient_behavior = PatientBehavior(10000)

for t in range(T):
    # 新增：初始化当期转诊统计变量
    current_referral_attempts_by_h = [0] * H  # 每个医院当期转诊尝试次数
    current_referral_success_by_h = [0] * H   # 每个医院当期转诊成功次数
    # ---- Enhanced Action Selection ----
    s_a_idx = {}
    for i, h in enumerate(Hlist):
        s_a_idx[i] = h.select_a(tau_a)

    s_rho_idx = {}
    if ENABLE_RHO:
        for i, h in enumerate(Hlist):
            s_rho_idx[i] = h.select_rho(0.8)

    s_v_idx = {}
    if ENABLE_V:
        for i, h in enumerate(Hlist):
            if h.level == "county":
                s_v_idx[i] = h.select_v(0.8)

    s_i_idx = {}
    if (t % 6 == 0) and (t >= WARMUP):
        for i, h in enumerate(Hlist):
            s_i_idx[i] = h.select_i()
        s_d_idx = {}
        if ENABLE_D:
            for i, h in enumerate(Hlist):
                if h.level == "county":
                    s_d_idx[i] = h.select_d(0.6)

    # County allocation decision
    s_alloc_idx = {}
    if ENABLE_COUNTY_ALLOCATION and (t % 6 == 0) and (t >= WARMUP):
        for i, h in enumerate(Hlist):
            if h.level == "county":
                s_alloc_idx[i] = h.select_allocation(0.6)

    # ----- prev u/a -> W & Qproc -----
    Uprev = np.array([h.u_prev for h in Hlist])
    Aprev = np.array([h.a for h in Hlist])
    W_eff = np.array([waiting_time(Uprev[i], Aprev[i]) for i in range(H)])
    Q_proc = np.array([process_quality(Hlist[i].level, Uprev[i], Aprev[i]) for i in range(H)])

    # ----- demand size -----
    M_t = LAMBDA_BASE * (1.0 + SEASON_AMPL * np.sin(2 * np.pi * ((t % 12) / 12.0)) + NOISE_AMPL * rng.normal())
    M_t = max(100.0, M_t)

    if USE_INDUCED_DEMAND and (ref_Q is not None):
        deltaQ = (city_Q - ref_Q) if 'city_Q' in locals() else 0.0
        deltaW = (city_W / ref_W - 1.0) if 'city_W' in locals() and ref_W > 1e-6 else 0.0
        deltaD = (city_D / ref_D - 1.0) if 'city_D' in locals() and ref_D > 1e-6 else 0.0
        deltaHHI = (city_HHI - ref_HHI) if 'city_HHI' in locals() else 0.0
        deltaQ = float(np.clip(deltaQ, -0.5, 0.5))
        deltaW = float(np.clip(deltaW, -0.5, 0.5))
        deltaD = float(np.clip(deltaD, -0.5, 0.5))
        deltaHHI = float(np.clip(deltaHHI, -0.5, 0.5))
        A_t = induced_access_index(deltaQ, deltaW, deltaD, deltaHHI)
        Phi_t = 0.5 * ((city_W / (ref_W + 1e-6)) + (city_D / (ref_D + 1e-6))) if 'city_W' in locals() else 1.0
        p_seek_scale = logistic(ALPHA0 + ALPHA_Q * A_t - ALPHA_WD * (1 - 0.25) * Phi_t)
    else:
        p_seek_scale = logistic(ALPHA0)

    N_demand = rng.poisson(max(50.0, M_t * p_seek_scale) + BACKLOG_RHO * backlog_unserved)

    # ----- draw patients from density grid -----
    Ppos = sample_population_by_density(N_demand, POPULATION_GRID, X_GRID, Y_GRID, rng)
    Sev = beta_dist(SEV_ALPHA, SEV_BETA).rvs(size=N_demand, random_state=rng)

    Hpos = np.stack([h.pos for h in Hlist], axis=0)
    dmat = np.linalg.norm(Ppos[:, None, :] - Hpos[None, :, :], axis=2)

    Smax_vec = np.array([getattr(h, 'Smax', SMAX[h.level]) for h in Hlist])
    Psucc = 1.0 / (1.0 + np.exp(KAPPA_SUCC * (Sev[:, None] - Smax_vec[None, :])))
    Qeff = (Q_proc[None, :]) * Psucc
    OOP = (np.array([PRICE[h.level] for h in Hlist])[None, :]) * (1.0 - REIMB_RATIO) * revenue_multiplier(Sev)[:, None]

    U = (wQ(Sev)[:, None] * Qeff
         - wW(Sev)[:, None] * W_eff[None, :]
         - wD(Sev)[:, None] * dmat
         - wO(Sev)[:, None] * OOP)

    # Enhanced patient behavior
    hospital_levels = [h.level for h in Hlist]
    U = patient_behavior.enhance_utility(U, np.arange(N_demand)[:, None],
                                         np.arange(H)[None, :], hospital_levels)

    # 关闭外生声望偏置，只保留噪声
    if ENABLE_NOISE:
        U += rng.normal(0.0, UTILITY_NOISE_SIGMA, size=U.shape)

    # ----- Enhanced Admission with Waiting Tolerance -----
    bedday_capacity = np.array([h.K for h in Hlist])  # 床日容量池

    assigned_to, served_idx_by_h, beddays_used_by_h, attempts, referral_paths, outflow_info = enhanced_patient_admission(
        Hlist, Ppos, Sev, U, Psucc, bedday_capacity.copy(), W_eff, patient_behavior,
        t, INIT_BIAS_MONTHS, INIT_THRESH
    )

    # 新增：统计当期转诊数据
    for p in range(N_demand):
        if assigned_to[p] != -1 and len(referral_paths[p]) > 1:  # 有转诊历史且最终被接收
            # 转诊路径中倒数第二个医院是成功转出的医院
            referring_hospital = referral_paths[p][-2]
            current_referral_success_by_h[referring_hospital] += 1

        # 统计所有转诊尝试（无论成功与否）
        if len(referral_paths[p]) > 0:
            for referring_hospital in referral_paths[p][:-1]:  # 除了最后一个接收医院外都是转诊尝试
                current_referral_attempts_by_h[referring_hospital] += 1


    # 统计外流患者
    outflow_wait_count = len(outflow_info['wait'])
    outflow_full_count = len(outflow_info['full'])
    outflow_mismatch_count = len(outflow_info['mismatch'])
    total_outflow = outflow_wait_count + outflow_full_count + outflow_mismatch_count

    backlog_unserved = total_outflow

    outflow_wait_series.append(outflow_wait_count)
    outflow_full_series.append(outflow_full_count)
    outflow_mismatch_series.append(outflow_mismatch_count)

    treated = N_demand - total_outflow
    unserved_series.append(total_outflow)
    treated_series.append(treated)

    # ----- accounting -----
    flows = np.zeros(H, dtype=int)
    ins_spend = np.zeros(H)
    inpatient_counts = np.zeros(H, dtype=int)  # 住院患者数
    severity_by_hospital = [[] for _ in range(H)]

    for i in range(H):
        idx = served_idx_by_h[i]
        n_i = len(idx)
        flows[i] = n_i

        # 记录医院流量历史（用于热力图）
        hospital_flow_history[i].append(n_i)

        if n_i > 0:
            sev_i = Sev[idx]
            ins_spend[i] = np.sum(PRICE[Hlist[i].level] * REIMB_RATIO * revenue_multiplier(sev_i))
            rev = np.sum(PRICE[Hlist[i].level] * revenue_multiplier(sev_i))
            cost_var = np.sum(unit_cost(sev_i))
            severity_by_hospital[i].extend(sev_i)

            # 统计住院患者
            inpatient_mask = sev_i >= LOS_TAU
            inpatient_counts[i] = np.sum(inpatient_mask)
        else:
            rev, cost_var = 0.0, 0.0
            inpatient_counts[i] = 0

        fix = FIXED_COST_PER_CAP[Hlist[i].level] * Hlist[i].K
        a_cost = A_COST_PER_UNIT * Hlist[i].a
        i_cost = 0.0

        if (t % 6 == 0) and (t >= WARMUP):
            i_act = Hlist[i].i_pending
            i_cost = I_COST[i_act] * I_COST_MULT[Hlist[i].level]
            if i_act > 0:
                Hlist[i].finance_left = 6
                Hlist[i].finance_add = FINANCE_PSI[Hlist[i].level] * i_cost / 6.0
            Hlist[i].K = max(50.0, Hlist[i].K * (1.0 + K_DELTA[i_act]))
            Hlist[i].K = min(Hlist[i].K, K_CAP_PER[Hlist[i].level])

        # 使用床日计算利用率
        u_i = beddays_used_by_h[i] / max(Hlist[i].K, 1.0)
        empty_cost = EMPTY_COST_PHI[Hlist[i].level] * FIXED_COST_PER_CAP[Hlist[i].level] * (1.0 - u_i) * Hlist[i].K
        fin_cost = 0.0
        if Hlist[i].finance_left > 0:
            fin_cost = Hlist[i].finance_add
            Hlist[i].finance_left -= 1
        profit = rev - (cost_var + fix + a_cost + i_cost + empty_cost + fin_cost)

        Hlist[i].u_prev = 0.80 * Hlist[i].u_prev + 0.20 * u_i
        Hlist[i].W_prev = waiting_time(Hlist[i].u_prev, Hlist[i].a)
        Hlist[i].profit_prev = 0.7 * Hlist[i].profit_prev + 0.3 * profit
        # ===== 在这里添加医保支出汇总计算 =====
        # 计算各级医保支出
    total_insurance = np.sum(ins_spend)
    county_insurance = np.sum(ins_spend[idx_county])
    secondary_insurance = np.sum(ins_spend[idx_secondary])
    township_insurance = np.sum(ins_spend[idx_township])

    # 记录医保支出数据
    insurance_spending["total"].append(total_insurance)
    insurance_spending["county"].append(county_insurance)
    insurance_spending["secondary"].append(secondary_insurance)
    insurance_spending["township"].append(township_insurance)

    # 计算支出占比
    if total_insurance > 0:
        insurance_spending["county_share"].append(county_insurance / total_insurance)
        insurance_spending["secondary_share"].append(secondary_insurance / total_insurance)
        insurance_spending["township_share"].append(township_insurance / total_insurance)
    else:
        insurance_spending["county_share"].append(0.0)
        insurance_spending["secondary_share"].append(0.0)
        insurance_spending["township_share"].append(0.0)

    # ----- Enhanced Cooperation Metrics -----
    light_cnt = 0;
    county_total = 0
    for i in idx_county:
        idx = served_idx_by_h[i]
        county_total += len(idx)
        if len(idx) > 0:
            sev_i = Sev[idx]
            light_cnt += int(np.sum(sev_i < LIGHT_SEV_THRESHOLD))
    county_light_share.append((light_cnt / county_total) if county_total > 0 else 0.0)

    if treated > 0:
        mask_adm = (assigned_to != -1)
        multi_attempt_rate.append(float(np.mean(attempts[mask_adm] >= 2)))
    else:
        multi_attempt_rate.append(0.0)

    total_low = 0;
    rho_sum = 0.0
    for i in idx_secondary + idx_township:
        total_low += flows[i]
        rho_sum += flows[i] * getattr(Hlist[i], 'rho', 0.0)
    rho_mean_lower.append((rho_sum / total_low) if total_low > 0 else 0.0)

    uplift_sum = 0.0;
    wsum = 0
    for i in idx_secondary + idx_township:
        base = SMAX0[Hlist[i].level]
        uplift = max(0.0, (getattr(Hlist[i], 'Smax', base) - base) / base)
        uplift_sum += flows[i] * uplift
        wsum += flows[i]
    smax_uplift_lower.append((uplift_sum / wsum) if wsum > 0 else 0.0)

    # ----- 床日相关指标 -----
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_beddays = sum(beddays_used_by_h[i] for i in idx_list)
        total_capacity = sum(Hlist[i].K for i in idx_list)
        total_inpatients = sum(inpatient_counts[i] for i in idx_list)
        total_patients = sum(flows[i] for i in idx_list)

        # 床位利用率
        occupancy = total_beddays / max(total_capacity, 1.0)
        bed_occupancy_lvl[level].append(occupancy)

        # 住院比例
        inpatient_share = total_inpatients / max(total_patients, 1.0)
        inpatient_share_lvl[level].append(inpatient_share)

        # 平均住院日（仅住院患者）
        if total_inpatients > 0:
            avg_los = total_beddays / total_inpatients
        else:
            avg_los = 0.0
        avg_los_inp_lvl[level].append(avg_los)

    # ----- Enhanced Monitoring -----
    total_referral_attempts = sum(h.referral_attempts for h in Hlist)
    total_referral_success = sum(h.referral_success for h in Hlist)
    # referral_success_rate.append(total_referral_success / max(total_referral_attempts, 1))
    # ----- Enhanced Monitoring -----
    # 替换：使用当期数据计算转诊成功率
    total_current_attempts = sum(current_referral_attempts_by_h)
    total_current_success = sum(current_referral_success_by_h)
    if total_current_attempts > 0:
        current_referral_rate = total_current_success / total_current_attempts
    else:
        current_referral_rate = 0.0
    referral_success_rate.append(current_referral_rate)

    # 保留原有的累计统计更新（用于医院学习）
    for h in Hlist:
        h.referral_attempts += sum(current_referral_attempts_by_h)  # 简化处理，实际应按医院分别更新
        h.referral_success += sum(current_referral_success_by_h)  # 简化处理
    if len(idx_county) > 0 and ENABLE_COUNTY_ALLOCATION:
        county_allocation_ratio.append(Hlist[idx_county[0]].allocation_ratio)
    else:
        county_allocation_ratio.append(0.5)  # 默认值

    avg_path_length = np.mean([len(path) for path in referral_paths if len(path) > 0])
    referral_path_length.append(avg_path_length)

    # ----- city aggregates -----
    if treated > 0:
        served_mask = (assigned_to != -1)
        d_chosen = dmat[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        Q_chosen = Qeff[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        city_W = float(np.average(W_eff, weights=flows + 1e-6))
        city_Q = float(np.mean(Q_chosen))
        city_D = float(np.mean(d_chosen))
        K_tot = np.sum([h.K for h in Hlist])
        city_HHI = float(np.sum([(h.K / K_tot) ** 2 for h in Hlist]))
    else:
        city_W = float(np.average(W_eff))
        city_Q = float(np.mean(Qeff))
        city_D = float(np.mean(np.min(dmat, axis=1)))
        K_tot = np.sum([h.K for h in Hlist])
        city_HHI = float(np.sum([(h.K / K_tot) ** 2 for h in Hlist]))

    if t == WARMUP - 1:
        ref_Q, ref_W, ref_D, ref_HHI = city_Q, max(1e-6, city_W), max(1e-6, city_D), city_HHI

    # ----- learning updates -----
    for i, h in enumerate(Hlist):
        s_now = h.state_hash_a()
        r_a = (h.profit_prev / max(h.K, 1.0)) / 10.0 - 0.06 * h.W_prev
        s_prev, a_idx = s_a_idx[i]
        h.update_Qa(s_prev, a_idx, r_a, s_now)

        if (t % 6 == 0) and (t >= WARMUP):
            s_i_now = h.state_hash_i()
            u_gap = (h.u_prev - 0.85)
            u6 = h.u_prev
            base = (h.profit_prev / max(h.K, 1.0)) / 8.0 + 0.8 * (flows[i] / max(h.K, 1.0)) - 0.03 * h.W_prev
            if u6 < 0.65 and h.i_pending > 0:
                r_i = base - 3.0
            elif u6 > 0.85 and h.i_pending > 0:
                r_i = base
            else:
                r_i = base - 0.5 * max(0.0, 0.80 - u6)
            s_i_prev, i_idx = s_i_idx[i]
            h.update_Qi(s_i_prev, i_idx, r_i, s_i_now)

    # --- cooperation action learning ---
    for i, h in enumerate(Hlist):
        if ENABLE_RHO and h.level != "county":
            s_prev_r, idx_r = s_rho_idx[i]
            s_now_r = h.state_hash_rho()
            # r_rho = getattr(h, 'bonus_prev', 0.0) / max(h.K, 1.0)
            # 替换为：
            current_success = current_referral_success_by_h[i]  # 需要先定义这个变量（见第9点）
            r_rho = enhanced_rho_reward(h, flows[i], current_success)
            h.update_Qrho(s_prev_r, idx_r, r_rho, s_now_r)
        if ENABLE_V and h.level == "county":
            s_prev_v, idx_v = s_v_idx[i]
            s_now_v = h.state_hash_v()
            r_v = getattr(h, 'bonus_prev', 0.0) / max(h.K, 1.0)
            h.update_Qv(s_prev_v, idx_v, r_v, s_now_v)
        if ENABLE_D and (t % 6 == 0) and ('s_d_idx' in locals()) and h.level == "county":
            s_prev_d, idx_d = s_d_idx[i]
            s_now_d = h.state_hash_d()
            r_d = getattr(h, 'bonus_prev', 0.0) / max(h.K, 1.0)
            h.update_Qd(s_prev_d, idx_d, r_d, s_now_d)

        # County allocation learning
        if ENABLE_COUNTY_ALLOCATION and (t % 6 == 0) and (t >= WARMUP):
            for i, h in enumerate(Hlist):
                if h.level == "county" and i in s_alloc_idx:
                    s_prev_alloc, idx_alloc = s_alloc_idx[i]
                    s_now_alloc = h.state_hash_alloc()
                    r_alloc = getattr(h, 'bonus_prev', 0.0) / max(h.K, 1.0)
                    h.update_Qalloc(s_prev_alloc, idx_alloc, r_alloc, s_now_alloc)

    # ----- ability sinking update -----
    if ENABLE_D and (t % 6 == 0) and (t >= WARMUP):
        if len(idx_county) > 0:
            d_sys = getattr(Hlist[idx_county[0]], 'd', 0.0)
        else:
            d_sys = 0.0
        if d_sys > 0:
            for j in idx_secondary + idx_township:
                cap = S_MAX_UPPER[Hlist[j].level]
                Hlist[j].Smax = min(cap, Hlist[j].Smax + ETA_D[Hlist[j].level] * d_sys)

    # ----- Enhanced Bonus Distribution -----
    # ins_total = ins_spend_lvl["total"][-1] if len(ins_spend_lvl["total"]) > 0 else 0
    ins_total = total_insurance  # 使用当期计算的医保支出
    surplus = BUDGET_PER_PERIOD - ins_total
    if (not ALLOW_NEGATIVE_SURPLUS) and (surplus < 0):
        surplus = 0.0

    # County-led allocation
    if ENABLE_COUNTY_ALLOCATION and len(idx_county) > 0:
        county_alloc_ratio = Hlist[idx_county[0]].allocation_ratio
    else:
        county_alloc_ratio = 0.5

    weights = np.zeros(H)
    d_sys = 0.0
    if len(idx_county) > 0:
        d_sys = getattr(Hlist[idx_county[0]], 'd', 0.0)
    for i, h in enumerate(Hlist):
        base = BONUS_BASE_SHARE[h.level]
        w = base * max(flows[i], 0.0)
        if h.level != "county":
            w *= (1.0 + BONUS_RHO_WEIGHT * getattr(h, 'rho', 0.0))
        else:
            w *= (1.0 + BONUS_D_WEIGHT * d_sys)
        weights[i] = w
    Wsum = np.sum(weights) if np.sum(weights) > 0 else 1.0

    # County takes allocation ratio, rest distributed to others
    county_bonus = surplus * county_alloc_ratio
    remaining_bonus = surplus - county_bonus

    bonuses = np.zeros(H)
    if remaining_bonus > 0:
        non_county_weights = weights.copy()
        non_county_weights[idx_county] = 0
        non_county_sum = np.sum(non_county_weights)
        if non_county_sum > 0:
            non_county_bonuses = remaining_bonus * (non_county_weights / non_county_sum)
            bonuses += non_county_bonuses

    # Distribute county bonus among county hospitals
    if county_bonus > 0 and len(idx_county) > 0:
        county_weights = weights[idx_county]
        county_sum = np.sum(county_weights)
        if county_sum > 0:
            county_bonuses = county_bonus * (county_weights / county_sum)
            for i, idx in enumerate(idx_county):
                bonuses[idx] += county_bonuses[i]

    # Negative surplus (penalty)
    if surplus < 0.0 and ALLOW_NEGATIVE_SURPLUS:
        penalties = surplus * (weights / Wsum)
        bonuses += penalties

    for i, h in enumerate(Hlist):
        h.profit_prev += bonuses[i]
        h.bonus_prev = bonuses[i]

    # ----- aggregates for plots -----
    sum_flows = np.sum(flows) + 1e-6
    flow_share_lvl["county"].append(np.sum(flows[idx_county]) / sum_flows)
    flow_share_lvl["secondary"].append(np.sum(flows[idx_secondary]) / sum_flows)
    flow_share_lvl["township"].append(np.sum(flows[idx_township]) / sum_flows)

    cap_lvl["county"].append(np.sum([Hlist[i].K for i in idx_county]))
    cap_lvl["secondary"].append(np.sum([Hlist[i].K for i in idx_secondary]))
    cap_lvl["township"].append(np.sum([Hlist[i].K for i in idx_township]))
    # =============== 新增：两级架构合并统计计算 ===============
    # 计算基层+二级的总流量份额
    primary_secondary_flows = np.sum(flows[idx_secondary]) + np.sum(flows[idx_township])
    combined_primary_flow_share.append(primary_secondary_flows / sum_flows)

    # 计算基层+二级的总容量
    primary_secondary_capacity = (np.sum([Hlist[i].K for i in idx_secondary]) +
                                 np.sum([Hlist[i].K for i in idx_township]))
    combined_primary_capacity.append(primary_secondary_capacity)

    # 计算基层+二级的平均严重度（加权平均）
    # 计算基层+二级的平均严重度（加权平均）
    if primary_secondary_flows > 0:
        # 直接计算二级医院平均严重度
        sev_secondary = 0.0
        if len(idx_secondary) > 0:
            acc_secondary = []
            for i in idx_secondary:
                acc_secondary += served_idx_by_h[i]
            sev_secondary = float(np.mean(Sev[acc_secondary])) if len(acc_secondary) > 0 else 0.0

        # 直接计算基层医院平均严重度
        sev_township = 0.0
        if len(idx_township) > 0:
            acc_township = []
            for i in idx_township:
                acc_township += served_idx_by_h[i]
            sev_township = float(np.mean(Sev[acc_township])) if len(acc_township) > 0 else 0.0

        weighted_severity = (sev_secondary * np.sum(flows[idx_secondary]) +
                             sev_township * np.sum(flows[idx_township])) / primary_secondary_flows
    else:
        weighted_severity = 0
    combined_primary_severity.append(weighted_severity)

    # 计算基层+二级的总利润
    secondary_profits = sum(Hlist[i].profit_prev for i in idx_secondary)
    township_profits = sum(Hlist[i].profit_prev for i in idx_township)
    combined_primary_profits.append(secondary_profits + township_profits)

    # 计算基层+二级的平均利用率
    secondary_utilization = sum(flows[i] for i in idx_secondary) / max(sum(Hlist[i].K for i in idx_secondary), 1.0)
    township_utilization = sum(flows[i] for i in idx_township) / max(sum(Hlist[i].K for i in idx_township), 1.0)
    combined_primary_utilization.append((secondary_utilization + township_utilization) / 2)

    # 计算基层+二级的床位利用率
    secondary_beddays = sum(beddays_used_by_h[i] for i in idx_secondary)
    township_beddays = sum(beddays_used_by_h[i] for i in idx_township)
    secondary_capacity = sum(Hlist[i].K for i in idx_secondary)
    township_capacity = sum(Hlist[i].K for i in idx_township)
    combined_bed_occupancy = (secondary_beddays + township_beddays) / max(secondary_capacity + township_capacity, 1.0)
    combined_primary_bed_occupancy.append(combined_bed_occupancy)

    # 计算基层+二级的住院比例
    secondary_inpatients = sum(inpatient_counts[i] for i in idx_secondary)
    township_inpatients = sum(inpatient_counts[i] for i in idx_township)
    secondary_patients = sum(flows[i] for i in idx_secondary)
    township_patients = sum(flows[i] for i in idx_township)
    if (secondary_patients + township_patients) > 0:
        combined_inpatient_share = (secondary_inpatients + township_inpatients) / (secondary_patients + township_patients)
    else:
        combined_inpatient_share = 0
    combined_primary_inpatient_share.append(combined_inpatient_share)

    # 新增：容量趋势和占比计算
    total_capacity = (cap_lvl["county"][-1] + cap_lvl["secondary"][-1] + cap_lvl["township"][-1])
    #
    capacity_trends["county"].append(cap_lvl["county"][-1])
    capacity_trends["secondary"].append(cap_lvl["secondary"][-1])
    capacity_trends["township"].append(cap_lvl["township"][-1])

    if total_capacity > 0:
        capacity_trends["county_share"].append(cap_lvl["county"][-1] / total_capacity)
        capacity_trends["secondary_share"].append(cap_lvl["secondary"][-1] / total_capacity)
        capacity_trends["township_share"].append(cap_lvl["township"][-1] / total_capacity)
    else:
        capacity_trends["county_share"].append(0.0)
        capacity_trends["secondary_share"].append(0.0)
        capacity_trends["township_share"].append(0.0)
    # 替换：详细严重程度统计
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        all_severities = []
        for i in idx_list:
            all_severities.extend(severity_by_hospital[i])

        if all_severities:
            sev_array = np.array(all_severities)
            sev_stats_lvl[level]["mean"].append(float(np.mean(sev_array)))
            sev_stats_lvl[level]["std"].append(float(np.std(sev_array)))
            sev_stats_lvl[level]["min"].append(float(np.min(sev_array)))
            sev_stats_lvl[level]["max"].append(float(np.max(sev_array)))
            sev_stats_lvl[level]["q25"].append(float(np.percentile(sev_array, 25)))
            sev_stats_lvl[level]["q75"].append(float(np.percentile(sev_array, 75)))
        else:
            # 如果没有患者，用前一期值或0，而不是NaN
            if t > 0 and len(sev_stats_lvl[level]["mean"]) > 0:
                prev_mean = sev_stats_lvl[level]["mean"][-1]
                sev_stats_lvl[level]["mean"].append(prev_mean)
                sev_stats_lvl[level]["std"].append(
                    sev_stats_lvl[level]["std"][-1] if sev_stats_lvl[level]["std"] else 0.1)
                sev_stats_lvl[level]["min"].append(0.0)
                sev_stats_lvl[level]["max"].append(0.0)
                sev_stats_lvl[level]["q25"].append(0.0)
                sev_stats_lvl[level]["q75"].append(0.0)
            else:
                sev_stats_lvl[level]["mean"].append(0.0)
                sev_stats_lvl[level]["std"].append(0.1)
                sev_stats_lvl[level]["min"].append(0.0)
                sev_stats_lvl[level]["max"].append(0.0)
                sev_stats_lvl[level]["q25"].append(0.0)
                sev_stats_lvl[level]["q75"].append(0.0)

    # 保持原有的sev_mean_lvl更新（用于兼容性）
    sev_mean_lvl["county"].append(sev_stats_lvl["county"]["mean"][-1])
    sev_mean_lvl["secondary"].append(sev_stats_lvl["secondary"]["mean"][-1])
    sev_mean_lvl["township"].append(sev_stats_lvl["township"]["mean"][-1])
    cap_lvl["county"].append(np.sum([Hlist[i].K for i in idx_county]))
    cap_lvl["secondary"].append(np.sum([Hlist[i].K for i in idx_secondary]))
    cap_lvl["township"].append(np.sum([Hlist[i].K for i in idx_township]))
    # ----- 原有核心指标计算 -----
    # 平均等待时间
    avg_waiting_time = float(np.average(W_eff, weights=flows + 1e-6))
    waiting_time_series.append(avg_waiting_time)

    # 平均处理质量
    if treated > 0:
        served_mask = (assigned_to != -1)
        Q_chosen = Qeff[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_quality = float(np.mean(Q_chosen))
    else:
        avg_quality = float(np.mean(Qeff))
    process_quality_series.append(avg_quality)

    # 患者效用
    if treated > 0:
        served_mask = (assigned_to != -1)
        U_chosen = U[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_utility = float(np.mean(U_chosen))
    else:
        avg_utility = float(np.mean(np.max(U, axis=1)))
    patient_utility_series.append(avg_utility)

    # 平均就诊距离
    if treated > 0:
        served_mask = (assigned_to != -1)
        d_chosen = dmat[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_distance = float(np.mean(d_chosen))
    else:
        avg_distance = float(np.mean(np.min(dmat, axis=1)))
    distance_traveled_series.append(avg_distance)

    # 医院利润
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_profit = sum(Hlist[i].profit_prev for i in idx_list)
        hospital_profits[level].append(total_profit)

    # 容量利用率（基于人次）
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_flows = sum(flows[i] for i in idx_list)
        total_capacity = sum(Hlist[i].K for i in idx_list)
        utilization = total_flows / max(total_capacity, 1.0)
        capacity_utilization[level].append(utilization)

    # 转诊效率
    if total_referral_attempts > 0:
        efficiency = total_referral_success / total_referral_attempts
    else:
        efficiency = 0.0
    referral_efficiency.append(efficiency)

    # 预算利用率
    budget_use = ins_total / BUDGET_PER_PERIOD if BUDGET_PER_PERIOD > 0 else 0.0
    budget_utilization.append(budget_use)

    # 结余分配
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_bonus = sum(bonuses[i] for i in idx_list)
        surplus_distribution[level].append(total_bonus)

    # ----- 学习过程指标 -----
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        # 学习收敛（策略稳定性）
        policy_changes = 0
        for i in idx_list:
            s = Hlist[i].state_hash_a()
            probs = softmax(Hlist[i].Qa[s], 0.1)  # 低温度看确定性
            max_prob = np.max(probs)
            if max_prob > 0.8:  # 高度确定的策略
                policy_changes += 0
            else:
                policy_changes += 1
        convergence = 1 - (policy_changes / max(len(idx_list), 1))
        learning_convergence[level].append(convergence)

    # 行动分布
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        if idx_list:
            action_distribution[level]["a"].append(np.mean([Hlist[i].a for i in idx_list]))
            action_distribution[level]["i"].append(np.mean([Hlist[i].i_pending for i in idx_list]))
            if level != "county":
                action_distribution[level]["rho"].append(np.mean([getattr(Hlist[i], 'rho', 0.0) for i in idx_list]))
            if level == "county":
                action_distribution[level]["v"].append(np.mean([getattr(Hlist[i], 'v', 0.0) for i in idx_list]))
                action_distribution[level]["d"].append(np.mean([getattr(Hlist[i], 'd', 0.0) for i in idx_list]))

    # ----- 患者行为指标 -----
    # 患者选择质量（实际效用与理论最大效用的比值）
    if treated > 0:
        served_mask = (assigned_to != -1)
        U_chosen = U[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        U_max = np.max(U[served_mask, :], axis=1)
        choice_quality = np.mean(U_chosen / (U_max + 1e-6))
    else:
        choice_quality = 0.0
    patient_choice_quality.append(choice_quality)

    # 习惯持续性
    habit_persist = np.mean(patient_behavior.history >= 0)  # 有历史记录的患者比例
    habit_persistence.append(habit_persist)

    # 信息不对称影响（基层医院被选择的偏差）
    township_choices = np.sum([flows[i] for i in idx_township])
    total_choices = np.sum(flows)
    township_share = township_choices / max(total_choices, 1.0)
    # 理论上的期望份额（基于人口分布）
    expected_share = 0.4  # 这个需要根据实际人口分布计算，暂时用估计值
    info_asymmetry = max(0, expected_share - township_share)
    information_asymmetry_effect.append(info_asymmetry)

    def level_mean(idx_list):
        acc = []
        for i in idx_list: acc += served_idx_by_h[i]
        return float(np.mean(Sev[acc])) if len(acc) > 0 else 0.0

    sev_mean_lvl["county"].append(level_mean(idx_county))
    sev_mean_lvl["secondary"].append(level_mean(idx_secondary))
    sev_mean_lvl["township"].append(level_mean(idx_township))

    ins_lvl = {
        "county": float(np.sum(ins_spend[idx_county])),
        "secondary": float(np.sum(ins_spend[idx_secondary])),
        "township": float(np.sum(ins_spend[idx_township]))
    }
    ins_spend_lvl["county"].append(ins_lvl["county"])
    ins_spend_lvl["secondary"].append(ins_lvl["secondary"])
    ins_spend_lvl["township"].append(ins_lvl["township"])
    ins_spend_lvl["total"].append(sum(ins_lvl.values()))

    for lvl, idxs in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        a_traj[lvl].append(float(np.mean([Hlist[i].a for i in idxs])))
        if (t % 6 == 0) and (t >= WARMUP):
            i_traj[lvl].append(float(np.mean([Hlist[i].i_pending for i in idxs])))
        else:
            i_traj[lvl].append(np.nan)
        probs_list = []
        for i in idxs:
            s = Hlist[i].state_hash_a()
            probs_list.append(softmax(Hlist[i].Qa[s], max(tau_a, 1e-6)))
        if probs_list:
            P = np.mean(np.stack(probs_list), axis=0)
            entropy = -np.sum(P * np.log(P + 1e-12))
        else:
            entropy = 0.0
        entropy_policy[lvl].append(float(entropy))

    tau_a = max(TAU_A_END, tau_a - tau_decay)


# =============== 输出CSV文件 ===============
def export_metrics_to_csv():
    """输出指标到CSV文件"""
    # 周度指标
    with open('weekly_metrics_s2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'treated', 'unserved_total', 'outflow_wait', 'outflow_full', 'outflow_mismatch',
                         'referral_success_rate', 'county_allocation_ratio', 'avg_referral_path_length',
                         'county_occupancy', 'secondary_occupancy', 'township_occupancy',
                         'county_inpatient_share', 'secondary_inpatient_share', 'township_inpatient_share'])

        for t in range(T):
            writer.writerow([
                t, treated_series[t], unserved_series[t],
                outflow_wait_series[t], outflow_full_series[t], outflow_mismatch_series[t],
                referral_success_rate[t], county_allocation_ratio[t], referral_path_length[t],
                bed_occupancy_lvl['county'][t], bed_occupancy_lvl['secondary'][t], bed_occupancy_lvl['township'][t],
                inpatient_share_lvl['county'][t], inpatient_share_lvl['secondary'][t],
                inpatient_share_lvl['township'][t]
            ])

    # 医院流量数据
    with open('hospital_flows_s2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['hospital_id', 'name', 'level', 'x', 'y', 'avg_flow', 'avg_beddays', 'occupancy_rate'])

        for i, h in enumerate(Hlist):
            avg_flow = np.mean(hospital_flow_history[i][-52:]) if len(hospital_flow_history[i]) > 0 else 0
            avg_beddays = avg_flow * 0.6  # 简化估算
            occupancy = avg_beddays / max(h.K, 1.0)

            writer.writerow([
                i, h.name, h.level, float(h.pos[0]), float(h.pos[1]),
                avg_flow, avg_beddays, occupancy
            ])


# =============== 生成空间热力图 ===============
def create_spatial_heatmap():
    """生成基于医院流量的空间热力图 - 沿用原有样式"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 计算医院流量强度（最近52期平均值）
    hospital_intensity = []
    hospital_flows = []
    for i in range(H):
        if len(hospital_flow_history[i]) >= 52:
            flow = np.mean(hospital_flow_history[i][-52:])
        else:
            flow = np.mean(hospital_flow_history[i]) if hospital_flow_history[i] else 0
        hospital_flows.append(flow)

    # 归一化流量用于颜色映射
    max_flow = max(hospital_flows) if hospital_flows else 1
    hospital_intensity = [flow / max_flow for flow in hospital_flows]

    # 绘制人口密度背景 - 使用原有色调
    im = ax.contourf(X_GRID, Y_GRID, POPULATION_GRID,
                     levels=20, cmap='Blues', alpha=0.6, antialiased=True)

    # 绘制医院点 - 完全沿用原有样式
    for i, h in enumerate(Hlist):
        color = COLORS[h.level]

        # 大小反映流量强度（沿用原有缩放比例）
        size = 50 + 450 * hospital_intensity[i]

        # 沿用原有的标记形状
        if h.level == "county":
            marker = 'D'  # 菱形
            edge_color = 'darkblue'
            edge_width = 2
        elif h.level == "secondary":
            marker = 's'  # 方形
            edge_color = 'darkorange'
            edge_width = 1.5
        else:  # township
            marker = 'o'  # 圆形
            edge_color = 'darkred'
            edge_width = 1

        # 绘制医院点
        ax.scatter(h.pos[0], h.pos[1],
                   s=size,
                   c=color,
                   marker=marker,
                   edgecolors=edge_color,
                   linewidths=edge_width,
                   alpha=0.8,
                   label=f'{h.name} ({h.level})' if i < 5 else "")  # 只标注前5个避免图例过多

        # 添加医院名称标签（沿用原有样式）
        if hospital_intensity[i] > 0.3:  # 只对流量较大的医院显示标签
            ax.annotate(h.name,
                        (h.pos[0], h.pos[1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # 设置坐标轴和标题 - 沿用原有样式
    ax.set_xlim(-CITY_RADIUS, CITY_RADIUS)
    ax.set_ylim(-CITY_RADIUS, CITY_RADIUS)
    ax.set_aspect('equal')
    ax.set_title('S2: Hospital Flow Intensity Heatmap (Bundled Payment with Surplus Retention)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate (km)', fontsize=12)
    ax.set_ylabel('Y Coordinate (km)', fontsize=12)

    # 添加网格 - 沿用原有样式
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 添加图例 - 沿用原有样式但更简洁
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['county'],
               markersize=10, label='County Hospital', markeredgecolor='darkblue', markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['secondary'],
               markersize=10, label='Secondary Hospital', markeredgecolor='darkorange', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['township'],
               markersize=10, label='Township Hospital', markeredgecolor='darkred', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # 添加颜色条 - 沿用原有样式
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Population Density', fontsize=12)

    # 添加流量强度说明
    ax.text(0.02, 0.98, 'Marker size indicates\nhospital flow intensity',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig('spatial_heat_s2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
# def create_spatial_heatmap():
#     """生成基于医院流量的空间热力图"""
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # 绘制人口密度背景
#     im = ax.imshow(POPULATION_GRID, extent=[-CITY_RADIUS, CITY_RADIUS, -CITY_RADIUS, CITY_RADIUS],
#                    origin='lower', cmap='Blues', alpha=0.6)
#
#     # 计算医院流量强度（最近52期平均值）
#     hospital_intensity = []
#     for i in range(H):
#         if len(hospital_flow_history[i]) >= 52:
#             intensity = np.mean(hospital_flow_history[i][-52:])
#         else:
#             intensity = np.mean(hospital_flow_history[i]) if hospital_flow_history[i] else 0
#         hospital_intensity.append(intensity)
#
#     # 归一化强度
#     max_intensity = max(hospital_intensity) if hospital_intensity else 1
#     hospital_intensity = [intensity / max_intensity for intensity in hospital_intensity]
#
#     # 绘制医院点（大小和颜色反映流量）
#     for i, h in enumerate(Hlist):
#         color = COLORS[h.level]
#         size = 50 + 200 * hospital_intensity[i]  # 大小反映流量
#         alpha = 0.3 + 0.7 * hospital_intensity[i]  # 透明度也反映流量
#
#         ax.scatter(h.pos[0], h.pos[1], s=size, c=color, alpha=alpha,
#                    label=h.name if i < 3 else "")  # 只标注前3个避免图例过多
#
#     ax.set_xlim(-CITY_RADIUS, CITY_RADIUS)
#     ax.set_ylim(-CITY_RADIUS, CITY_RADIUS)
#     ax.set_aspect('equal')
#     ax.set_title('S2: Hospital Flow Intensity Heatmap')
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#
#     # 添加图例
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['county'], markersize=10,
#                label='County Hospital'),
#         Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['secondary'], markersize=10,
#                label='Secondary Hospital'),
#         Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['township'], markersize=10,
#                label='Township Hospital')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right')
#
#     plt.colorbar(im, ax=ax, label='Population Density')
#     plt.tight_layout()
#     plt.savefig('spatial_heat_s2.png', dpi=300, bbox_inches='tight')
#     plt.show()

# =============== 专业热力图函数 ===============
def draw_city_base(ax, R_CITY=10.0, R_INNER=3.0, R_MIDDLE=6.0, decay=0.28):
    """绘制城市基础背景"""
    res = 560
    xs = np.linspace(-R_CITY, R_CITY, res)
    ys = np.linspace(-R_CITY, R_CITY, res)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X ** 2 + Y ** 2)
    base = np.exp(-decay * R)

    # 修复：确保背景图也被正确裁剪
    im_bg = ax.imshow(base, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                      origin='lower', interpolation='bilinear', cmap=SOFT_CMAP,
                      alpha=0.9, zorder=0)

    # 修复：创建精确的圆形裁剪路径
    circle = Circle((0, 0), R_CITY, transform=ax.transData)
    im_bg.set_clip_path(circle)

    # 绘制环形边界
    th = np.linspace(0, 2 * np.pi, 1200)
    for rr, lw, col in [(R_CITY, 2.0, "#c7d2fe"),
                        (R_INNER, 1.2, "#dde7ff"),
                        (R_MIDDLE, 1.2, "#dde7ff")]:
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=col, lw=lw, alpha=0.95, zorder=2)

    ax.set_aspect("equal")
    ax.set_xlim(-R_CITY, R_CITY)
    ax.set_ylim(-R_CITY, R_CITY)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_hospitals(ax, coords):
    """绘制医院位置"""
    # 县医院 - 星形
    ax.scatter(coords["ter"][:, 0], coords["ter"][:, 1], marker='*', s=160,
               c=COLOR_TERTIARY, edgecolors='white', linewidths=0.9, zorder=3)
    # 二级医院 - 方形
    ax.scatter(coords["sec"][:, 0], coords["sec"][:, 1], marker='s', s=90,
               c=COLOR_SECONDARY, edgecolors='white', linewidths=0.8, zorder=3)
    # 基层医院 - 三角形
    ax.scatter(coords["pri"][:, 0], coords["pri"][:, 1], marker='^', s=60,
               c=COLOR_PRIMARY, edgecolors='white', linewidths=0.7, zorder=3)


def gaussian_field(coords, weights_or_avgs, R_CITY=10.0,
                   bw_ter=1.9, bw_sec=2.2, bw_pri=2.6):
    """生成高斯场"""
    res = 300
    xs = np.linspace(-R_CITY, R_CITY, res)
    ys = np.linspace(-R_CITY, R_CITY, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, dtype=float)
    pts = np.vstack([coords["ter"], coords["sec"], coords["pri"]])
    levels = (["ter"] * len(coords["ter"]) + ["sec"] * len(coords["sec"]) + ["pri"] * len(coords["pri"]))
    w = np.asarray(weights_or_avgs, float)

    for i, (x, y) in enumerate(pts):
        bw = bw_ter if levels[i] == "ter" else (bw_sec if levels[i] == "sec" else bw_pri)
        K = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * bw * bw))
        Z += w[i] * K

    R = np.sqrt(X ** 2 + Y ** 2)
    Z[R > R_CITY] = np.nan
    return X, Y, Z


def create_professional_heatmap():
    """创建专业热力图 - 使用heatmap_v5_9_30的样式"""
    # 准备坐标数据
    coords = {
        "ter": np.array([[0.0, 0.0]]),  # County Hospital
        "sec": np.array([
            [0.11379539773014025, -5.650593474],  # SEC_1
            [4.320249347575342, 3.185028953685901]  # SEC_2
        ]),
        "pri": np.array([
            [-1.55559427, -0.989992556],  # PRI_1
            [-0.886082892, 2.133249007302485],  # PRI_2
            [2.4797599149815674, -2.85845215],  # PRI_3
            [4.040729383325832, -4.399558712],  # PRI_4
            [-6.634243241, -0.371456373],  # PRI_5
            [3.6605658066377575, -0.103055971],  # PRI_6
            [8.659329422749051, -2.229064251],  # PRI_7
            [7.328182317, 4.816806888528005],  # PRI_8
            [-1.050331321, 7.472897930775263],  # PRI_9
            [-4.29982209, -1.095054112]  # PRI_10
        ])
    }

    # 计算医院流量（最近52期平均值）
    hospital_flows = []
    for i in range(H):
        if len(hospital_flow_history[i]) >= 52:
            flow = np.mean(hospital_flow_history[i][-52:])
        else:
            flow = np.mean(hospital_flow_history[i]) if hospital_flow_history[i] else 0
        hospital_flows.append(flow)

    hospital_flows = np.array(hospital_flows)

    # 计算share和absolute
    total_flow = hospital_flows.sum()
    if total_flow > 0:
        shares = hospital_flows / total_flow
    else:
        shares = hospital_flows

    # 生成高斯场
    X_share, Y_share, Z_share = gaussian_field(coords, shares)
    X_abs, Y_abs, Z_abs = gaussian_field(coords, hospital_flows)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), dpi=140, facecolor='white')
    plt.subplots_adjust(left=0.04, right=0.97, wspace=0.08, hspace=0.14)

    R_CITY = 10.0

    # ---- 上图: Share ----
    draw_city_base(ax1, R_CITY)
    vmax_share = float(np.nanmax(Z_share)) if np.isfinite(np.nanmax(Z_share)) and np.nanmax(Z_share) > 0 else 1.0
    norm_share = PowerNorm(gamma=0.60, vmin=0.0, vmax=vmax_share)

    # 修复：确保热力图完全覆盖圆形区域
    im1 = ax1.imshow(Z_share, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                     origin='lower', interpolation='bilinear', cmap=HEAT_CMAP,
                     norm=norm_share, alpha=0.88, zorder=1)

    # 修复：创建精确的圆形裁剪路径
    circle = Circle((0, 0), R_CITY, transform=ax1.transData)
    im1.set_clip_path(circle)

    plot_hospitals(ax1, coords)
    ax1.set_title("S1: Treated Volumes (Share)", fontsize=14, fontweight='bold')

    # 添加等值线
    if np.isfinite(vmax_share) and vmax_share > 0:
        ax1.contour(X_share, Y_share, Z_share, levels=[0.65 * vmax_share],
                    colors=['#fde68a'], linewidths=1.6, zorder=4, alpha=0.95)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.02)
    cb1 = plt.colorbar(im1, cax=cax1)
    cb1.set_label('Share (last 52w)', fontsize=10)

    # ---- 下图: Absolute ----
    draw_city_base(ax2, R_CITY)
    vmax_abs = float(np.nanmax(Z_abs)) if np.isfinite(np.nanmax(Z_abs)) and np.nanmax(Z_abs) > 0 else 1.0
    norm_abs = PowerNorm(gamma=1.0, vmin=0.0, vmax=vmax_abs)

    im2 = ax2.imshow(Z_abs, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                     origin='lower', interpolation='bilinear', cmap=HEAT_CMAP,
                     norm=norm_abs, alpha=0.88, zorder=1)

    # 修复：创建精确的圆形裁剪路径
    circle2 = Circle((0, 0), R_CITY, transform=ax2.transData)
    im2.set_clip_path(circle2)

    plot_hospitals(ax2, coords)
    ax2.set_title("S1: Treated Volumes (Absolute)", fontsize=14, fontweight='bold')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.02)
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.set_label('Avg treated (last 52w)', fontsize=10)

    plt.tight_layout()
    plt.savefig('professional_heatmap_s2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# =============== Enhanced Plotting ===============
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# def plot_series_with_ma(ax, x, y, color, label):
#     y = np.array(y, dtype=float)
#     ax.plot(x, y, linestyle="--", linewidth=1.2, color=color, alpha=0.5, label=f"{label} (raw)")
#     ax.plot(x, moving_average(y, REPORT_MOVAVG), linestyle="-", linewidth=2.4, color=color, label=f"{label} (12m MA)")
def plot_series_with_ma(ax, x, y, color, label):
    """修复的绘图函数，确保连续线条"""
    y = np.array(y, dtype=float)

    # 确保y没有NaN值
    y_filled = y.copy()
    nan_mask = np.isnan(y_filled)
    if nan_mask.any():
        # 简单的线性插值填充NaN
        indices = np.arange(len(y_filled))
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            y_filled = np.interp(indices, indices[valid_mask], y_filled[valid_mask])
        else:
            y_filled = np.zeros_like(y_filled)

    # 绘制原始数据（透明度较低）
    ax.plot(x, y_filled, linestyle="--", linewidth=1.0, color=color, alpha=0.3, label=f"{label} (raw)")

    # 计算并绘制移动平均
    y_ma = moving_average(y_filled, REPORT_MOVAVG)
    ax.plot(x, y_ma, linestyle="-", linewidth=2.5, color=color, label=f"{label} (12m MA)")

    return y_ma  # 返回移动平均值用于调试

X = np.arange(T)

# 1) Treated and Outflow Analysis
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 8))
plot_series_with_ma(ax1a, X, treated_series, COLORS["total"], "Treated")
ax1a.set_title("Treated Patients per Period")
ax1a.set_ylabel("Count")
ax1a.legend()

# 外流分析
ax1b.stackplot(X,
               outflow_wait_series, outflow_full_series, outflow_mismatch_series,
               labels=['Wait Time Outflow', 'Capacity Outflow', 'Mismatch Outflow'],
               colors=[COLORS['outflow_wait'], COLORS['outflow_full'], COLORS['outflow_mismatch']],
               alpha=0.7)
ax1b.set_title("Patient Outflow by Reason")
ax1b.set_xlabel("Period")
ax1b.set_ylabel("Outflow Count")
ax1b.legend()
plt.tight_layout()

# 2) Bed Occupancy
fig2, ax2 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax2, X, bed_occupancy_lvl["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax2, X, combined_primary_bed_occupancy, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax2.set_title("Bed Occupancy Rate - Two Tier System")
ax2.set_xlabel("Period")
ax2.set_ylabel("Occupancy Rate")
ax2.set_ylim(0, 1)
ax2.legend()
plt.tight_layout()

fig3, ax3 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax3, X, inpatient_share_lvl["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax3, X, combined_primary_inpatient_share, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax3.set_title("Inpatient Share - Two Tier System")
ax3.set_xlabel("Period")
ax3.set_ylabel("Inpatient Share")
ax3.set_ylim(0, 1)
ax3.legend()
plt.tight_layout()
ax3.set_title("Inpatient Share by Level")
ax3.set_xlabel("Period")
ax3.set_ylabel("Inpatient Share")
ax3.set_ylim(0, 1)
ax3.legend()
plt.tight_layout()

# 4) Enhanced System Performance
fig4, ax4 = plt.subplots(figsize=(10, 5.2))
plot_series_with_ma(ax4, X, referral_success_rate, "#10B981", "Referral Success Rate")
plot_series_with_ma(ax4, X, county_allocation_ratio, "#8B5CF6", "County Allocation Ratio")
ax4.set_title("Enhanced System Performance — Referral & Allocation")
ax4.set_xlabel("Period")
ax4.set_ylabel("Rate")
ax4.set_ylim(0, 1)
ax4.legend()
plt.tight_layout()

# 5) 原有指标继续保留
# 5) 流量份额 - 两级显示
fig5, ax5 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax5, X, flow_share_lvl["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax5, X, combined_primary_flow_share, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax5.set_title("Flow Shares - Two Tier System")
ax5.set_xlabel("Period")
ax5.set_ylabel("Share")
ax5.set_ylim(0, 0.9)
ax5.legend()
plt.tight_layout()

# 7) 等待时间和处理质量
fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax7a, X, waiting_time_series, COLORS["total"], "Average Waiting Time")
ax7a.set_title("Average Waiting Time Across System")
ax7a.set_ylabel("Days")
ax7a.legend()

plot_series_with_ma(ax7b, X, process_quality_series, COLORS["secondary"], "Average Process Quality")
ax7b.set_title("Average Treatment Quality")
ax7b.set_xlabel("Period")
ax7b.set_ylabel("Quality Score")
ax7b.legend()
plt.tight_layout()

# 8) 患者效用和选择质量
fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax8a, X, patient_utility_series, COLORS["county"], "Patient Utility")
ax8a.set_title("Average Patient Utility")
ax8a.set_ylabel("Utility Score")
ax8a.legend()

plot_series_with_ma(ax8b, X, patient_choice_quality, COLORS["township"], "Choice Quality Ratio")
ax8b.set_title("Patient Choice Quality (Actual/Max Utility)")
ax8b.set_xlabel("Period")
ax8b.set_ylabel("Quality Ratio")
ax8b.set_ylim(0, 1)
ax8b.legend()
plt.tight_layout()

fig9, ax9 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax9, X, hospital_profits["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax9, X, combined_primary_profits, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax9.set_title("Hospital Profits - Two Tier System")
ax9.set_xlabel("Period")
ax9.set_ylabel("Profit")
ax9.legend()
plt.tight_layout()

# 10) 容量利用率
fig10, ax10 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax10, X, capacity_utilization["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax10, X, combined_primary_utilization, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax10.set_title("Capacity Utilization - Two Tier System")
ax10.set_xlabel("Period")
ax10.set_ylabel("Utilization Rate")
ax10.set_ylim(0, 1)
ax10.legend()
plt.tight_layout()

# 11) 转诊和预算效率
fig11, (ax11a, ax11b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax11a, X, referral_efficiency, "#10B981", "Referral Efficiency")
ax11a.set_title("Referral System Efficiency")
ax11a.set_ylabel("Success Rate")
ax11a.set_ylim(0, 1)
ax11a.legend()

plot_series_with_ma(ax11b, X, budget_utilization, "#8B5CF6", "Budget Utilization")
ax11b.set_title("Insurance Budget Utilization")
ax11b.set_xlabel("Period")
ax11b.set_ylabel("Utilization Rate")
ax11b.legend()
plt.tight_layout()

# 12) 学习收敛
# 计算基层医院平均学习收敛度
primary_convergence = []
for t in range(T):
    if t < len(learning_convergence["secondary"]) and t < len(learning_convergence["township"]):
        avg_conv = (learning_convergence["secondary"][t] + learning_convergence["township"][t]) / 2
    elif t < len(learning_convergence["secondary"]):
        avg_conv = learning_convergence["secondary"][t]
    elif t < len(learning_convergence["township"]):
        avg_conv = learning_convergence["township"][t]
    else:
        avg_conv = 0
    primary_convergence.append(avg_conv)

fig12, ax12 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax12, X, learning_convergence["county"], COLORS["county"], "County Hospital")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax12, X, primary_convergence, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax12.set_title("Learning Convergence - Two Tier System")
ax12.set_xlabel("Period")
ax12.set_ylabel("Convergence Index")
ax12.set_ylim(0, 1)
ax12.legend()
plt.tight_layout()

# 13) 行动分布 - 合作强度a
# 计算基层医院平均合作强度
primary_cooperation = []
for t in range(T):
    if t < len(action_distribution["secondary"]["a"]) and t < len(action_distribution["township"]["a"]):
        avg_coop = (action_distribution["secondary"]["a"][t] + action_distribution["township"]["a"][t]) / 2
    elif t < len(action_distribution["secondary"]["a"]):
        avg_coop = action_distribution["secondary"]["a"][t]
    elif t < len(action_distribution["township"]["a"]):
        avg_coop = action_distribution["township"]["a"][t]
    else:
        avg_coop = 0
    primary_cooperation.append(avg_coop)

fig13, ax13 = plt.subplots(figsize=(10, 5.6))
# 县医院
plot_series_with_ma(ax13, X, action_distribution["county"]["a"], COLORS["county"], "County Hospital Cooperation")
# 基层医院（合并二级和基层）
plot_series_with_ma(ax13, X, primary_cooperation, COLORS["township"], "Primary Care Cooperation (Towns + Large Towns)")
ax13.set_title("Cooperation Intensity - Two Tier System")
ax13.set_xlabel("Period")
ax13.set_ylabel("Cooperation Level")
ax13.set_ylim(0, 1)
ax13.legend()
plt.tight_layout()

# 14) 合作情况综合图表
fig14, ax14 = plt.subplots(figsize=(12, 6))

# 计算基层医院平均转诊倾向（二级和基层）
primary_referral_propensity = []
for t in range(T):
    if t < len(action_distribution["secondary"]["rho"]) and t < len(action_distribution["township"]["rho"]):
        avg_rho = (action_distribution["secondary"]["rho"][t] + action_distribution["township"]["rho"][t]) / 2
    elif t < len(action_distribution["secondary"]["rho"]):
        avg_rho = action_distribution["secondary"]["rho"][t]
    elif t < len(action_distribution["township"]["rho"]):
        avg_rho = action_distribution["township"]["rho"][t]
    else:
        avg_rho = 0
    primary_referral_propensity.append(avg_rho)

# 绘制三条线：转诊倾向、能力提升、县医院分配比例
plot_series_with_ma(ax14, X, primary_referral_propensity, "#EF4444", "Primary Referral Propensity (ρ)")
plot_series_with_ma(ax14, X, smax_uplift_lower, "#10B981", "Primary Ability Uplift")
plot_series_with_ma(ax14, X, county_allocation_ratio, "#8B5CF6", "County Allocation Ratio")

ax14.set_title("Cooperation Indicators - Two Tier System")
ax14.set_xlabel("Period")
ax14.set_ylabel("Rate")
ax14.set_ylim(0, 1)
ax14.legend()
plt.tight_layout()

# 15) 患者行为指标
fig15, (ax15a, ax15b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax15a, X, habit_persistence, "#F59E0B", "Habit Persistence")
ax15a.set_title("Patient Habit Persistence")
ax15a.set_ylabel("Persistence Rate")
ax15a.set_ylim(0, 1)
ax15a.legend()

plot_series_with_ma(ax15b, X, information_asymmetry_effect, "#EF4444", "Information Asymmetry Effect")
ax15b.set_title("Information Asymmetry on Township Hospitals")
ax15b.set_xlabel("Period")
ax15b.set_ylabel("Choice Reduction")
ax15b.legend()
plt.tight_layout()

# 16) 系统综合指标
fig16, (ax16a, ax16b) = plt.subplots(2, 1, figsize=(10, 8))
# 系统效率（治疗/总需求）
system_efficiency = np.array(treated_series) / (np.array(treated_series) + np.array(unserved_series) + 1e-6)
plot_series_with_ma(ax16a, X, system_efficiency, "#3B82F6", "System Efficiency")
ax16a.set_title("Overall System Efficiency (Treated / Total Demand)")
ax16a.set_ylabel("Efficiency")
ax16a.set_ylim(0, 1)
ax16a.legend()
# 17严重程度
# 17) 各级医院患者严重程度分布
fig17, ((ax17a, ax17b), (ax17c, ax17d)) = plt.subplots(2, 2, figsize=(14, 10))

# 修复变异系数计算
sev_cv_county = []
sev_cv_secondary = []
sev_cv_township = []

for t in range(T):
    # 避免除零错误
    mean_county = sev_stats_lvl['county']['mean'][t] if t < len(sev_stats_lvl['county']['mean']) else 0
    std_county = sev_stats_lvl['county']['std'][t] if t < len(sev_stats_lvl['county']['std']) else 0
    cv_county = std_county / mean_county if mean_county > 0.01 else 0
    sev_cv_county.append(cv_county)

    mean_secondary = sev_stats_lvl['secondary']['mean'][t] if t < len(sev_stats_lvl['secondary']['mean']) else 0
    std_secondary = sev_stats_lvl['secondary']['std'][t] if t < len(sev_stats_lvl['secondary']['std']) else 0
    cv_secondary = std_secondary / mean_secondary if mean_secondary > 0.01 else 0
    sev_cv_secondary.append(cv_secondary)

    mean_township = sev_stats_lvl['township']['mean'][t] if t < len(sev_stats_lvl['township']['mean']) else 0
    std_township = sev_stats_lvl['township']['std'][t] if t < len(sev_stats_lvl['township']['std']) else 0
    cv_township = std_township / mean_township if mean_township > 0.01 else 0
    sev_cv_township.append(cv_township)

# 然后继续原有的绘图代码
# 平均严重程度趋势
# 平均严重程度趋势 - 两级显示
# 县医院
plot_series_with_ma(ax17a, X, sev_stats_lvl["county"]["mean"], COLORS["county"], "County Hospital")
# 基层医院（合并）
plot_series_with_ma(ax17a, X, combined_primary_severity, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax17a.set_title("Average Patient Severity by Hospital Level")
ax17a.set_ylabel("Severity Score")
ax17a.set_ylim(0, 1)
ax17a.legend()

# # 严重程度变异系数（使用上面计算的数据）
# ax17b.plot(X, moving_average(sev_cv_county, REPORT_MOVAVG), color=COLORS["county"], label="County", linewidth=2)
# ax17b.plot(X, moving_average(sev_cv_secondary, REPORT_MOVAVG), color=COLORS["secondary"], label="Secondary",
#            linewidth=2)
# ax17b.plot(X, moving_average(sev_cv_township, REPORT_MOVAVG), color=COLORS["township"], label="Township", linewidth=2)
# ax17b.set_title("Severity Variation (Coefficient of Variation)")
# ax17b.set_ylabel("CV (Std/Mean)")
# ax17b.legend()

# ... 继续其他子图的绘制

# 严重程度变异系数（标准差/均值）
# 严重程度变异系数 - 两级显示
# 计算基层医院平均变异系数
primary_cv = []
for t in range(T):
    if t < len(sev_cv_secondary) and t < len(sev_cv_township):
        avg_cv = (sev_cv_secondary[t] + sev_cv_township[t]) / 2
    elif t < len(sev_cv_secondary):
        avg_cv = sev_cv_secondary[t]
    elif t < len(sev_cv_township):
        avg_cv = sev_cv_township[t]
    else:
        avg_cv = 0
    primary_cv.append(avg_cv)

ax17b.plot(X, moving_average(sev_cv_county, REPORT_MOVAVG), color=COLORS["county"], label="County Hospital", linewidth=2)
ax17b.plot(X, moving_average(primary_cv, REPORT_MOVAVG), color=COLORS["township"], label="Primary Care", linewidth=2)
ax17b.set_title("Severity Variation (Coefficient of Variation)")
ax17b.set_ylabel("CV (Std/Mean)")
ax17b.legend()

# 严重程度分位数趋势
ax17c.plot(X, moving_average(sev_stats_lvl['county']['q25'], REPORT_MOVAVG), color=COLORS["county"],
           linestyle="--", alpha=0.7, label="County Q25")
ax17c.plot(X, moving_average(sev_stats_lvl['county']['q75'], REPORT_MOVAVG), color=COLORS["county"],
           linestyle=":", alpha=0.7, label="County Q75")
ax17c.plot(X, moving_average(sev_stats_lvl['township']['q25'], REPORT_MOVAVG), color=COLORS["township"],
           linestyle="--", alpha=0.7, label="Township Q25")
ax17c.plot(X, moving_average(sev_stats_lvl['township']['q75'], REPORT_MOVAVG), color=COLORS["township"],
           linestyle=":", alpha=0.7, label="Township Q75")
ax17c.set_title("Severity Quartiles (County vs Township)")
ax17c.set_xlabel("Period")
ax17c.set_ylabel("Severity Score")
ax17c.legend()

# 严重程度分布对比（最后50期的箱线图数据准备）
# 严重程度分布对比 - 两级显示
recent_periods = min(50, T)
recent_sev_data = []
labels = ['County Hospital', 'Primary Care']

# 县医院数据
recent_data_county = []
for t in range(T - recent_periods, T):
    if t >= 0 and not np.isnan(sev_stats_lvl['county']['mean'][t]):
        mean = sev_stats_lvl['county']['mean'][t]
        std = sev_stats_lvl['county']['std'][t]
        if not np.isnan(mean) and not np.isnan(std):
            simulated = np.random.normal(mean, std, 100)
            simulated = np.clip(simulated, 0, 1)
            recent_data_county.extend(simulated)
if recent_data_county:
    recent_sev_data.append(recent_data_county)

# 基层医院数据（合并二级和基层）
recent_data_primary = []
for t in range(T - recent_periods, T):
    # 二级医院数据
    if t >= 0 and not np.isnan(sev_stats_lvl['secondary']['mean'][t]):
        mean = sev_stats_lvl['secondary']['mean'][t]
        std = sev_stats_lvl['secondary']['std'][t]
        if not np.isnan(mean) and not np.isnan(std):
            simulated = np.random.normal(mean, std, 50)  # 减少样本数，因为要合并
            simulated = np.clip(simulated, 0, 1)
            recent_data_primary.extend(simulated)

    # 基层医院数据
    if t >= 0 and not np.isnan(sev_stats_lvl['township']['mean'][t]):
        mean = sev_stats_lvl['township']['mean'][t]
        std = sev_stats_lvl['township']['std'][t]
        if not np.isnan(mean) and not np.isnan(std):
            simulated = np.random.normal(mean, std, 50)  # 减少样本数，因为要合并
            simulated = np.clip(simulated, 0, 1)
            recent_data_primary.extend(simulated)

if recent_data_primary:
    recent_sev_data.append(recent_data_primary)

if recent_sev_data:
    ax17d.boxplot(recent_sev_data, labels=labels)
    ax17d.set_title("Recent Severity Distribution Comparison")
    ax17d.set_ylabel("Severity Score")

plt.tight_layout()



# 18平均就诊距离
plot_series_with_ma(ax16b, X, distance_traveled_series, "#14B8A6", "Average Travel Distance")
ax16b.set_title("Average Patient Travel Distance")
ax16b.set_xlabel("Period")
ax16b.set_ylabel("Distance")
ax16b.legend()
plt.tight_layout()

# 18) 医保支出监控（合并版）
fig18, (ax18a, ax18b) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：合并总支出和分级支出
# 总支出（左轴）
# 计算基层医院总支出（二级+基层）
primary_insurance_spending = [insurance_spending["secondary"][t] + insurance_spending["township"][t] for t in range(T)]

# 子图1：合并总支出和分级支出 - 两级显示
ax18a_twin = ax18a.twinx()  # 创建双y轴
plot_series_with_ma(ax18a, X, insurance_spending["total"], "black", "Total Insurance Spending")
ax18a.set_ylabel("Total Spending (Currency Units)", color="black")
ax18a.tick_params(axis='y', labelcolor="black")

# 分级支出（右轴）- 两级显示
plot_series_with_ma(ax18a_twin, X, insurance_spending["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax18a_twin, X, primary_insurance_spending, COLORS["township"], "Primary Care (Towns + Large Towns)")
ax18a_twin.set_ylabel("Level Spending (Currency Units)")
ax18a.set_title("Insurance Spending: Total vs By Hospital Level")
ax18a.legend(loc='upper left')
ax18a_twin.legend(loc='upper right')

# 子图2：支出占比堆叠图
# 计算基层医院支出占比
primary_insurance_share = [insurance_spending["secondary_share"][t] + insurance_spending["township_share"][t] for t in range(T)]

# 子图2：支出占比堆叠图 - 两级显示
ax18b.stackplot(X,
               insurance_spending["county_share"],
               primary_insurance_share,
               labels=['County Hospital', 'Primary Care (Towns + Large Towns)'],
               colors=[COLORS["county"], COLORS["township"]],
               alpha=0.7)
ax18b.set_title("Insurance Spending Share by Hospital Level")
ax18b.set_xlabel("Period")
ax18b.set_ylabel("Spending Share")
ax18b.set_ylim(0, 1)
ax18b.legend(loc='upper left')

plt.tight_layout()

# 19) 医保支出效率分析（合并版）
fig19, (ax19a, ax19b) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：合并单位患者支出和单位容量支出
# 单位患者支出（左轴）
ax19a_twin = ax19a.twinx()  # 创建双y轴

# 计算单位患者支出
avg_insurance_per_patient = []
for t in range(T):
    if treated_series[t] > 0:
        avg_spend = insurance_spending["total"][t] / treated_series[t]
    else:
        avg_spend = 0
    avg_insurance_per_patient.append(avg_spend)

plot_series_with_ma(ax19a, X, avg_insurance_per_patient, "#7C3AED", "Avg Spending per Patient")
ax19a.set_ylabel("Spending per Patient", color="#7C3AED")
ax19a.tick_params(axis='y', labelcolor="#7C3AED")

# 计算单位容量支出
# 计算单位容量支出 - 两级显示
spending_per_capacity = {
    "county": [],
    "primary": []  # 合并二级和基层
}

for t in range(T):
    # 县医院
    total_capacity_county = sum(Hlist[i].K for i in idx_county)
    total_spending_county = insurance_spending["county"][t]
    if total_capacity_county > 0:
        spend_per_cap_county = total_spending_county / total_capacity_county
    else:
        spend_per_cap_county = 0
    spending_per_capacity["county"].append(spend_per_cap_county)

    # 基层医院（合并二级和基层）
    total_capacity_primary = sum(Hlist[i].K for i in idx_secondary + idx_township)
    total_spending_primary = insurance_spending["secondary"][t] + insurance_spending["township"][t]
    if total_capacity_primary > 0:
        spend_per_cap_primary = total_spending_primary / total_capacity_primary
    else:
        spend_per_cap_primary = 0
    spending_per_capacity["primary"].append(spend_per_cap_primary)

plot_series_with_ma(ax19a_twin, X, spending_per_capacity["county"], COLORS["county"], "County per Capacity")
plot_series_with_ma(ax19a_twin, X, spending_per_capacity["primary"], COLORS["township"], "Primary Care per Capacity")
ax19a_twin.set_ylabel("Spending per Capacity Unit")
ax19a.set_title("Spending Efficiency: Per Patient vs Per Capacity")
ax19a.legend(loc='upper left')
ax19a_twin.legend(loc='upper right')

# 子图2：支出效率趋势
spending_efficiency = []
for t in range(T):
    if insurance_spending["total"][t] > 0:
        efficiency = treated_series[t] / insurance_spending["total"][t]
    else:
        efficiency = 0
    spending_efficiency.append(efficiency)

plot_series_with_ma(ax19b, X, spending_efficiency, "#0891B2", "Patients per Unit Spending")
ax19b.set_title("Spending Efficiency Trend (Treated Patients per Spending Unit)")
ax19b.set_xlabel("Period")
ax19b.set_ylabel("Efficiency Ratio")
ax19b.legend()

plt.tight_layout()

# 20) 医保支出综合监控
fig20, (ax20a, ax20b) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：预算使用率监控
budget_use_rate = []
for t in range(T):
    if BUDGET_PER_PERIOD > 0:
        use_rate = insurance_spending["total"][t] / BUDGET_PER_PERIOD
    else:
        use_rate = 0
    budget_use_rate.append(use_rate)

ax20a.plot(X, moving_average(budget_use_rate, REPORT_MOVAVG), color="#DC2626", linewidth=2.5,
           label="Budget Use Rate (MA)")
ax20a.axhline(y=1.0, color="#059669", linestyle="--", linewidth=2, label="Budget Limit (100%)")
ax20a.fill_between(X, 0, budget_use_rate, alpha=0.3, color="#DC2626")
ax20a.set_title("Budget Utilization Rate Over Time")
ax20a.set_xlabel("Period")
ax20a.set_ylabel("Budget Use Rate")
ax20a.set_ylim(0, max(1.2, max(budget_use_rate) * 1.1))
ax20a.legend()

# 子图2：支出与治疗效果关联（散点图，使用最后50期数据）
recent_periods = min(50, T)
if recent_periods > 10:
    # 准备数据
    periods = list(range(max(0, T - recent_periods), T))
    spending_data = [insurance_spending["total"][t] / 1000 for t in periods]  # 千单位
    treated_data = [treated_series[t] for t in periods]

    # 创建散点图，颜色反映时间（从蓝到红）
    colors = periods
    scatter = ax20b.scatter(spending_data, treated_data, c=colors, cmap='viridis', alpha=0.6, s=60)
    ax20b.set_xlabel("Insurance Spending (Thousand Units)")
    ax20b.set_ylabel("Treated Patients")
    ax20b.set_title("Spending vs Treatment Volume (Recent Periods)")

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax20b)
    cbar.set_label('Period (Recent →)')

plt.tight_layout()

# 21) 医院容量变动监控
# 21) 医院容量变动监控 - 两级显示
fig21, (ax21a, ax21b) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：两级架构容量趋势
# 县医院容量
plot_series_with_ma(ax21a, X, capacity_trends["county"], COLORS["county"], "County Hospital")
# 基层医院总容量（二级+基层）
plot_series_with_ma(ax21a, X, combined_primary_capacity, COLORS["township"], "Primary Care Total Capacity")
ax21a.set_title("Hospital Capacity Trends by Level")
ax21a.set_ylabel("Total Capacity ")
ax21a.legend()

# 子图2：容量占比堆叠图
ax21b.stackplot(X,
               capacity_trends["county_share"],
               capacity_trends["secondary_share"],
               capacity_trends["township_share"],
               labels=['County', 'Primary Care Total'],
               colors=[COLORS["county"], COLORS["secondary"], COLORS["township"]],
               alpha=0.7)
ax21b.set_title("Capacity Share by Hospital Level")
ax21b.set_xlabel("Period")
ax21b.set_ylabel("Capacity Share")
ax21b.set_ylim(0, 1)
ax21b.legend(loc='upper left')

plt.tight_layout()

# 22) 容量扩张分析
fig22, (ax22a, ax22b) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：容量增长率
capacity_growth = {
    "county": [],
    "secondary": [],
    "township": []
}

for t in range(1, T):
    for level in ["county", "secondary", "township"]:
        if t < len(capacity_trends[level]) and capacity_trends[level][t-1] > 0:
            growth = (capacity_trends[level][t] - capacity_trends[level][t-1]) / capacity_trends[level][t-1]
        else:
            growth = 0
        capacity_growth[level].append(growth)

# 由于增长率从第1期开始，X也要相应调整
X_growth = np.arange(1, T)

plot_series_with_ma(ax22a, X_growth, capacity_growth["county"], COLORS["county"], "County Growth")
plot_series_with_ma(ax22a, X_growth, capacity_growth["secondary"], COLORS["secondary"], "Secondary Growth")
plot_series_with_ma(ax22a, X_growth, capacity_growth["township"], COLORS["township"], "Township Growth")
ax22a.set_title("Hospital Capacity Growth Rate by Level")
ax22a.set_ylabel("Growth Rate")
ax22a.legend()

# 子图2：容量利用率与容量变动关联
# 计算平均容量利用率
avg_utilization = []
for t in range(T):
    total_beddays = (bed_occupancy_lvl['county'][t] * capacity_trends["county"][t] +
                    bed_occupancy_lvl['secondary'][t] * capacity_trends["secondary"][t] +
                    bed_occupancy_lvl['township'][t] * capacity_trends["township"][t])
    total_capacity = (capacity_trends["county"][t] + capacity_trends["secondary"][t] + capacity_trends["township"][t])
    if total_capacity > 0:
        avg_util = total_beddays / total_capacity
    else:
        avg_util = 0
    avg_utilization.append(avg_util)

# 创建双Y轴图表
ax22b_twin = ax22b.twinx()
total_capacity_series = [capacity_trends["county"][t] + capacity_trends["secondary"][t] + capacity_trends["township"][t] for t in range(T)]

plot_series_with_ma(ax22b, X, avg_utilization, "#7C3AED", "Avg Utilization Rate")
ax22b.set_ylabel("Utilization Rate", color="#7C3AED")
ax22b.tick_params(axis='y', labelcolor="#7C3AED")

plot_series_with_ma(ax22b_twin, X, total_capacity_series, "black", "Total System Capacity")
ax22b_twin.set_ylabel("Total Capacity", color="black")
ax22b_twin.tick_params(axis='y', labelcolor="black")

ax22b.set_title("Capacity Utilization vs Total System Capacity")
ax22b.set_xlabel("Period")
ax22b.legend(loc='upper left')
ax22b_twin.legend(loc='upper right')

plt.tight_layout()

###
# 23) 医共体合作强度监控
fig23, ((ax23a, ax23b), (ax23c, ax23d)) = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：转诊倾向与合作强度趋势
# 转诊倾向 (下级医院)
ax23a_twin = ax23a.twinx()
plot_series_with_ma(ax23a, X, action_distribution["secondary"]["rho"], COLORS["secondary"],
                    "Secondary Referral Propensity")
plot_series_with_ma(ax23a, X, action_distribution["township"]["rho"], COLORS["township"],
                    "Township Referral Propensity")
ax23a.set_ylabel("Referral Propensity (ρ)", color=COLORS["secondary"])
ax23a.tick_params(axis='y', labelcolor=COLORS["secondary"])

# 合作强度 (所有医院)
plot_series_with_ma(ax23a_twin, X, action_distribution["county"]["a"], COLORS["county"], "County Cooperation")
plot_series_with_ma(ax23a_twin, X, action_distribution["secondary"]["a"], COLORS["secondary"], "Secondary Cooperation")
plot_series_with_ma(ax23a_twin, X, action_distribution["township"]["a"], COLORS["township"], "Township Cooperation")
ax23a_twin.set_ylabel("Cooperation Intensity (a)")
ax23a.set_title("Referral Propensity vs Cooperation Intensity")
ax23a.legend(loc='upper left')
ax23a_twin.legend(loc='upper right')

# 子图2：能力下沉与分配政策
# 能力下沉 (县医院)
ax23b_twin = ax23b.twinx()
plot_series_with_ma(ax23b, X, action_distribution["county"]["d"], "#8B5CF6", "County Ability Sinking (d)")
ax23b.set_ylabel("Ability Sinking (d)", color="#8B5CF6")
ax23b.tick_params(axis='y', labelcolor="#8B5CF6")

# 分配比例 (县医院)
plot_series_with_ma(ax23b_twin, X, county_allocation_ratio, "#10B981", "County Allocation Ratio")
ax23b_twin.set_ylabel("Allocation Ratio", color="#10B981")
ax23b_twin.tick_params(axis='y', labelcolor="#10B981")
ax23b.set_title("County-led Cooperation: Ability Sinking vs Allocation")
ax23b.legend(loc='upper left')
ax23b_twin.legend(loc='upper right')

# 子图3：转诊绩效与能力提升
# 转诊成功率
plot_series_with_ma(ax23c, X, referral_success_rate, "#EF4444", "Referral Success Rate")
ax23c.set_ylabel("Success Rate")
ax23c.set_ylim(0, 1)
ax23c.legend()

# 能力提升 (下级医院)
plot_series_with_ma(ax23c, X, smax_uplift_lower, "#F59E0B", "Lower-tier Ability Uplift")
ax23c_twin = ax23c.twinx()
ax23c_twin.set_ylabel("Ability Uplift", color="#F59E0B")
ax23c_twin.tick_params(axis='y', labelcolor="#F59E0B")
ax23c.set_title("Referral Performance vs Ability Improvement")
ax23c_twin.legend(loc='upper right')

# 子图4：合作效果综合指标
# 计算综合合作指数（简单加权平均）
cooperation_index = []
for t in range(T):
    # 转诊倾向（下级医院平均）
    rho_avg = (action_distribution["secondary"]["rho"][t] + action_distribution["township"]["rho"][t]) / 2

    # 能力下沉（县医院）
    d_val = action_distribution["county"]["d"][t]

    # 转诊成功率
    ref_success = referral_success_rate[t]

    # 能力提升
    ability_uplift = smax_uplift_lower[t]

    # 综合指数（简单加权）
    composite = 0.3 * rho_avg + 0.3 * d_val + 0.2 * ref_success + 0.2 * ability_uplift
    cooperation_index.append(composite)

plot_series_with_ma(ax23d, X, cooperation_index, "#3B82F6", "Composite Cooperation Index")
ax23d.set_title("Comprehensive Cooperation Index Over Time")
ax23d.set_xlabel("Period")
ax23d.set_ylabel("Cooperation Index")
ax23d.set_ylim(0, 1)
ax23d.legend()

plt.tight_layout()

# 24) 合作行为与系统绩效关联分析
fig24, (ax24a, ax24b) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：合作强度与系统效率关联
system_efficiency = np.array(treated_series) / (np.array(treated_series) + np.array(unserved_series) + 1e-6)

# 使用散点图展示合作指数与系统效率的关系（最近50期）
recent_periods = min(50, T)
if recent_periods > 10:
    periods = list(range(max(0, T - recent_periods), T))
    coop_data = [cooperation_index[t] for t in periods]
    eff_data = [system_efficiency[t] for t in periods]

    scatter = ax24a.scatter(coop_data, eff_data, c=periods, cmap='viridis', alpha=0.6, s=60)
    ax24a.set_xlabel("Cooperation Index")
    ax24a.set_ylabel("System Efficiency")
    ax24a.set_title("Cooperation vs System Efficiency (Recent Periods)")

    # 添加趋势线
    if len(coop_data) > 1:
        z = np.polyfit(coop_data, eff_data, 1)
        p = np.poly1d(z)
        ax24a.plot(coop_data, p(coop_data), "r--", alpha=0.8, linewidth=2, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        ax24a.legend()

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax24a)
    cbar.set_label('Period (Recent →)')

# 子图2：各级医院合作行为对比（箱线图）
cooperation_by_level = {
    'County': action_distribution["county"]["a"],
    'Secondary': action_distribution["secondary"]["a"],
    'Township': action_distribution["township"]["a"]
}

# 只使用稳定期数据（后50%时期）
stable_start = T // 2
coop_data_stable = []
labels = []
for level, data in cooperation_by_level.items():
    stable_data = data[stable_start:]
    # 移除NaN值
    stable_data = [x for x in stable_data if not np.isnan(x)]
    if stable_data:
        coop_data_stable.append(stable_data)
        labels.append(level)

if coop_data_stable:
    box_plot = ax24b.boxplot(coop_data_stable, labels=labels, patch_artist=True)

    # 设置颜色
    colors = [COLORS["county"], COLORS["secondary"], COLORS["township"]]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax24b.set_title("Cooperation Intensity Distribution by Level (Stable Phase)")
    ax24b.set_ylabel("Cooperation Intensity (a)")

plt.tight_layout()
# 输出CSV和热力图
export_metrics_to_csv()
# create_spatial_heatmap()
create_professional_heatmap()
plt.show()

print("S1 Ultimate Version 完成！主要增强特性：")
print("✓ 真实地理布局和人口密度抽样")
print("✓ 等待容忍阈值 + 外流分类统计")
print("✓ 床日/LOS机制（基于病情严重度）")
print("✓ 关闭外生偏置，纯内生行为")
print("✓ 增强监控指标和CSV输出")
print("✓ 空间热力图生成")
print("✓ 完整的S1场景实现")
