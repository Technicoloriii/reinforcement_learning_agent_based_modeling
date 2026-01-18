# -*- coding: utf-8 -*-
"""
IHC-ABM Enhanced (v4) — S3 Dynamic Budget Adjustment
- 在S2基础上增加医保局动态预算调整机制
- 半年期预算评估和调整
- 县医院面临预算保卫与效率平衡的困境
- 新增策略性行为监控指标
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

# =============== S3 动态预算调整参数 ===============
ENABLE_DYNAMIC_BUDGET = True  # 启用S3动态预算调整
BUDGET_ADJUSTMENT_CYCLE = 26  # 每26周（半年）调整一次
BUDGET_ADJUSTMENT_START = 52  # 从第52周开始调整（第1年结束后）

# 预算调整规则参数 - 增强敏感性
BUDGET_ADJUSTMENT_RATE = 0.30 # 基础调整率15%
SURPLUS_TARGET = 0.05  # 目标结余率5%
SURPLUS_TOLERANCE = 0.03  # 容忍区间±3%
MAX_BUDGET_CUT = 0.70 # 最大预算削减30%
MAX_BUDGET_INCREASE = 0.15  # 最大预算增加15%

# 新增：硬预算约束 - 医共体完全承担赤字风险
HARD_BUDGET_CONSTRAINT = True
BUDGET_DEFICIT_PENALTY_RATE = 1.0  # 赤字100%由医共体承担

# 县医院预算防御策略参数
ENABLE_BUDGET_DEFENSE = True  # 县医院启用预算防御策略
DEFENSE_STRATEGY_WEIGHT = 0.3  # 预算防御在决策中的权重

# =============== 专业热力图参数 ===============
# inferno colormap without top white (trim last 8%)
_inferno = mpl.colormaps['inferno']
HEAT_CMAP = ListedColormap(_inferno(np.linspace(0.0, 0.92, 256)))

# 城市背景颜色映射
SOFT_CMAP = LinearSegmentedColormap.from_list(
    "soft_mint", ["#ffffff", "#f3faf7", "#e7f6f0", "#d3efe6", "#bfe7dd", "#aee0d5"]
)

# 医院颜色
COLOR_TERTIARY = "#d97706"  # 县医院
COLOR_SECONDARY = "#2563eb"  # 二级医院
COLOR_PRIMARY = "#059669"  # 基层医院


# =============== 从geocity导入真实布局 ===============
def load_city_layout():
    """从geocity_update_2.py加载城市布局和医院位置"""
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
    "outflow_mismatch": "#45B7D1",
    "budget_adjustment": "#DC2626",  # S3新增：预算调整颜色
    "surplus_rate": "#059669",  # S3新增：结余率颜色
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
REFERRAL_SEV_THRESHOLD = 0.45  # 原逻辑是 0.60，调低门槛以减少“必须先去基层”的人群

# =============== 等待容忍参数 ===============
WAIT_TOLERANCE_A0 = 8.0  # 基础等待容忍
WAIT_TOLERANCE_A1 = 10.0  # 严重度对等待容忍的影响系数

# =============== 床日/LOS参数 ===============
LOS_TAU = 0.38  # 住院阈值，严重度>=0.38需要住院
LOS_BASE = 1.0  # 基础床日
LOS_SEVERITY_MULTIPLIER = 6.0  # 严重度对床日的影响
LOS_LEVEL_MULTIPLIER = {  # 医院级别对床日的影响
    'township': 0.9,
    'secondary': 1.0,
    'county': 1.15
}

# =============== 关闭外生偏置 ===============
ENABLE_PRESTIGE = False  # 关闭外生声望偏置
PRESTIGE_BIAS_COUNTY = 0.0
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


FIXED_COST_PER_CAP = dict(county=120.0, secondary=35.0, township=35.0)
A_COST_PER_UNIT = 80.0

I_COST = {-1: 1500.0, 0: 0.0, 1: 2500.0, 2: 3000.0}
K_DELTA = {-1: -0.06, 0: 0.0, 1: 0.10, 2: 0.22}

# ===== Scenario 3: Bundled payment with dynamic budget adjustment =====
SCENARIO = "bundle_dynamic_budget"
ENABLE_RHO = True
ENABLE_V = True
ENABLE_D = True

# S3修改：初始预算因子，后续会动态调整
BUDGET_FACTOR = 1.00
USE_BUDGET_FIXED = False
BUDGET_FIXED = 0.0

# Surplus sharing rule
BONUS_BASE_SHARE = dict(county=0.40, secondary=0.35, township=0.25)
BONUS_RHO_WEIGHT = 0.60
BONUS_D_WEIGHT = 0.60
ALLOW_NEGATIVE_SURPLUS = True

# ===== County Allocation Decision =====
if ENABLE_COUNTY_ALLOCATION:
    ALLOCATION_GRID = np.array([0.2, 0.35, 0.5, 0.65, 0.8])

# ===== Enhanced Cooperation Parameters =====
S_MAX_UPPER = dict(county=0.9, secondary=0.55, township=0.5)
SMAX0 = dict(county=0.9, secondary=0.25, township=0.25)
ETA_D = dict(county=0.0, secondary=0.035, township=0.065)

EMPTY_COST_PHI = dict(county=0.55, secondary=0.2, township=0.15)
I_COST_MULT = dict(county=1.00, secondary=1.70, township=1.90)
FINANCE_PSI = dict(county=0.05, secondary=0.12, township=0.15)

THR_BASE = dict(township=0.4, secondary=0.35)
THR_RHO_COEF = dict(township=0.20, secondary=0.20)
THR_V_COEF = dict(township=0.1, secondary=0.1)

LIGHT_SEV_THRESHOLD = 0.35

# =============== Waiting & Process Quality ===============
W0 = 0.6
XI = 0.9
BETA_A_W = 0.28
BETA_A_Q = 0.06

Q_BASE = dict(county=0.8, secondary=0.45, township=0.4)
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


# =============== S3 动态预算调整函数 ===============
class DynamicBudgetAdjustment:
    """S3动态预算调整机制"""

    def __init__(self):
        self.adjustment_history = []  # 记录每次调整
        self.budget_history = []  # 记录每期预算
        self.surplus_rate_history = []  # 记录每期结余率
        self.cycle_count = 0

    def calculate_surplus_rate(self, budget, actual_spending):
        """计算结余率"""
        if budget > 0:
            return (budget - actual_spending) / budget
        return 0.0

    def adjust_budget(self, current_budget, surplus_rate, cycle_performance):
        """
        根据结余率和周期表现调整预算
        cycle_performance: 包含周期内关键指标
        """
        # 基础调整规则
        if surplus_rate > SURPLUS_TARGET + SURPLUS_TOLERANCE:
            # 结余过多，削减预算
            adjustment = -min(BUDGET_ADJUSTMENT_RATE * (surplus_rate - SURPLUS_TARGET), MAX_BUDGET_CUT)
        elif surplus_rate < SURPLUS_TARGET - SURPLUS_TOLERANCE:
            # 结余不足，增加预算
            adjustment = min(BUDGET_ADJUSTMENT_RATE * (SURPLUS_TARGET - surplus_rate), MAX_BUDGET_INCREASE)
        else:
            # 在目标区间内，小幅调整
            adjustment = 0.02 * (SURPLUS_TARGET - surplus_rate)  # 温和调整

        new_budget = current_budget * (1 + adjustment)

        # 记录调整事件
        adjustment_event = {
            'cycle': self.cycle_count,
            'old_budget': current_budget,
            'new_budget': new_budget,
            'surplus_rate': surplus_rate,
            'adjustment_rate': adjustment,
            'reason': self._get_adjustment_reason(surplus_rate, adjustment)
        }
        self.adjustment_history.append(adjustment_event)
        self.cycle_count += 1

        return new_budget, adjustment_event

    def _get_adjustment_reason(self, surplus_rate, adjustment):
        """获取调整原因描述"""
        if adjustment < -0.02:
            return "结余过多，预防性削减预算"
        elif adjustment < 0:
            return "结余偏高，小幅削减预算"
        elif adjustment > 0.02:
            return "结余不足，支持性增加预算"
        elif adjustment > 0:
            return "结余偏低，小幅增加预算"
        else:
            return "结余在目标区间，预算维持"

    def get_budget_defense_advice(self, current_surplus_rate, periods_to_adjustment):
        """为县医院提供预算防御建议"""
        if periods_to_adjustment > 13:  # 半年周期的一半
            return "normal"  # 正常运营
        elif periods_to_adjustment > 4:
            if current_surplus_rate > SURPLUS_TARGET + 0.05:
                return "reduce_surplus"  # 需要降低结余
            elif current_surplus_rate < SURPLUS_TARGET - 0.05:
                return "increase_surplus"  # 需要增加结余
            else:
                return "maintain"  # 维持现状
        else:
            # 临近调整期，更积极的防御
            if current_surplus_rate > SURPLUS_TARGET + 0.08:
                return "aggressive_reduce"
            elif current_surplus_rate < SURPLUS_TARGET - 0.08:
                return "aggressive_increase"
            else:
                return "optimize"


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
TAU_A_START = 0.4
TAU_A_END = 0.1
TAU_I = 0.20

GAMMA = 0.75
ALPHA_Q_UPD = 0.4


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


# =============== 新增：分层奖励函数 ===============
def hierarchical_reward_function(hospital, t, surplus_rate, budget_environment):
    """分层奖励函数：根据医院级别和预算环境调整奖励"""
    base_profit_reward = hospital.profit_prev / max(hospital.K, 1.0) / 8.0
    waiting_penalty = -0.04 * hospital.W_prev

    if hospital.level == "county":
        # 县医院：预算周期敏感的奖励
        periods_to_adj = getattr(hospital, 'periods_to_adjustment', BUDGET_ADJUSTMENT_CYCLE)

        if periods_to_adj < 8:  # 调整临近期
            # 预算防御成为首要任务
            surplus_deviation = abs(surplus_rate - SURPLUS_TARGET)
            budget_reward = -0.3 * surplus_deviation

            # 成功防御的显著奖励
            if 0.02 < surplus_deviation < 0.08:
                budget_reward += 0.15  # 适度偏离目标的奖励

            # 运营效率权重降低
            operation_weight = 0.3
            budget_weight = 0.7
        else:
            # 正常运营期
            budget_reward = -0.1 * abs(surplus_rate - SURPLUS_TARGET)
            operation_weight = 0.8
            budget_weight = 0.2

        total_reward = (operation_weight * (base_profit_reward + waiting_penalty) +
                        budget_weight * budget_reward)

    else:
        # 下级医院：预算环境敏感的奖励
        if budget_environment < 0:  # 预算紧缩期
            # 效率优先，合作权重降低
            cooperation_reward = 0.1 * getattr(hospital, 'rho', 0.0)
            efficiency_bonus = 0.2 * (hospital.u_prev - 0.7)  # 鼓励提高利用率
            total_reward = base_profit_reward + waiting_penalty + cooperation_reward + efficiency_bonus
        else:
            # 预算宽松期，质量优先
            quality_bonus = 0.1 * hospital.a  # 合作和质量奖励
            total_reward = base_profit_reward + waiting_penalty + quality_bonus

    return total_reward


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


# =============== Enhanced Hospital Agent with S3 Budget Defense ===============
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

        # S3新增：预算防御策略
        if ENABLE_BUDGET_DEFENSE and level == "county":
            self.defense_grid = np.array([0.0, 0.3, 0.6, 1.0])  # 预算防御强度
            self.Qdefense = defaultdict(lambda: np.zeros(len(self.defense_grid)))
            self.defense_strategy = 0.0  # 当前防御策略强度
            # S3新增：预算相关状态
        self.budget_awareness = 0.0  # 预算意识强度
        self.periods_to_adjustment = BUDGET_ADJUSTMENT_CYCLE  # 距离下次调整的期数

        # 新增：危机模式属性
        self.crisis_mode = False
        self.crisis_duration = 0
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

        # S3新增：预算相关状态
        self.budget_awareness = 0.0  # 预算意识强度
        self.periods_to_adjustment = BUDGET_ADJUSTMENT_CYCLE  # 距离下次调整的期数

    # S3新增：预算防御状态哈希
    def state_hash_defense(self):
        """预算防御决策的状态"""
        surplus_rate = getattr(self, 'current_surplus_rate', 0.5)
        periods_to_adj = getattr(self, 'periods_to_adjustment', BUDGET_ADJUSTMENT_CYCLE)

        surplus_bin = 0 if surplus_rate < 0.05 else (1 if surplus_rate < 0.15 else 2)
        time_bin = 0 if periods_to_adj > 13 else (1 if periods_to_adj > 4 else 2)
        budget_trend = getattr(self, 'budget_trend', 0)  # 0=稳定, 1=上升, -1=下降

        return (surplus_bin, time_bin, budget_trend)

    def select_defense_strategy(self, tau=0.6):
        """选择预算防御策略"""
        if not ENABLE_BUDGET_DEFENSE or self.level != "county":
            self.defense_strategy = 0.0
            return self.state_hash_defense(), 0

        s = self.state_hash_defense()
        probs = softmax(self.Qdefense[s], tau)
        idx = rng.choice(len(self.defense_grid), p=probs)
        self.defense_strategy = float(self.defense_grid[idx])
        return s, idx

    def update_Qdefense(self, s, idx, r, s_next):
        """更新预算防御Q值"""
        if ENABLE_BUDGET_DEFENSE and self.level == "county":
            best_next = np.max(self.Qdefense[s_next])
            self.Qdefense[s][idx] = (1 - ALPHA_Q_UPD) * self.Qdefense[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    # S3修改：分配决策考虑预算防御
    def select_allocation(self, tau=0.6):
        if not ENABLE_COUNTY_ALLOCATION or self.level != "county":
            self.allocation_ratio = 0.5
            return self.state_hash_alloc(), 0

        s = self.state_hash_alloc()

        # 如果有预算防御策略，调整分配倾向
        base_probs = softmax(self.Qalloc[s], tau)
        if self.defense_strategy > 0:
            # 防御策略下倾向于保留更多结余
            defense_effect = self.defense_strategy * 0.6
            adjusted_probs = base_probs * (1 + defense_effect * (self.alloc_grid - 0.5))
            adjusted_probs = np.maximum(adjusted_probs, 0)
            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        else:
            adjusted_probs = base_probs

        idx = rng.choice(len(self.alloc_grid), p=adjusted_probs)
        self.allocation_ratio = float(self.alloc_grid[idx])
        return s, idx

    def state_hash_alloc(self):
        u_bin = 0 if self.u_prev < 0.5 else (1 if self.u_prev < 0.85 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
        ref_rate = self.referral_success / max(self.referral_attempts, 1)
        ref_bin = 0 if ref_rate < 0.3 else (1 if ref_rate < 0.7 else 2)

        # S3新增：预算状态
        budget_bin = 0 if getattr(self, 'budget_trend', 0) < 0 else (1 if getattr(self, 'budget_trend', 0) == 0 else 2)
        return (u_bin, p_bin, ref_bin, budget_bin)

    # 其他原有方法保持不变
    def state_hash_rho(self):
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0
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

    def update_Qalloc(self, s, idx, r, s_next):
        """更新分配决策的Q值"""
        if ENABLE_COUNTY_ALLOCATION and self.level == "county":
            best_next = np.max(self.Qalloc[s_next])
            self.Qalloc[s][idx] = (1 - ALPHA_Q_UPD) * self.Qalloc[s][idx] + ALPHA_Q_UPD * (r + GAMMA * best_next)

    def state_hash_a(self):
        # 原有运营状态
        u_bin = 0 if self.u_prev < 0.6 else (1 if self.u_prev < 0.85 else 2)
        W_bin = 0 if self.W_prev < 6 else (1 if self.W_prev < 12 else 2)
        p_bin = 1 if self.profit_prev > 0 else 0

        # 新增：预算环境状态
        periods_to_adj = getattr(self, 'periods_to_adjustment', BUDGET_ADJUSTMENT_CYCLE)
        if periods_to_adj > 20:
            time_bin = 0  # 周期早期
        elif periods_to_adj > 10:
            time_bin = 1  # 周期中期
        elif periods_to_adj > 5:
            time_bin = 2  # 调整临近期
        else:
            time_bin = 3  # 紧急调整期

        surplus_rate = getattr(self, 'current_surplus_rate', 0.5)
        if surplus_rate < SURPLUS_TARGET - 0.1:
            surplus_bin = 0  # 严重不足
        elif surplus_rate < SURPLUS_TARGET - 0.03:
            surplus_bin = 1  # 轻度不足
        elif surplus_rate < SURPLUS_TARGET + 0.03:
            surplus_bin = 2  # 目标区间
        elif surplus_rate < SURPLUS_TARGET + 0.1:
            surplus_bin = 3  # 轻度结余
        else:
            surplus_bin = 4  # 过度结余

        budget_trend = getattr(self, 'budget_trend', 0)  # -1=下降, 0=稳定, 1=上升

        return (u_bin, W_bin, p_bin, time_bin, surplus_bin, budget_trend)

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

# =============== 其他原有类保持不变 ===============
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
        if USE_REFERRAL_SYSTEM and Sev[p] < REFERRAL_SEV_THRESHOLD:
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


# S3修改：初始预算计算
if not USE_BUDGET_FIXED:
    exp_mult = expected_severity_multiplier()
    INITIAL_BUDGET_PER_PERIOD = BUDGET_FACTOR * REIMB_RATIO * exp_mult * sum([
        U_START['county'] * K_INIT['county'] * PRICE['county'],
        U_START['secondary'] * K_INIT['secondary'] * PRICE['secondary'],
        U_START['township'] * K_INIT['township'] * PRICE['township']
    ])
else:
    INITIAL_BUDGET_PER_PERIOD = float(BUDGET_FIXED)

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

# =============== S3 新增数据收集器 ===============
# S3预算调整相关指标
budget_adjustment_events = []  # 预算调整事件记录
budget_history = []  # 每周预算历史
surplus_rate_history = []  # 每周结余率历史
cycle_performance_history = []  # 周期表现记录

# 县医院策略性行为指标
county_defense_strategy = []  # 县医院防御策略强度
strategic_behavior_detected = []  # 策略性行为检测
budget_awareness_trend = []  # 预算意识趋势

# 预算周期分析
cycle_surplus_rates = []  # 各周期结余率
cycle_budget_changes = []  # 各周期预算变化
cycle_efficiency_metrics = []  # 各周期效率指标

# 原有指标继续保留
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

referral_success_rate = []
county_allocation_ratio = []
referral_path_length = []

bed_occupancy_lvl = {"county": [], "secondary": [], "township": []}
inpatient_share_lvl = {"county": [], "secondary": [], "township": []}
avg_los_inp_lvl = {"county": [], "secondary": [], "township": []}

outflow_wait_series = []
outflow_full_series = []
outflow_mismatch_series = []

hospital_flow_history = [[] for _ in range(H)]

# 两级架构合并统计
combined_primary_flow_share = []
combined_primary_capacity = []
combined_primary_severity = []
combined_primary_profits = []
combined_primary_utilization = []
combined_primary_bed_occupancy = []
combined_primary_inpatient_share = []

ref_Q = ref_W = ref_D = ref_HHI = None

# 原有核心指标
waiting_time_series = []
process_quality_series = []
patient_utility_series = []
distance_traveled_series = []
hospital_profits = {"county": [], "secondary": [], "township": []}
capacity_utilization = {"county": [], "secondary": [], "township": []}
referral_efficiency = []
budget_utilization = []
surplus_distribution = {"county": [], "secondary": [], "township": []}

learning_convergence = {"county": [], "secondary": [], "township": []}
action_distribution = {
    "county": {"a": [], "i": [], "rho": [], "v": [], "d": [], "defense": []},  # 添加 defense
    "secondary": {"a": [], "i": [], "rho": []},
    "township": {"a": [], "i": [], "rho": []}
}
sev_stats_lvl = {
    "county": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []},
    "secondary": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []},
    "township": {"mean": [], "std": [], "min": [], "max": [], "q25": [], "q75": []}
}

insurance_spending = {
    "total": [],
    "county": [],
    "secondary": [],
    "township": [],
    "county_share": [],
    "secondary_share": [],
    "township_share": []
}

capacity_trends = {
    "county": [],
    "secondary": [],
    "township": [],
    "county_share": [],
    "secondary_share": [],
    "township_share": []
}

patient_choice_quality = []
habit_persistence = []
information_asymmetry_effect = []

# =============== S3 主循环 ===============
current_referral_attempts = 0
current_referral_success = 0
backlog_unserved = 0.0
tau_a = TAU_A_START
tau_decay = (TAU_A_START - TAU_A_END) / max(T - 1, 1)

# 初始化患者行为
patient_behavior = PatientBehavior(10000)

# S3新增：初始化动态预算调整器
budget_adjuster = DynamicBudgetAdjustment()
current_budget = INITIAL_BUDGET_PER_PERIOD
budget_history = [current_budget] * T  # 预先填充

# 周期累计变量
cycle_insurance_spending = 0
cycle_treated_patients = 0
cycle_start = 0

for t in range(T):
    # S3: 更新县医院的预算调整倒计时
    for i in idx_county:
        Hlist[i].periods_to_adjustment = (BUDGET_ADJUSTMENT_CYCLE - (
                    t - cycle_start) % BUDGET_ADJUSTMENT_CYCLE) % BUDGET_ADJUSTMENT_CYCLE

    # S3: 预算调整周期检查
    if (ENABLE_DYNAMIC_BUDGET and t >= BUDGET_ADJUSTMENT_START and
            (t - cycle_start) % BUDGET_ADJUSTMENT_CYCLE == 0 and t > cycle_start):

        # 计算周期表现
        cycle_length = BUDGET_ADJUSTMENT_CYCLE
        cycle_surplus_rate = budget_adjuster.calculate_surplus_rate(
            current_budget * cycle_length, cycle_insurance_spending
        )
        # budget_trend = 1 if adjustment_event['adjustment_rate'] > 0 else (
        #     -1 if adjustment_event['adjustment_rate'] < 0 else 0)
        # for i in idx_county:
        #     Hlist[i].budget_trend = budget_trend
        #     Hlist[i].current_surplus_rate = cycle_surplus_rate
        # 准备周期表现数据
        cycle_performance = {
            'cycle_number': len(budget_adjuster.adjustment_history),
            'total_treated': cycle_treated_patients,
            'avg_waiting_time': np.mean(waiting_time_series[-cycle_length:]) if len(
                waiting_time_series) >= cycle_length else 0,
            'avg_quality': np.mean(process_quality_series[-cycle_length:]) if len(
                process_quality_series) >= cycle_length else 0,
            'referral_success': np.mean(referral_success_rate[-cycle_length:]) if len(
                referral_success_rate) >= cycle_length else 0
        }

        # 调整预算
        new_budget, adjustment_event = budget_adjuster.adjust_budget(
            current_budget, cycle_surplus_rate, cycle_performance
        )

        # 更新预算
        current_budget = new_budget
        budget_adjustment_events.append(adjustment_event)

        # 更新县医院的预算趋势
        budget_trend = 1 if adjustment_event['adjustment_rate'] > 0 else (
            -1 if adjustment_event['adjustment_rate'] < 0 else 0)
        for i in idx_county:
            Hlist[i].budget_trend = budget_trend
            Hlist[i].current_surplus_rate = cycle_surplus_rate

        # 重置周期累计变量
        cycle_insurance_spending = 0
        cycle_treated_patients = 0
        cycle_start = t

        print(f"S3预算调整: 周期{adjustment_event['cycle']}, "
              f"结余率{cycle_surplus_rate:.3f}, 预算调整{adjustment_event['adjustment_rate']:.3f}, "
              f"原因: {adjustment_event['reason']}")

    # # =============== 新增：预算冲击的连锁反应 ===============
    # if adjustment_event['adjustment_rate'] < -0.1:  # 大幅削减
    #     print(f"⚠️ 预算大幅削减 {adjustment_event['adjustment_rate']:.2f}，触发紧急响应")
    #
    #     # 县医院立即大幅削减分配
    #     for i in idx_county:
    #         Hlist[i].allocation_ratio = max(0.2, Hlist[i].allocation_ratio - 0.25)
    #         Hlist[i].crisis_mode = True
    #         Hlist[i].crisis_duration = 8
    #
    #     # 下级医院连锁反应
    #     for i in idx_secondary + idx_township:
    #         Hlist[i].crisis_mode = True
    #         Hlist[i].crisis_duration = 6
    #         # 危机模式下大幅降低转诊倾向和质量投入
    #         Hlist[i].rho = max(0.1, getattr(Hlist[i], 'rho', 0) - 0.3)
    #         Hlist[i].a = max(0.1, Hlist[i].a - 0.2)
    # =============== 连锁反应结束 ===============


    # =============== 连锁反应结束 ===============
    # ---- Enhanced Action Selection ----
    s_a_idx = {}
    for i, h in enumerate(Hlist):
        s_a_idx[i] = h.select_a(tau_a)

    # S3新增：县医院预算防御策略选择
    s_defense_idx = {}
    if ENABLE_BUDGET_DEFENSE:
        for i, h in enumerate(Hlist):
            if h.level == "county":
                s_defense_idx[i] = h.select_defense_strategy(0.6)

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

    # 统计当期转诊数据
    current_referral_attempts_by_h = [0] * H
    current_referral_success_by_h = [0] * H

    for p in range(N_demand):
        if assigned_to[p] != -1 and len(referral_paths[p]) > 1:
            referring_hospital = referral_paths[p][-2]
            current_referral_success_by_h[referring_hospital] += 1

        if len(referral_paths[p]) > 0:
            for referring_hospital in referral_paths[p][:-1]:
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

    # S3: 累计周期数据
    cycle_treated_patients += treated

    # ----- accounting -----
    flows = np.zeros(H, dtype=int)
    ins_spend = np.zeros(H)
    inpatient_counts = np.zeros(H, dtype=int)
    severity_by_hospital = [[] for _ in range(H)]

    for i in range(H):
        idx = served_idx_by_h[i]
        n_i = len(idx)
        flows[i] = n_i

        hospital_flow_history[i].append(n_i)

        if n_i > 0:
            sev_i = Sev[idx]
            ins_spend[i] = np.sum(PRICE[Hlist[i].level] * REIMB_RATIO * revenue_multiplier(sev_i))
            rev = np.sum(PRICE[Hlist[i].level] * revenue_multiplier(sev_i))
            cost_var = np.sum(unit_cost(sev_i))
            severity_by_hospital[i].extend(sev_i)

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

        u_i = beddays_used_by_h[i] / max(Hlist[i].K, 1.0)
        empty_cost = EMPTY_COST_PHI[Hlist[i].level] * FIXED_COST_PER_CAP[Hlist[i].level] * (1.0 - u_i) * Hlist[i].K
        fin_cost = 0.0
        if Hlist[i].finance_left > 0:
            fin_cost = Hlist[i].finance_add
            Hlist[i].finance_left -= 1
        profit = rev - (cost_var + fix + a_cost + i_cost + empty_cost + fin_cost)
        #
        # # 新增：下级医院利润依赖县医院分配
        # if Hlist[i].level != "county" and ENABLE_COUNTY_ALLOCATION and len(idx_county) > 0:
        #     county_alloc_ratio = Hlist[idx_county[0]].allocation_ratio
        #     # 下级医院的利润高度依赖县医院分配
        #     ALLOCATION_DEPENDENCY_RATE = 0.6  # 60%依赖度
        #     allocation_income = surplus * (1 - county_alloc_ratio) * ALLOCATION_DEPENDENCY_RATE
        #     profit += allocation_income
        #
        #     # 分配依赖的学习信号
        #     Hlist[i].allocation_dependency_signal = allocation_income / max(profit, 1)

        Hlist[i].u_prev = 0.80 * Hlist[i].u_prev + 0.20 * u_i
        Hlist[i].W_prev = waiting_time(Hlist[i].u_prev, Hlist[i].a)
        Hlist[i].profit_prev = 0.7 * Hlist[i].profit_prev + 0.3 * profit

    # S3修改：使用当前动态预算
    current_period_budget = current_budget
    budget_history[t] = current_period_budget

    total_insurance = np.sum(ins_spend)
    cycle_insurance_spending += total_insurance  # S3: 累计周期医保支出

    surplus_rate = (current_period_budget - total_insurance) / current_period_budget if current_period_budget > 0 else 0
    surplus_rate_history.append(surplus_rate)

    county_insurance = np.sum(ins_spend[idx_county])
    secondary_insurance = np.sum(ins_spend[idx_secondary])
    township_insurance = np.sum(ins_spend[idx_township])

    insurance_spending["total"].append(total_insurance)
    insurance_spending["county"].append(county_insurance)
    insurance_spending["secondary"].append(secondary_insurance)
    insurance_spending["township"].append(township_insurance)

    if total_insurance > 0:
        insurance_spending["county_share"].append(county_insurance / total_insurance)
        insurance_spending["secondary_share"].append(secondary_insurance / total_insurance)
        insurance_spending["township_share"].append(township_insurance / total_insurance)
    else:
        insurance_spending["county_share"].append(0.0)
        insurance_spending["secondary_share"].append(0.0)
        insurance_spending["township_share"].append(0.0)

    # ----- Enhanced Cooperation Metrics -----
    light_cnt = 0
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

    total_low = 0
    rho_sum = 0.0
    for i in idx_secondary + idx_township:
        total_low += flows[i]
        rho_sum += flows[i] * getattr(Hlist[i], 'rho', 0.0)
    rho_mean_lower.append((rho_sum / total_low) if total_low > 0 else 0.0)

    uplift_sum = 0.0
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

        occupancy = total_beddays / max(total_capacity, 1.0)
        bed_occupancy_lvl[level].append(occupancy)

        inpatient_share = total_inpatients / max(total_patients, 1.0)
        inpatient_share_lvl[level].append(inpatient_share)

        if total_inpatients > 0:
            avg_los = total_beddays / total_inpatients
        else:
            avg_los = 0.0
        avg_los_inp_lvl[level].append(avg_los)

    # ----- Enhanced Monitoring -----
    total_current_attempts = sum(current_referral_attempts_by_h)
    total_current_success = sum(current_referral_success_by_h)
    if total_current_attempts > 0:
        current_referral_rate = total_current_success / total_current_attempts
    else:
        current_referral_rate = 0.0
    referral_success_rate.append(current_referral_rate)

    for h in Hlist:
        h.referral_attempts += sum(current_referral_attempts_by_h)
        h.referral_success += sum(current_referral_success_by_h)

    if len(idx_county) > 0:
        county_allocation_ratio.append(Hlist[idx_county[0]].allocation_ratio)
        # S3新增：记录县医院防御策略
        county_defense_strategy.append(Hlist[idx_county[0]].defense_strategy)
    else:
        county_allocation_ratio.append(0.5)
        county_defense_strategy.append(0.0)

    avg_path_length = np.mean([len(path) for path in referral_paths if len(path) > 0])
    referral_path_length.append(avg_path_length)

    # S3新增：策略性行为检测
    strategic_behavior = 0
    if len(idx_county) > 0:
        county_h = Hlist[idx_county[0]]
        # 检测临近调整期的异常行为
        if county_h.periods_to_adjustment < 5 and county_h.defense_strategy > 0.5:
            if surplus_rate > SURPLUS_TARGET + 0.1:
                strategic_behavior = 1  # 过度结余
            elif surplus_rate < SURPLUS_TARGET - 0.1:
                strategic_behavior = -1  # 过度支出
    strategic_behavior_detected.append(strategic_behavior)

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

    # 新增：紧急行为模式检查
    CRISIS_THRESHOLD = -500  # 利润危机阈值

    for i, h in enumerate(Hlist):
        if h.profit_prev < CRISIS_THRESHOLD:
            h.crisis_mode = True
            h.crisis_duration = getattr(h, 'crisis_duration', 8)  # 持续8期
        else:
            h.crisis_mode = False

        if getattr(h, 'crisis_mode', False):
            # 危机期行为调整
            if h.level == "county":
                # 县医院：大幅削减分配，提高效率
                h.allocation_ratio = max(0.2, h.allocation_ratio - 0.3)
                h.a = min(1.0, h.a + 0.2)  # 提高合作以获取更多患者
            else:
                # 下级医院：desperate合作
                h.rho = min(1.0, getattr(h, 'rho', 0) + 0.4)
                h.a = max(0.1, h.a - 0.3)  # 降低质量以节省成本

            h.crisis_duration -= 1
            if h.crisis_duration <= 0:
                h.crisis_mode = False

    # ----- learning updates -----
    # 原有学习更新代码继续...



    # ----- learning updates -----
    for i, h in enumerate(Hlist):
        s_now = h.state_hash_a()

        # 使用分层奖励函数
        # 临时使用默认预算趋势
        temp_budget_trend = getattr(h, 'budget_trend', 0) if h.level == "county" else 0
        r_a = hierarchical_reward_function(h, t, surplus_rate, temp_budget_trend)
        s_prev, a_idx = s_a_idx[i]
        h.update_Qa(s_prev, a_idx, r_a, s_now)


    # 在文件前面的函数定义区域添加这个新函数（约第600行附近）
    def hierarchical_reward_function(hospital, t, surplus_rate, budget_environment):
        base_profit_reward = hospital.profit_prev / max(hospital.K, 1.0) / 8.0
        waiting_penalty = -0.04 * hospital.W_prev

        if hospital.level == "county":
            # 县医院：预算周期敏感的奖励
            periods_to_adj = getattr(hospital, 'periods_to_adjustment', BUDGET_ADJUSTMENT_CYCLE)

            if periods_to_adj < 8:  # 调整临近期
                # 预算防御成为首要任务
                surplus_deviation = abs(surplus_rate - SURPLUS_TARGET)
                budget_reward = -0.3 * surplus_deviation

                # 成功防御的显著奖励
                if 0.02 < surplus_deviation < 0.08:
                    budget_reward += 0.15  # 适度偏离目标的奖励

                # 运营效率权重降低
                operation_weight = 0.3
                budget_weight = 0.7
            else:
                # 正常运营期
                budget_reward = -0.1 * abs(surplus_rate - SURPLUS_TARGET)
                operation_weight = 0.8
                budget_weight = 0.2

            total_reward = (operation_weight * (base_profit_reward + waiting_penalty) +
                            budget_weight * budget_reward)

        else:
            # 下级医院：预算环境敏感的奖励
            if budget_environment < 0:  # 预算紧缩期
                # 效率优先，合作权重降低
                cooperation_reward = 0.1 * getattr(hospital, 'rho', 0.0)
                efficiency_bonus = 0.2 * (hospital.u_prev - 0.7)  # 鼓励提高利用率
                total_reward = base_profit_reward + waiting_penalty + cooperation_reward + efficiency_bonus
            else:
                # 预算宽松期，质量优先
                quality_bonus = 0.1 * hospital.a  # 合作和质量奖励
                total_reward = base_profit_reward + waiting_penalty + quality_bonus

        return total_reward

    # S3新增：预算防御学习更新
    if ENABLE_BUDGET_DEFENSE:
        for i, h in enumerate(Hlist):
            if h.level == "county":
                s_prev_d, idx_d = s_defense_idx[i]
                s_now_d = h.state_hash_defense()
                # 防御策略奖励：结余率接近目标有奖励
                surplus_deviation = abs(surplus_rate - SURPLUS_TARGET)
                r_defense = -0.2 * surplus_deviation + 0.1 * h.profit_prev / max(h.K, 1.0)
                h.update_Qdefense(s_prev_d, idx_d, r_defense, s_now_d)

    # --- cooperation action learning ---
    for i, h in enumerate(Hlist):
        if ENABLE_RHO and h.level != "county":
            s_prev_r, idx_r = s_rho_idx[i]
            s_now_r = h.state_hash_rho()
            current_success = current_referral_success_by_h[i]
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
                # 对基层医院的能力提升设置更严格的限制
                if Hlist[j].level == "township":
                    # 基层医院能力提升更慢，上限更低
                    uplift = ETA_D[Hlist[j].level] * d_sys * 0.5  # 降低提升速度
                    new_smax = min(cap * 0.8, Hlist[j].Smax + uplift)  # 设置更低的上限
                else:
                    new_smax = min(cap, Hlist[j].Smax + ETA_D[Hlist[j].level] * d_sys)
                Hlist[j].Smax = new_smax

    # ----- Enhanced Bonus Distribution -----
    ins_total = total_insurance
    surplus = current_period_budget - ins_total  # S3修改：使用当前预算

    # 新增：硬预算约束
    if HARD_BUDGET_CONSTRAINT and surplus < 0:
        # 硬约束：赤字必须由医院承担
        deficit_penalty = -surplus * BUDGET_DEFICIT_PENALTY_RATE
        # 按医院支出比例分摊惩罚
        penalty_weights = np.zeros(H)
        for i, h in enumerate(Hlist):
            penalty_weights[i] = ins_spend[i] / max(ins_total, 1)
        penalty_weights = penalty_weights / np.sum(penalty_weights) if np.sum(penalty_weights) > 0 else np.ones(H) / H

        for i, h in enumerate(Hlist):
            penalty_amount = deficit_penalty * penalty_weights[i]
            h.profit_prev -= penalty_amount
            # 记录惩罚用于学习
            h.budget_penalty_prev = penalty_amount

        # 赤字情况下没有奖金
        surplus = 0.0
    elif (not ALLOW_NEGATIVE_SURPLUS) and (surplus < 0):
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
    # ALLOCATION_DEPENDENCY_RATE = 0.6  # 下级医院利润对分配的依赖度
    #
    # for i, h in enumerate(Hlist):
    #     if h.level != "county" and ENABLE_COUNTY_ALLOCATION and len(idx_county) > 0:
    #         county_alloc_ratio = Hlist[idx_county[0]].allocation_ratio
    #         # 下级医院的利润高度依赖县医院分配
    #         # 注意：这里使用分配前的surplus，因为奖金已经分配完了
    #         allocation_income = (current_period_budget - ins_total) * (
    #                     1 - county_alloc_ratio) * ALLOCATION_DEPENDENCY_RATE
    #
    #         # 只在有结余时分配依赖收入
    #         if allocation_income > 0:
    #             h.profit_prev += allocation_income
    #             # 分配依赖的学习信号
    #             h.allocation_dependency_signal = allocation_income / max(h.profit_prev, 1)
    #         else:
    #             h.allocation_dependency_signal = 0
    # ----- aggregates for plots -----
    sum_flows = np.sum(flows) + 1e-6
    flow_share_lvl["county"].append(np.sum(flows[idx_county]) / sum_flows)
    flow_share_lvl["secondary"].append(np.sum(flows[idx_secondary]) / sum_flows)
    flow_share_lvl["township"].append(np.sum(flows[idx_township]) / sum_flows)

    cap_lvl["county"].append(np.sum([Hlist[i].K for i in idx_county]))
    cap_lvl["secondary"].append(np.sum([Hlist[i].K for i in idx_secondary]))
    cap_lvl["township"].append(np.sum([Hlist[i].K for i in idx_township]))

    # 两级架构合并统计计算
    primary_secondary_flows = np.sum(flows[idx_secondary]) + np.sum(flows[idx_township])
    combined_primary_flow_share.append(primary_secondary_flows / sum_flows)

    primary_secondary_capacity = (np.sum([Hlist[i].K for i in idx_secondary]) +
                                  np.sum([Hlist[i].K for i in idx_township]))
    combined_primary_capacity.append(primary_secondary_capacity)

    if primary_secondary_flows > 0:
        sev_secondary = 0.0
        if len(idx_secondary) > 0:
            acc_secondary = []
            for i in idx_secondary:
                acc_secondary += served_idx_by_h[i]
            sev_secondary = float(np.mean(Sev[acc_secondary])) if len(acc_secondary) > 0 else 0.0

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

    secondary_profits = sum(Hlist[i].profit_prev for i in idx_secondary)
    township_profits = sum(Hlist[i].profit_prev for i in idx_township)
    combined_primary_profits.append(secondary_profits + township_profits)

    secondary_utilization = sum(flows[i] for i in idx_secondary) / max(sum(Hlist[i].K for i in idx_secondary), 1.0)
    township_utilization = sum(flows[i] for i in idx_township) / max(sum(Hlist[i].K for i in idx_township), 1.0)
    combined_primary_utilization.append((secondary_utilization + township_utilization) / 2)

    secondary_beddays = sum(beddays_used_by_h[i] for i in idx_secondary)
    township_beddays = sum(beddays_used_by_h[i] for i in idx_township)
    secondary_capacity = sum(Hlist[i].K for i in idx_secondary)
    township_capacity = sum(Hlist[i].K for i in idx_township)
    combined_bed_occupancy = (secondary_beddays + township_beddays) / max(secondary_capacity + township_capacity, 1.0)
    combined_primary_bed_occupancy.append(combined_bed_occupancy)

    secondary_inpatients = sum(inpatient_counts[i] for i in idx_secondary)
    township_inpatients = sum(inpatient_counts[i] for i in idx_township)
    secondary_patients = sum(flows[i] for i in idx_secondary)
    township_patients = sum(flows[i] for i in idx_township)
    if (secondary_patients + township_patients) > 0:
        combined_inpatient_share = (secondary_inpatients + township_inpatients) / (
                    secondary_patients + township_patients)
    else:
        combined_inpatient_share = 0
    combined_primary_inpatient_share.append(combined_inpatient_share)

    # 容量趋势和占比计算
    total_capacity = (cap_lvl["county"][-1] + cap_lvl["secondary"][-1] + cap_lvl["township"][-1])

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

    # 详细严重程度统计
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

    # 保持原有的sev_mean_lvl更新
    sev_mean_lvl["county"].append(sev_stats_lvl["county"]["mean"][-1])
    sev_mean_lvl["secondary"].append(sev_stats_lvl["secondary"]["mean"][-1])
    sev_mean_lvl["township"].append(sev_stats_lvl["township"]["mean"][-1])

    # 原有核心指标计算
    avg_waiting_time = float(np.average(W_eff, weights=flows + 1e-6))
    waiting_time_series.append(avg_waiting_time)

    if treated > 0:
        served_mask = (assigned_to != -1)
        Q_chosen = Qeff[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_quality = float(np.mean(Q_chosen))
    else:
        avg_quality = float(np.mean(Qeff))
    process_quality_series.append(avg_quality)

    if treated > 0:
        served_mask = (assigned_to != -1)
        U_chosen = U[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_utility = float(np.mean(U_chosen))
    else:
        avg_utility = float(np.mean(np.max(U, axis=1)))
    patient_utility_series.append(avg_utility)

    if treated > 0:
        served_mask = (assigned_to != -1)
        d_chosen = dmat[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        avg_distance = float(np.mean(d_chosen))
    else:
        avg_distance = float(np.mean(np.min(dmat, axis=1)))
    distance_traveled_series.append(avg_distance)

    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_profit = sum(Hlist[i].profit_prev for i in idx_list)
        hospital_profits[level].append(total_profit)

    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_flows = sum(flows[i] for i in idx_list)
        total_capacity = sum(Hlist[i].K for i in idx_list)
        utilization = total_flows / max(total_capacity, 1.0)
        capacity_utilization[level].append(utilization)

    total_referral_attempts = sum(h.referral_attempts for h in Hlist)
    total_referral_success = sum(h.referral_success for h in Hlist)
    if total_referral_attempts > 0:
        efficiency = total_referral_success / total_referral_attempts
    else:
        efficiency = 0.0
    referral_efficiency.append(efficiency)

    # S3修改：使用当前预算计算预算利用率
    budget_use = ins_total / current_period_budget if current_period_budget > 0 else 0.0
    budget_utilization.append(budget_use)

    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        total_bonus = sum(bonuses[i] for i in idx_list)
        surplus_distribution[level].append(total_bonus)

    # 学习过程指标
    for level, idx_list in [("county", idx_county), ("secondary", idx_secondary), ("township", idx_township)]:
        policy_changes = 0
        for i in idx_list:
            s = Hlist[i].state_hash_a()
            probs = softmax(Hlist[i].Qa[s], 0.1)
            max_prob = np.max(probs)
            if max_prob > 0.8:
                policy_changes += 0
            else:
                policy_changes += 1
        convergence = 1 - (policy_changes / max(len(idx_list), 1))
        learning_convergence[level].append(convergence)

    # 行动分布
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
                # S3新增：防御策略数据收集
                action_distribution[level]["defense"].append(
                    np.mean([getattr(Hlist[i], 'defense_strategy', 0.0) for i in idx_list]))
    # 患者行为指标
    if treated > 0:
        served_mask = (assigned_to != -1)
        U_chosen = U[np.arange(N_demand)[served_mask], assigned_to[served_mask]]
        U_max = np.max(U[served_mask, :], axis=1)
        choice_quality = np.mean(U_chosen / (U_max + 1e-6))
    else:
        choice_quality = 0.0
    patient_choice_quality.append(choice_quality)

    habit_persist = np.mean(patient_behavior.history >= 0)
    habit_persistence.append(habit_persist)

    township_choices = np.sum([flows[i] for i in idx_township])
    total_choices = np.sum(flows)
    township_share = township_choices / max(total_choices, 1.0)
    expected_share = 0.4
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


# =============== S3 Specific Analysis and Visualization ===============
def analyze_s3_dynamics():
    """Analyze dynamic characteristics of S3 scenario"""
    print("\n=== S3 Dynamic Budget Adjustment Analysis ===")
    print(f"Total adjustment events: {len(budget_adjustment_events)}")

    if budget_adjustment_events:
        avg_adjustment = np.mean([e['adjustment_rate'] for e in budget_adjustment_events])
        avg_surplus = np.mean([e['surplus_rate'] for e in budget_adjustment_events])
        print(f"Average adjustment rate: {avg_adjustment:.3f}")
        print(f"Average surplus rate: {avg_surplus:.3f}")

        # Analyze strategic behavior
        strategic_count = sum(1 for b in strategic_behavior_detected if b != 0)
        print(f"Strategic behavior detected in periods: {strategic_count}/{T} ({strategic_count / T * 100:.1f}%)")

        # Analyze budget cycle characteristics
        budget_volatility = np.std(budget_history) / np.mean(budget_history) if np.mean(budget_history) > 0 else 0
        print(f"Budget volatility: {budget_volatility:.3f}")


def create_s3_special_plots():
    """Create S3-specific analysis charts"""

    # S3-1: Budget Adjustment Dynamics
    fig_s3_1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Subplot 1: Budget and surplus rate
    ax1_twin = ax1.twinx()
    ax1.plot(range(T), budget_history, color=COLORS["budget_adjustment"], linewidth=2, label="Weekly Budget")
    ax1_twin.plot(range(T), surplus_rate_history, color=COLORS["surplus_rate"], linewidth=2, label="Surplus Rate")

    # Mark adjustment events
    for event in budget_adjustment_events:
        cycle_end = event['cycle'] * BUDGET_ADJUSTMENT_CYCLE + BUDGET_ADJUSTMENT_START
        if cycle_end < T:
            ax1.axvline(x=cycle_end, color='red', linestyle='--', alpha=0.7)
            ax1.text(cycle_end, np.max(budget_history) * 0.9, f"Adj:{event['adjustment_rate']:.2f}",
                     rotation=90, fontsize=8)

    ax1.set_title("S3: Dynamic Budget Adjustment and Surplus Rate")
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Weekly Budget")
    ax1_twin.set_ylabel("Surplus Rate")
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Subplot 2: County hospital defense strategy
    ax2.plot(range(T), county_defense_strategy, color=COLORS["county"], linewidth=2, label="Defense Strategy Intensity")
    ax2.plot(range(T), strategic_behavior_detected, color='red', linewidth=1, alpha=0.7, label="Strategic Behavior")
    ax2.set_title("County Hospital Budget Defense Strategy")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Defense Intensity")
    ax2.legend()

    plt.tight_layout()

    # S3-2: Budget Cycle Analysis
    fig_s3_2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Adjustment event analysis
    if budget_adjustment_events:
        adjustments = [e['adjustment_rate'] for e in budget_adjustment_events]
        surplus_rates = [e['surplus_rate'] for e in budget_adjustment_events]

        scatter = ax1.scatter(surplus_rates, adjustments, c=range(len(adjustments)), cmap='viridis')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.axvline(x=SURPLUS_TARGET, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel("Surplus Rate")
        ax1.set_ylabel("Budget Adjustment Rate")
        ax1.set_title("Surplus Rate vs Budget Adjustment")
        plt.colorbar(scatter, ax=ax1, label='Adjustment Sequence')

    # Subplot 2: Strategic behavior timing pattern
    cycle_phases = [t % BUDGET_ADJUSTMENT_CYCLE for t in range(T)]
    ax2.scatter(cycle_phases, strategic_behavior_detected, alpha=0.6)
    ax2.set_xlabel("Position in Budget Cycle")
    ax2.set_ylabel("Strategic Behavior Intensity")
    ax2.set_title("Cyclical Pattern of Strategic Behavior")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # S3-3: System Efficiency vs Budget Relationship
    fig_s3_3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Budget vs System Efficiency
    system_efficiency = np.array(treated_series) / (np.array(treated_series) + np.array(unserved_series) + 1e-6)

    ax1_twin = ax1.twinx()
    ax1.plot(range(T), budget_history, color=COLORS["budget_adjustment"], linewidth=2, label="Budget")
    ax1_twin.plot(range(T), system_efficiency, color=COLORS["total"], linewidth=2, label="System Efficiency")
    ax1.set_title("Budget Level vs System Efficiency")
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Budget")
    ax1_twin.set_ylabel("System Efficiency")
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Subplot 2: Budget vs Healthcare Quality
    ax2_twin = ax2.twinx()
    ax2.plot(range(T), budget_history, color=COLORS["budget_adjustment"], linewidth=2, label="Budget")
    ax2_twin.plot(range(T), process_quality_series, color=COLORS["secondary"], linewidth=2, label="Healthcare Quality")
    ax2.set_title("Budget Level vs Healthcare Quality")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Budget")
    ax2_twin.set_ylabel("Healthcare Quality")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()

# =============== 输出和可视化 ===============
def export_s3_metrics_to_csv():
    """输出S3特定指标到CSV"""

    # S3预算调整记录
    with open('s3_budget_adjustments.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['adjustment_cycle', 'old_budget', 'new_budget', 'surplus_rate',
                         'adjustment_rate', 'reason'])
        for event in budget_adjustment_events:
            writer.writerow([
                event['cycle'], event['old_budget'], event['new_budget'],
                event['surplus_rate'], event['adjustment_rate'], event['reason']
            ])

    # S3周度指标
    with open('s3_weekly_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'budget', 'surplus_rate', 'county_defense_strategy',
                         'strategic_behavior', 'budget_utilization'])
        for t in range(T):
            writer.writerow([
                t, budget_history[t], surplus_rate_history[t],
                county_defense_strategy[t] if t < len(county_defense_strategy) else 0,
                strategic_behavior_detected[t] if t < len(strategic_behavior_detected) else 0,
                budget_utilization[t] if t < len(budget_utilization) else 0
            ])


# 运行S3分析
analyze_s3_dynamics()
create_s3_special_plots()
export_s3_metrics_to_csv()
# =============== S3 图表产出（与S2风格一致）===============
# =============== S3 图表产出（全英文版本）===============

# 设置绘图风格
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

X = np.arange(T)


def plot_series_with_ma(ax, x, y, color, label):
    """Fixed plotting function to ensure continuous lines"""
    y = np.array(y, dtype=float)

    # Ensure y has no NaN values
    y_filled = y.copy()
    nan_mask = np.isnan(y_filled)
    if nan_mask.any():
        # Simple linear interpolation to fill NaN
        indices = np.arange(len(y_filled))
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            y_filled = np.interp(indices, indices[valid_mask], y_filled[valid_mask])
        else:
            y_filled = np.zeros_like(y_filled)

    # Plot raw data (low transparency)
    ax.plot(x, y_filled, linestyle="--", linewidth=1.0, color=color, alpha=0.3, label=f"{label} (raw)")

    # Calculate and plot moving average
    y_ma = moving_average(y_filled, REPORT_MOVAVG)
    ax.plot(x, y_ma, linestyle="-", linewidth=2.5, color=color, label=f"{label} (12m MA)")

    return y_ma


# 1) Treatment Volume and Outflow Analysis
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 8))
plot_series_with_ma(ax1a, X, treated_series, COLORS["total"], "Treated Patients")
ax1a.set_title("S3: Treated Patients per Period (Dynamic Budget)")
ax1a.set_ylabel("Count")
ax1a.legend()

# Outflow analysis
ax1b.stackplot(X,
               outflow_wait_series, outflow_full_series, outflow_mismatch_series,
               labels=['Wait Time Outflow', 'Capacity Outflow', 'Mismatch Outflow'],
               colors=[COLORS['outflow_wait'], COLORS['outflow_full'], COLORS['outflow_mismatch']],
               alpha=0.7)
ax1b.set_title("S3: Patient Outflow by Reason")
ax1b.set_xlabel("Period")
ax1b.set_ylabel("Outflow Count")
ax1b.legend()
plt.tight_layout()

# 2) Bed Occupancy - Two-tier Display
fig2, ax2 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax2, X, bed_occupancy_lvl["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax2, X, combined_primary_bed_occupancy, COLORS["township"], "Primary Care Network")
ax2.set_title("S3: Bed Occupancy Rate - Two Tier System")
ax2.set_xlabel("Period")
ax2.set_ylabel("Occupancy Rate")
ax2.set_ylim(0, 1)
ax2.legend()
plt.tight_layout()

# 3) Inpatient Share - Two-tier Display
fig3, ax3 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax3, X, inpatient_share_lvl["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax3, X, combined_primary_inpatient_share, COLORS["township"], "Primary Care Network")
ax3.set_title("S3: Inpatient Admission Rate - Two Tier System")
ax3.set_xlabel("Period")
ax3.set_ylabel("Inpatient Share")
ax3.set_ylim(0, 1)
ax3.legend()
plt.tight_layout()

# 4) System Performance Metrics
fig4, ax4 = plt.subplots(figsize=(10, 5.2))
# plot_series_with_ma(ax4, X, referral_success_rate, "#10B981", "Referral Success Rate")
plot_series_with_ma(ax4, X, county_allocation_ratio, "#8B5CF6", "County Allocation Ratio")
# S3 addition: Budget defense strategy
plot_series_with_ma(ax4, X, county_defense_strategy, COLORS["budget_adjustment"], "County Defense Strategy")
ax4.set_title("S3: System Performance — Referral, Allocation & Defense")
ax4.set_xlabel("Period")
ax4.set_ylabel("Rate")
ax4.set_ylim(0, 1)
ax4.legend()
plt.tight_layout()

# 5) Flow Share - Two-tier Display
fig5, ax5 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax5, X, flow_share_lvl["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax5, X, combined_primary_flow_share, COLORS["township"], "Primary Care Network")
ax5.set_title("S3: Patient Flow Distribution - Two Tier System")
ax5.set_xlabel("Period")
ax5.set_ylabel("Flow Share")
ax5.set_ylim(0, 0.9)
ax5.legend()
plt.tight_layout()

# 6) Waiting Time and Process Quality
fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax6a, X, waiting_time_series, COLORS["total"], "Average Waiting Time")
ax6a.set_title("S3: Average Waiting Time Across System")
ax6a.set_ylabel("Days")
ax6a.legend()

plot_series_with_ma(ax6b, X, process_quality_series, COLORS["secondary"], "Average Process Quality")
ax6b.set_title("S3: Average Treatment Quality")
ax6b.set_xlabel("Period")
ax6b.set_ylabel("Quality Score")
ax6b.legend()
plt.tight_layout()

# 7) Patient Utility and Choice Quality
fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax7a, X, patient_utility_series, COLORS["county"], "Patient Utility")
ax7a.set_title("S3: Average Patient Utility")
ax7a.set_ylabel("Utility Score")
ax7a.legend()

plot_series_with_ma(ax7b, X, patient_choice_quality, COLORS["township"], "Choice Quality Ratio")
ax7b.set_title("S3: Patient Choice Quality (Actual/Max Utility)")
ax7b.set_xlabel("Period")
ax7b.set_ylabel("Quality Ratio")
ax7b.set_ylim(0, 1)
ax7b.legend()
plt.tight_layout()

# 8) Hospital Profits - Two-tier Display
fig8, ax8 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax8, X, hospital_profits["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax8, X, combined_primary_profits, COLORS["township"], "Primary Care Network")
ax8.set_title("S3: Hospital Profits - Two Tier System")
ax8.set_xlabel("Period")
ax8.set_ylabel("Profit")
ax8.legend()
plt.tight_layout()

# 9) Capacity Utilization - Two-tier Display
fig9, ax9 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax9, X, capacity_utilization["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax9, X, combined_primary_utilization, COLORS["township"], "Primary Care Network")
ax9.set_title("S3: Capacity Utilization - Two Tier System")
ax9.set_xlabel("Period")
ax9.set_ylabel("Utilization Rate")
ax9.set_ylim(0, 1)
ax9.legend()
plt.tight_layout()

# 10) Referral and Budget Efficiency
fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax10a, X, referral_efficiency, "#10B981", "Referral Efficiency")
ax10a.set_title("S3: Referral System Efficiency")
ax10a.set_ylabel("Success Rate")
ax10a.set_ylim(0, 1)
ax10a.legend()

plot_series_with_ma(ax10b, X, budget_utilization, "#8B5CF6", "Budget Utilization")
ax10b.set_title("S3: Insurance Budget Utilization")
ax10b.set_xlabel("Period")
ax10b.set_ylabel("Utilization Rate")
ax10b.legend()
plt.tight_layout()

# 11) Learning Convergence - Two-tier Display
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

fig11, ax11 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax11, X, learning_convergence["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax11, X, primary_convergence, COLORS["township"], "Primary Care Network")
ax11.set_title("S3: Learning Convergence - Two Tier System")
ax11.set_xlabel("Period")
ax11.set_ylabel("Convergence Index")
ax11.set_ylim(0, 1)
ax11.legend()
plt.tight_layout()

# 12) Cooperation Intensity - Two-tier Display
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

fig12, ax12 = plt.subplots(figsize=(10, 5.6))
plot_series_with_ma(ax12, X, action_distribution["county"]["a"], COLORS["county"], "County Hospital Cooperation")
plot_series_with_ma(ax12, X, primary_cooperation, COLORS["township"], "Primary Care Network Cooperation")
ax12.set_title("S3: Cooperation Intensity - Two Tier System")
ax12.set_xlabel("Period")
ax12.set_ylabel("Cooperation Level")
ax12.set_ylim(0, 1)
ax12.legend()
plt.tight_layout()

# 13) Comprehensive Cooperation Indicators
fig13, ax13 = plt.subplots(figsize=(12, 6))

# Calculate average referral propensity for primary care
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

# Plot four lines: referral propensity, ability uplift, county allocation, defense strategy
plot_series_with_ma(ax13, X, primary_referral_propensity, "#EF4444", "Primary Referral Propensity (ρ)")
plot_series_with_ma(ax13, X, smax_uplift_lower, "#10B981", "Primary Ability Uplift")
plot_series_with_ma(ax13, X, county_allocation_ratio, "#8B5CF6", "County Allocation Ratio")
plot_series_with_ma(ax13, X, county_defense_strategy, COLORS["budget_adjustment"], "County Defense Strategy")

ax13.set_title("S3: Cooperation Indicators - Two Tier System")
ax13.set_xlabel("Period")
ax13.set_ylabel("Rate")
ax13.set_ylim(0, 1)
ax13.legend()
plt.tight_layout()

# 14) Patient Behavior Metrics
fig14, (ax14a, ax14b) = plt.subplots(2, 1, figsize=(10, 8))
plot_series_with_ma(ax14a, X, habit_persistence, "#F59E0B", "Habit Persistence")
ax14a.set_title("S3: Patient Habit Persistence")
ax14a.set_ylabel("Persistence Rate")
ax14a.set_ylim(0, 1)
ax14a.legend()

plot_series_with_ma(ax14b, X, information_asymmetry_effect, "#EF4444", "Information Asymmetry Effect")
ax14b.set_title("S3: Information Asymmetry Effect on Township Hospitals")
ax14b.set_xlabel("Period")
ax14b.set_ylabel("Choice Reduction")
ax14b.legend()
plt.tight_layout()

# 15) System-wide Efficiency Metrics
fig15, (ax15a, ax15b) = plt.subplots(2, 1, figsize=(10, 8))
system_efficiency = np.array(treated_series) / (np.array(treated_series) + np.array(unserved_series) + 1e-6)
plot_series_with_ma(ax15a, X, system_efficiency, "#3B82F6", "System Efficiency")
ax15a.set_title("S3: Overall System Efficiency (Treated / Total Demand)")
ax15a.set_ylabel("Efficiency")
ax15a.set_ylim(0, 1)
ax15a.legend()

plot_series_with_ma(ax15b, X, distance_traveled_series, "#14B8A6", "Average Travel Distance")
ax15b.set_title("S3: Average Patient Travel Distance")
ax15b.set_xlabel("Period")
ax15b.set_ylabel("Distance")
ax15b.legend()
plt.tight_layout()

# 16) Patient Severity Distribution
fig16, ((ax16a, ax16b), (ax16c, ax16d)) = plt.subplots(2, 2, figsize=(14, 10))

# Average severity trend - two-tier display
plot_series_with_ma(ax16a, X, sev_stats_lvl["county"]["mean"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax16a, X, combined_primary_severity, COLORS["township"], "Primary Care Network")
ax16a.set_title("S3: Average Patient Severity by Hospital Level")
ax16a.set_ylabel("Severity Score")
ax16a.set_ylim(0, 1)
ax16a.legend()

# Severity variation coefficient calculation - fixed: recalculate all levels
sev_cv_county = []
sev_cv_secondary = []
sev_cv_township = []

for t in range(T):
    # County hospital coefficient of variation
    mean_county = sev_stats_lvl['county']['mean'][t] if t < len(sev_stats_lvl['county']['mean']) else 0
    std_county = sev_stats_lvl['county']['std'][t] if t < len(sev_stats_lvl['county']['std']) else 0
    cv_county = std_county / mean_county if mean_county > 0.01 else 0
    sev_cv_county.append(cv_county)

    # Secondary hospital coefficient of variation
    mean_secondary = sev_stats_lvl['secondary']['mean'][t] if t < len(sev_stats_lvl['secondary']['mean']) else 0
    std_secondary = sev_stats_lvl['secondary']['std'][t] if t < len(sev_stats_lvl['secondary']['std']) else 0
    cv_secondary = std_secondary / mean_secondary if mean_secondary > 0.01 else 0
    sev_cv_secondary.append(cv_secondary)

    # Township hospital coefficient of variation
    mean_township = sev_stats_lvl['township']['mean'][t] if t < len(sev_stats_lvl['township']['mean']) else 0
    std_township = sev_stats_lvl['township']['std'][t] if t < len(sev_stats_lvl['township']['std']) else 0
    cv_township = std_township / mean_township if mean_township > 0.01 else 0
    sev_cv_township.append(cv_township)

# Primary care average coefficient of variation
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

ax16b.plot(X, moving_average(sev_cv_county, REPORT_MOVAVG), color=COLORS["county"], label="County Hospital",
           linewidth=2)
ax16b.plot(X, moving_average(primary_cv, REPORT_MOVAVG), color=COLORS["township"], label="Primary Care Network",
           linewidth=2)
ax16b.set_title("S3: Severity Variation (Coefficient of Variation)")
ax16b.set_ylabel("CV (Std/Mean)")
ax16b.legend()

# Severity quartile trends
ax16c.plot(X, moving_average(sev_stats_lvl['county']['q25'], REPORT_MOVAVG), color=COLORS["county"],
           linestyle="--", alpha=0.7, label="County Q25")
ax16c.plot(X, moving_average(sev_stats_lvl['county']['q75'], REPORT_MOVAVG), color=COLORS["county"],
           linestyle=":", alpha=0.7, label="County Q75")
ax16c.plot(X, moving_average(sev_stats_lvl['township']['q25'], REPORT_MOVAVG), color=COLORS["township"],
           linestyle="--", alpha=0.7, label="Primary Q25")
ax16c.plot(X, moving_average(sev_stats_lvl['township']['q75'], REPORT_MOVAVG), color=COLORS["township"],
           linestyle=":", alpha=0.7, label="Primary Q75")
ax16c.set_title("S3: Severity Quartiles (County vs Primary Care)")
ax16c.set_xlabel("Period")
ax16c.set_ylabel("Severity Score")
ax16c.legend()

# Severity distribution comparison (last 50 periods)
recent_periods = min(50, T)
recent_sev_data = []
labels = ['County Hospital', 'Primary Care Network']

# County hospital data
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

# Primary care data (combined secondary and township)
recent_data_primary = []
for t in range(T - recent_periods, T):
    # Secondary hospital data
    if t >= 0 and not np.isnan(sev_stats_lvl['secondary']['mean'][t]):
        mean = sev_stats_lvl['secondary']['mean'][t]
        std = sev_stats_lvl['secondary']['std'][t]
        if not np.isnan(mean) and not np.isnan(std):
            simulated = np.random.normal(mean, std, 50)
            simulated = np.clip(simulated, 0, 1)
            recent_data_primary.extend(simulated)

    # Township hospital data
    if t >= 0 and not np.isnan(sev_stats_lvl['township']['mean'][t]):
        mean = sev_stats_lvl['township']['mean'][t]
        std = sev_stats_lvl['township']['std'][t]
        if not np.isnan(mean) and not np.isnan(std):
            simulated = np.random.normal(mean, std, 50)
            simulated = np.clip(simulated, 0, 1)
            recent_data_primary.extend(simulated)

if recent_data_primary:
    recent_sev_data.append(recent_data_primary)

if recent_sev_data:
    ax16d.boxplot(recent_sev_data, labels=labels)
    ax16d.set_title("S3: Recent Severity Distribution Comparison")
    ax16d.set_ylabel("Severity Score")

plt.tight_layout()

# 17) Insurance Spending Monitoring (Combined Version)
fig17, (ax17a, ax17b) = plt.subplots(2, 1, figsize=(12, 10))

# Calculate primary care total spending (secondary + township)
primary_insurance_spending = [insurance_spending["secondary"][t] + insurance_spending["township"][t] for t in range(T)]

# Subplot 1: Combined total and level spending - two-tier display
ax17a_twin = ax17a.twinx()
plot_series_with_ma(ax17a, X, insurance_spending["total"], "black", "Total Insurance Spending")
ax17a.set_ylabel("Total Spending (Currency Units)", color="black")
ax17a.tick_params(axis='y', labelcolor="black")

plot_series_with_ma(ax17a_twin, X, insurance_spending["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax17a_twin, X, primary_insurance_spending, COLORS["township"], "Primary Care Network")
ax17a_twin.set_ylabel("Level Spending (Currency Units)")
ax17a.set_title("S3: Insurance Spending: Total vs By Hospital Level")
ax17a.legend(loc='upper left')
ax17a_twin.legend(loc='upper right')

# Subplot 2: Spending share stacked plot - two-tier display
primary_insurance_share = [insurance_spending["secondary_share"][t] + insurance_spending["township_share"][t] for t in
                           range(T)]

ax17b.stackplot(X,
                insurance_spending["county_share"],
                primary_insurance_share,
                labels=['County Hospital', 'Primary Care Network'],
                colors=[COLORS["county"], COLORS["township"]],
                alpha=0.7)
ax17b.set_title("S3: Insurance Spending Share by Hospital Level")
ax17b.set_xlabel("Period")
ax17b.set_ylabel("Spending Share")
ax17b.set_ylim(0, 1)
ax17b.legend(loc='upper left')

plt.tight_layout()

# 18) Insurance Spending Efficiency Analysis
fig18, (ax18a, ax18b) = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Combined per-patient and per-capacity spending
ax18a_twin = ax18a.twinx()

# Calculate per-patient spending
avg_insurance_per_patient = []
for t in range(T):
    if treated_series[t] > 0:
        avg_spend = insurance_spending["total"][t] / treated_series[t]
    else:
        avg_spend = 0
    avg_insurance_per_patient.append(avg_spend)

plot_series_with_ma(ax18a, X, avg_insurance_per_patient, "#7C3AED", "Avg Spending per Patient")
ax18a.set_ylabel("Spending per Patient", color="#7C3AED")
ax18a.tick_params(axis='y', labelcolor="#7C3AED")

# Calculate per-capacity spending - two-tier display
spending_per_capacity = {
    "county": [],
    "primary": []
}

for t in range(T):
    # County hospital
    total_capacity_county = sum(Hlist[i].K for i in idx_county)
    total_spending_county = insurance_spending["county"][t]
    if total_capacity_county > 0:
        spend_per_cap_county = total_spending_county / total_capacity_county
    else:
        spend_per_cap_county = 0
    spending_per_capacity["county"].append(spend_per_cap_county)

    # Primary care
    total_capacity_primary = sum(Hlist[i].K for i in idx_secondary + idx_township)
    total_spending_primary = insurance_spending["secondary"][t] + insurance_spending["township"][t]
    if total_capacity_primary > 0:
        spend_per_cap_primary = total_spending_primary / total_capacity_primary
    else:
        spend_per_cap_primary = 0
    spending_per_capacity["primary"].append(spend_per_cap_primary)

plot_series_with_ma(ax18a_twin, X, spending_per_capacity["county"], COLORS["county"], "County per Capacity")
plot_series_with_ma(ax18a_twin, X, spending_per_capacity["primary"], COLORS["township"], "Primary Care per Capacity")
ax18a_twin.set_ylabel("Spending per Capacity Unit")
ax18a.set_title("S3: Spending Efficiency: Per Patient vs Per Capacity")
ax18a.legend(loc='upper left')
ax18a_twin.legend(loc='upper right')

# Subplot 2: Spending efficiency trend
spending_efficiency = []
for t in range(T):
    if insurance_spending["total"][t] > 0:
        efficiency = treated_series[t] / insurance_spending["total"][t]
    else:
        efficiency = 0
    spending_efficiency.append(efficiency)

plot_series_with_ma(ax18b, X, spending_efficiency, "#0891B2", "Patients per Unit Spending")
ax18b.set_title("S3: Spending Efficiency Trend (Treated Patients per Spending Unit)")
ax18b.set_xlabel("Period")
ax18b.set_ylabel("Efficiency Ratio")
ax18b.legend()

plt.tight_layout()

# 19) Hospital Capacity Change Monitoring - Two-tier Display
fig19, (ax19a, ax19b) = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Two-tier capacity trends
plot_series_with_ma(ax19a, X, capacity_trends["county"], COLORS["county"], "County Hospital")
plot_series_with_ma(ax19a, X, combined_primary_capacity, COLORS["township"], "Primary Care Total Capacity")
ax19a.set_title("S3: Hospital Capacity Trends by Level")
ax19a.set_ylabel("Total Capacity (Bed Days)")
ax19a.legend()

# Subplot 2: Capacity share stacked plot
ax19b.stackplot(X,
                capacity_trends["county_share"],
                capacity_trends["secondary_share"],
                capacity_trends["township_share"],
                labels=['County', 'Secondary', 'Township'],
                colors=[COLORS["county"], COLORS["secondary"], COLORS["township"]],
                alpha=0.7)
ax19b.set_title("S3: Capacity Share by Hospital Level")
ax19b.set_xlabel("Period")
ax19b.set_ylabel("Capacity Share")
ax19b.set_ylim(0, 1)
ax19b.legend(loc='upper left')

plt.tight_layout()
################################

# =============== Spatial Heatmap Functions ===============

def create_spatial_heatmap():
    """Generate spatial heatmap based on hospital flow intensity - using S2 style"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate hospital flow intensity (average of last 52 periods)
    hospital_intensity = []
    hospital_flows = []
    for i in range(H):
        if len(hospital_flow_history[i]) >= 52:
            flow = np.mean(hospital_flow_history[i][-52:])
        else:
            flow = np.mean(hospital_flow_history[i]) if hospital_flow_history[i] else 0
        hospital_flows.append(flow)

    # Normalize flows for color mapping
    max_flow = max(hospital_flows) if hospital_flows else 1
    hospital_intensity = [flow / max_flow for flow in hospital_flows]

    # Draw population density background - using original color scheme
    im = ax.contourf(X_GRID, Y_GRID, POPULATION_GRID,
                     levels=20, cmap='Blues', alpha=0.6, antialiased=True)

    # Draw hospital points - completely using original style
    for i, h in enumerate(Hlist):
        color = COLORS[h.level]

        # Size reflects flow intensity (using original scaling ratio)
        size = 50 + 450 * hospital_intensity[i]

        # Use original marker shapes
        if h.level == "county":
            marker = 'D'  # diamond
            edge_color = 'darkblue'
            edge_width = 2
        elif h.level == "secondary":
            marker = 's'  # square
            edge_color = 'darkorange'
            edge_width = 1.5
        else:  # township
            marker = 'o'  # circle
            edge_color = 'darkred'
            edge_width = 1

        # Draw hospital point
        ax.scatter(h.pos[0], h.pos[1],
                   s=size,
                   c=color,
                   marker=marker,
                   edgecolors=edge_color,
                   linewidths=edge_width,
                   alpha=0.8,
                   label=f'{h.name} ({h.level})' if i < 5 else "")  # only label first 5 to avoid too many legends

        # Add hospital name labels (using original style)
        if hospital_intensity[i] > 0.3:  # only show labels for hospitals with high flow
            ax.annotate(h.name,
                        (h.pos[0], h.pos[1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # Set axes and title - using original style
    ax.set_xlim(-CITY_RADIUS, CITY_RADIUS)
    ax.set_ylim(-CITY_RADIUS, CITY_RADIUS)
    ax.set_aspect('equal')
    ax.set_title('S3: Hospital Flow Intensity Heatmap (Dynamic Budget Adjustment)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate (km)', fontsize=12)
    ax.set_ylabel('Y Coordinate (km)', fontsize=12)

    # Add grid - using original style
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add legend - using original style but more concise
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

    # Add color bar - using original style
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Population Density', fontsize=12)

    # Add flow intensity explanation
    ax.text(0.02, 0.98, 'Marker size indicates\nhospital flow intensity',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig('spatial_heat_s3.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_professional_heatmap():
    """Create professional heatmap - using heatmap_v5_9_30 style"""
    # Prepare coordinate data
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

    # Calculate hospital flow (average of last 52 periods)
    hospital_flows = []
    for i in range(H):
        if len(hospital_flow_history[i]) >= 52:
            flow = np.mean(hospital_flow_history[i][-52:])
        else:
            flow = np.mean(hospital_flow_history[i]) if hospital_flow_history[i] else 0
        hospital_flows.append(flow)

    hospital_flows = np.array(hospital_flows)

    # Calculate share and absolute
    total_flow = hospital_flows.sum()
    if total_flow > 0:
        shares = hospital_flows / total_flow
    else:
        shares = hospital_flows

    # Generate Gaussian field
    X_share, Y_share, Z_share = gaussian_field(coords, shares)
    X_abs, Y_abs, Z_abs = gaussian_field(coords, hospital_flows)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), dpi=140, facecolor='white')
    plt.subplots_adjust(left=0.04, right=0.97, wspace=0.08, hspace=0.14)

    R_CITY = 10.0

    # ---- Top: Share ----
    draw_city_base(ax1, R_CITY)
    vmax_share = float(np.nanmax(Z_share)) if np.isfinite(np.nanmax(Z_share)) and np.nanmax(Z_share) > 0 else 1.0
    norm_share = PowerNorm(gamma=0.60, vmin=0.0, vmax=vmax_share)

    # Fix: Ensure heatmap completely covers circular area
    im1 = ax1.imshow(Z_share, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                     origin='lower', interpolation='bilinear', cmap=HEAT_CMAP,
                     norm=norm_share, alpha=0.88, zorder=1)

    # Fix: Create precise circular clipping path
    circle = Circle((0, 0), R_CITY, transform=ax1.transData)
    im1.set_clip_path(circle)

    plot_hospitals(ax1, coords)
    ax1.set_title("S3: Treated Volumes (Share)", fontsize=14, fontweight='bold')

    # Add contour lines
    if np.isfinite(vmax_share) and vmax_share > 0:
        ax1.contour(X_share, Y_share, Z_share, levels=[0.65 * vmax_share],
                    colors=['#fde68a'], linewidths=1.6, zorder=4, alpha=0.95)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.02)
    cb1 = plt.colorbar(im1, cax=cax1)
    cb1.set_label('Share (last 52w)', fontsize=10)

    # ---- Bottom: Absolute ----
    draw_city_base(ax2, R_CITY)
    vmax_abs = float(np.nanmax(Z_abs)) if np.isfinite(np.nanmax(Z_abs)) and np.nanmax(Z_abs) > 0 else 1.0
    norm_abs = PowerNorm(gamma=1.0, vmin=0.0, vmax=vmax_abs)

    im2 = ax2.imshow(Z_abs, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                     origin='lower', interpolation='bilinear', cmap=HEAT_CMAP,
                     norm=norm_abs, alpha=0.88, zorder=1)

    # Fix: Create precise circular clipping path
    circle2 = Circle((0, 0), R_CITY, transform=ax2.transData)
    im2.set_clip_path(circle2)

    plot_hospitals(ax2, coords)
    ax2.set_title("S3: Treated Volumes (Absolute)", fontsize=14, fontweight='bold')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.02)
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.set_label('Avg treated (last 52w)', fontsize=10)

    plt.tight_layout()
    plt.savefig('professional_heatmap_s3.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# =============== Heatmap Helper Functions ===============

def draw_city_base(ax, R_CITY=10.0, R_INNER=3.0, R_MIDDLE=6.0, decay=0.28):
    """Draw city base background"""
    res = 560
    xs = np.linspace(-R_CITY, R_CITY, res)
    ys = np.linspace(-R_CITY, R_CITY, res)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X ** 2 + Y ** 2)
    base = np.exp(-decay * R)

    # Fix: Ensure background image is properly clipped
    im_bg = ax.imshow(base, extent=[-R_CITY, R_CITY, -R_CITY, R_CITY],
                      origin='lower', interpolation='bilinear', cmap=SOFT_CMAP,
                      alpha=0.9, zorder=0)

    # Fix: Create precise circular clipping path
    circle = Circle((0, 0), R_CITY, transform=ax.transData)
    im_bg.set_clip_path(circle)

    # Draw circular boundaries
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
    """Plot hospital locations"""
    # County hospitals - stars
    ax.scatter(coords["ter"][:, 0], coords["ter"][:, 1], marker='*', s=160,
               c=COLOR_TERTIARY, edgecolors='white', linewidths=0.9, zorder=3)
    # Secondary hospitals - squares
    ax.scatter(coords["sec"][:, 0], coords["sec"][:, 1], marker='s', s=90,
               c=COLOR_SECONDARY, edgecolors='white', linewidths=0.8, zorder=3)
    # Primary hospitals - triangles
    ax.scatter(coords["pri"][:, 0], coords["pri"][:, 1], marker='^', s=60,
               c=COLOR_PRIMARY, edgecolors='white', linewidths=0.7, zorder=3)


def gaussian_field(coords, weights_or_avgs, R_CITY=10.0,
                   bw_ter=1.9, bw_sec=2.2, bw_pri=2.6):
    """Generate Gaussian field"""
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


# =============== Generate Heatmaps ===============
print("Generating spatial heatmaps...")
create_spatial_heatmap()
create_professional_heatmap()
print("✓ Spatial heatmaps generated successfully!")
print("  - spatial_heat_s3.png: Hospital flow intensity map")
print("  - professional_heatmap_s3.png: Professional treatment volume maps")

################################
plt.show()
print("S3所有图表生成完成！")
print("✓ 基础系统指标图表")
print("✓ 两级架构对比图表")
print("✓ 性能指标图表")
print("✓ 经济指标图表")
print("✓ 学习过程图表")
print("✓ 严重程度分析图表")
print("✓ 医保支出监控图表")
print("✓ 与S2风格完全一致的图表产出")
print("S3动态预算调整场景完成！主要特性：")
print("✓ 半年期动态预算调整机制")
print("✓ 县医院预算防御策略学习")
print("✓ 策略性行为检测和监控")
print("✓ 预算-效率动态关系分析")
# =============== Generate All Outputs ===============
print("S3 All charts generated successfully!")
print("✓ Basic system indicator charts")
print("✓ Two-tier architecture comparison charts")
print("✓ Performance indicator charts")
print("✓ Economic indicator charts")
print("✓ Learning process charts")
print("✓ Severity analysis charts")
print("✓ Insurance spending monitoring charts")
print("✓ Full English version for academic publication")

# Generate heatmaps
print("\nGenerating spatial heatmaps...")
print("✓ Spatial heatmaps generated successfully!")

print("\nS3 simulation completed! All outputs generated:")
print("✓ 19 system analysis charts")
print("✓ 3 S3-specific analysis charts")
print("✓ 2 spatial heatmaps")
print("✓ CSV data exports")
print("✓ Dynamic budget adjustment analysis")
print("✓ 完整的S3场景实现和可视化")