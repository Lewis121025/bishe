"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 实证分析脚本
=============================================================================

参考论文：刘剑民(2019)《政府补助、管理层权力与国有企业高管超额薪酬》
开题报告：基于数据挖掘的上市公司财政补贴与高管超额薪酬研究

研究模型：
  模型1（期望薪酬）: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ΣYear + ε
  模型2（主回归）: Overpay = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + μᵢ + λₜ + ε
  模型3（中介总效应）: Overpay = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + μᵢ + λₜ + ε
  模型4（中介路径a）: Power = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + μᵢ + λₜ + ε
  模型5（中介路径b与直接效应）: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + α₃·Roa + α₄·Lever + α₅·Top1 + μᵢ + λₜ + ε
"""

import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.multivariate.factor import Factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 行业管制口径（参考文献常见口径并结合现有行业编码字母）
# B: 采矿业, D: 电力热力燃气及水生产和供应业, G: 交通运输仓储和邮政业
# I: 信息传输软件和信息技术服务业, N: 水利环境和公共设施管理业
REGULATED_INDUSTRY_SECTORS = {"B", "D", "G", "I", "N"}

EASTERN_REGION_PREFIXES = {
    "北京市",
    "天津市",
    "河北省",
    "辽宁省",
    "上海市",
    "江苏省",
    "浙江省",
    "福建省",
    "山东省",
    "广东省",
    "海南省",
}

NON_EASTERN_REGION_PREFIXES = {
    "山西省",
    "吉林省",
    "黑龙江省",
    "安徽省",
    "江西省",
    "河南省",
    "湖北省",
    "湖南省",
    "广西壮族自治区",
    "重庆市",
    "四川省",
    "贵州省",
    "云南省",
    "西藏自治区",
    "陕西省",
    "甘肃省",
    "青海省",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
    "内蒙古自治区",
}

SPECIAL_REGION_PREFIXES = sorted(
    EASTERN_REGION_PREFIXES | NON_EASTERN_REGION_PREFIXES,
    key=len,
    reverse=True,
)

CITY_FALLBACK_PREFIXES = {
    "北京": 0,
    "天津": 0,
    "上海": 0,
    "重庆": 1,
}

MUNICIPALITY_CITY_TO_PROVINCE = {
    "北京市": "北京市",
    "天津市": "天津市",
    "上海市": "上海市",
    "重庆市": "重庆市",
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
}

# 实控人性质代码口径（用于央企/地方国企分组）
# 2100: 中央国有企业
# 2120: 地方国有企业
# 注：其余国有相关代码（如 1100）含义在样本中存在混合，不强行归类以避免误分。
CENTRAL_CONTROLLER_CODES = {"2100"}
LOCAL_CONTROLLER_CODES = {"2120"}
PRIVATE_OWNERSHIP_VALUE = "私营企业"
BOOTSTRAP_RANDOM_SEED = 20260322
MAIN_MEDIATION_BOOTSTRAP_REPS = 300
GROUPED_MEDIATION_BOOTSTRAP_REPS = 300
GROUPED_BOOTSTRAP_TRIGGER_P = 0.10
FE_CONTROL_VARS = ["Roa", "Lever", "Top1"]
BASE_SUBSIDY_COL = "lnSubsidy"
BASE_SUBSIDY_LAG_COL = "lnSubsidy_l1"
BASE_SUBSIDY_LEAD1_COL = "lnSubsidy_f1"
BASE_SUBSIDY_LEAD2_COL = "lnSubsidy_f2"
ALT_SUBSIDY_COL = "lnSubsidy_pos"
ALT_SUBSIDY_LAG_COL = "lnSubsidy_pos_l1"
EVENT_STUDY_THRESHOLD_QUANTILE = 0.75
EVENT_STUDY_WINDOW = 3
SPECIAL_TREATMENT_PREFIX_PATTERN = re.compile(
    r"^(?:[A-Z]+)?(?:S\*ST|\*ST|SST|PT|ST)",
    re.IGNORECASE,
)
SPECIAL_TREATMENT_SUFFIX_PATTERN = re.compile(r"(?:退市整理|退市|退)$", re.IGNORECASE)


def _assert_not_lfs_pointer(file_path):
    """若文件仍是 Git LFS 指针，给出明确报错，避免后续读数失败。"""
    with open(file_path, "rb") as f:
        first_line = f.readline().decode("utf-8", errors="ignore").strip()
    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"检测到 Git LFS 指针文件: {file_path}\n"
            "请先执行 `git lfs install && git lfs pull` 拉取真实数据后再运行。"
        )


def _controller_first_code(value):
    """提取 ActualControllerNatureID 的首个代码。"""
    if pd.isna(value):
        return np.nan
    code = str(value).strip().split(",")[0].strip()
    if code in {"", "nan", "None"}:
        return np.nan
    if code.endswith(".0"):
        code = code[:-2]
    return code


def _extract_region_prefix(address):
    """从注册地址/办公地址中提取省级行政区前缀。"""
    if pd.isna(address):
        return None
    text = str(address).strip()
    if not text or text in {"None", "nan", "NaN"}:
        return None
    for prefix in SPECIAL_REGION_PREFIXES:
        if prefix in text:
            return prefix
    return None


def _classify_zone_from_location(city, address_register=None, address_office=None):
    """按注册地址优先、办公地址次之、城市名兜底的规则划分东部/中西部。"""
    for address in (address_register, address_office):
        region_prefix = _extract_region_prefix(address)
        if region_prefix in EASTERN_REGION_PREFIXES:
            return 0
        if region_prefix in NON_EASTERN_REGION_PREFIXES:
            return 1

    if pd.isna(city):
        return np.nan
    city_text = str(city).strip()
    if not city_text or city_text in {"None", "nan", "NaN"}:
        return np.nan
    for prefix, zone in CITY_FALLBACK_PREFIXES.items():
        if city_text.startswith(prefix):
            return zone
    return np.nan


def _extract_province_from_location(city, address_register=None, address_office=None):
    """提取公司所在省级行政区，用于构造省级工具变量。"""
    for address in (address_register, address_office):
        region_prefix = _extract_region_prefix(address)
        if region_prefix is not None:
            return region_prefix

    if pd.isna(city):
        return np.nan
    city_text = str(city).strip()
    if not city_text or city_text in {"None", "nan", "NaN"}:
        return np.nan
    return MUNICIPALITY_CITY_TO_PROVINCE.get(city_text, np.nan)


def _build_leave_one_out_mean(df, group_cols, value_col, output_col):
    """按组构造留一法均值工具变量。"""
    group_sum = df.groupby(group_cols)[value_col].transform(lambda x: x.sum(skipna=True))
    group_count = df.groupby(group_cols)[value_col].transform(lambda x: x.count())
    df[output_col] = np.where(
        group_count > 1,
        (group_sum - df[value_col].fillna(0)) / (group_count - 1),
        np.nan,
    )
    return df


def _build_leave_one_out_positive_share(df, group_cols, value_col, output_col):
    """按组构造留一法正补助占比，用于 Heckman 选择方程排除变量。"""
    positive_flag = (df[value_col].fillna(0) > 0).astype(float)
    tmp = df.assign(_positive_flag=positive_flag)
    group_count = tmp.groupby(group_cols)["_positive_flag"].transform("count")
    positive_sum = tmp.groupby(group_cols)["_positive_flag"].transform("sum")
    df[output_col] = np.where(
        group_count > 1,
        (positive_sum - positive_flag) / (group_count - 1),
        np.nan,
    )
    return df


def _build_industry_year_excluding_province_mean(df, output_col, value_col=BASE_SUBSIDY_COL):
    """构造同行业同年、排除本省后的平均补贴工具变量。"""
    values = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby(["IndustrySector", "Year"], sort=False):
        province_sum = group.groupby("Province")[value_col].sum(min_count=1)
        province_count = group.groupby("Province")[value_col].count()
        total_sum = group[value_col].sum(skipna=True)
        total_count = group[value_col].count()
        for idx, row in group.iterrows():
            province = row.get("Province")
            if pd.isna(province):
                continue
            adjusted_sum = total_sum - province_sum.get(province, 0)
            adjusted_count = total_count - province_count.get(province, 0)
            if adjusted_count > 0:
                values.at[idx] = adjusted_sum / adjusted_count
    df[output_col] = values
    return df


def _build_province_industry_excluding_city_mean(df, output_col, value_col=BASE_SUBSIDY_COL):
    """构造同省同行业同年、排除本市后的平均补贴工具变量。"""
    values = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby(["Province", "IndustrySector", "Year"], sort=False):
        city_sum = group.groupby("City")[value_col].sum(min_count=1)
        city_count = group.groupby("City")[value_col].count()
        total_sum = group[value_col].sum(skipna=True)
        total_count = group[value_col].count()
        for idx, row in group.iterrows():
            city = row.get("City")
            adjusted_sum = total_sum - city_sum.get(city, 0)
            adjusted_count = total_count - city_count.get(city, 0)
            if adjusted_count > 0:
                values.at[idx] = adjusted_sum / adjusted_count
    df[output_col] = values
    return df


def _format_pvalue(pvalue):
    """格式化 p 值展示。"""
    if pd.isna(pvalue):
        return "nan"
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.4f}"


def _is_special_treatment_name(short_name):
    """按公司简称标签识别 ST/*ST/S*ST/SST/PT/退市整理期样本。"""
    if pd.isna(short_name):
        return False
    normalized = re.sub(r"\s+", "", str(short_name).strip())
    if not normalized:
        return False
    return bool(
        SPECIAL_TREATMENT_PREFIX_PATTERN.search(normalized)
        or SPECIAL_TREATMENT_SUFFIX_PATTERN.search(normalized)
    )


def _sample_stage_row(stage, df_subset, note=""):
    """统一记录样本筛选各阶段的观测数与公司数。"""
    return {
        "stage": stage,
        "observations": int(len(df_subset)),
        "unique_companies": int(df_subset["Symbol"].nunique()) if "Symbol" in df_subset.columns else np.nan,
        "note": note,
    }


def _classify_pvalue_signal(pvalue):
    """按统一阈值给出显著性信号标签。"""
    if pd.isna(pvalue):
        return "无法判定"
    if pvalue < 0.05:
        return "显著"
    if pvalue < 0.10:
        return "边际显著"
    return "不显著"


def _stable_seed_from_text(label, base_seed=BOOTSTRAP_RANDOM_SEED):
    """为每个分组生成稳定的 bootstrap 随机种子。"""
    return int(base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(label))))


def _ci_excludes_zero(lower, upper):
    """判断置信区间是否排除 0。"""
    if pd.isna(lower) or pd.isna(upper):
        return False
    return lower > 0 or upper < 0


def _effect_direction_consistent(total_effect, indirect_effect):
    """判断间接效应方向是否与总效应一致。"""
    if pd.isna(total_effect) or pd.isna(indirect_effect):
        return False
    if np.isclose(total_effect, 0.0) or np.isclose(indirect_effect, 0.0):
        return False
    return bool(total_effect * indirect_effect > 0)


def _cluster_bootstrap_indirect_effect(x4, y_overpay, x5, y_power, groups, reps, seed, mediator_col="Power", subsidy_col="lnSubsidy_l1"):
    """公司层面 cluster bootstrap 估计间接效应 a×b 的分布（基于合并OLS作为近似或传入已去均值数据）。"""
    if reps <= 0:
        return {
            "bootstrap_reps": 0,
            "bootstrap_mean_indirect": np.nan,
            "bootstrap_p": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_conclusion": "未执行",
        }

    x4_array = np.asarray(x4, dtype=float)
    x5_array = np.asarray(x5, dtype=float)
    y_over_array = np.asarray(y_overpay, dtype=float)
    y_power_array = np.asarray(y_power, dtype=float)
    group_array = pd.Series(groups).astype(str).to_numpy()

    power_idx = x4.columns.get_loc(mediator_col)
    subsidy_idx = x5.columns.get_loc(subsidy_col)

    unique_groups = pd.unique(group_array)
    cluster_rows = [np.flatnonzero(group_array == group) for group in unique_groups]
    n_clusters = len(cluster_rows)
    if n_clusters < 20:
        return {
            "bootstrap_reps": 0,
            "bootstrap_mean_indirect": np.nan,
            "bootstrap_p": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_conclusion": "聚类数过少，未执行",
        }

    rng = np.random.default_rng(seed)
    indirect_effects = np.empty(reps)
    indirect_effects.fill(np.nan)

    for rep in range(reps):
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        sampled_rows = np.concatenate([cluster_rows[idx] for idx in sampled_clusters])
        beta4 = np.linalg.lstsq(x4_array[sampled_rows], y_over_array[sampled_rows], rcond=None)[0]
        beta5 = np.linalg.lstsq(x5_array[sampled_rows], y_power_array[sampled_rows], rcond=None)[0]
        indirect_effects[rep] = beta5[subsidy_idx] * beta4[power_idx]

    valid_effects = indirect_effects[np.isfinite(indirect_effects)]
    if len(valid_effects) == 0:
        return {
            "bootstrap_reps": reps,
            "bootstrap_mean_indirect": np.nan,
            "bootstrap_p": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_conclusion": "估计失败",
        }

    ci_lower, ci_upper = np.quantile(valid_effects, [0.025, 0.975])
    p_left = np.mean(valid_effects <= 0)
    p_right = np.mean(valid_effects >= 0)
    bootstrap_p = min(1.0, 2 * min(p_left, p_right))
    bootstrap_conclusion = "95%CI不含0" if (ci_lower > 0 or ci_upper < 0) else "95%CI跨0"

    return {
        "bootstrap_reps": int(reps),
        "bootstrap_mean_indirect": float(valid_effects.mean()),
        "bootstrap_p": float(bootstrap_p),
        "bootstrap_ci_lower": float(ci_lower),
        "bootstrap_ci_upper": float(ci_upper),
        "bootstrap_conclusion": bootstrap_conclusion,
    }


def _classify_mediation_type(summary):
    """按学术严谨性区分常规中介、间接传导线索和不支持。"""
    path_a_sig = bool(summary.get("path_a_significant", False))
    path_b_sig = bool(summary.get("path_b_significant", False))
    direction_consistent = bool(summary.get("indirect_direction_consistent", False))
    total_effect_sig = bool(pd.notna(summary.get("p_c", np.nan)) and summary.get("p_c", np.nan) < 0.05)
    bootstrap_supported = _ci_excludes_zero(
        summary.get("bootstrap_ci_lower", np.nan),
        summary.get("bootstrap_ci_upper", np.nan),
    )
    if path_a_sig and path_b_sig and bootstrap_supported and direction_consistent and total_effect_sig:
        return "常规中介效应"
    if bootstrap_supported and path_a_sig and path_b_sig:
        return "间接传导线索"
    if bootstrap_supported and not (path_a_sig and path_b_sig):
        return "间接传导线索"
    if bootstrap_supported and not direction_consistent:
        return "间接传导线索"
    return "不支持中介效应"


def _classify_group_evidence(summary):
    """结合 FDR 给出分组结果的严谨口径。"""
    mediation_type = summary.get("mediation_type", "未评估")
    fdr_p = summary.get("fdr_p", np.nan)

    if mediation_type == "常规中介效应":
        if pd.isna(fdr_p):
            return "常规中介效应"
        if fdr_p < 0.05:
            return "通过FDR的常规中介证据"
        if fdr_p < 0.10:
            return "边际通过FDR的常规中介证据"
        return "未通过FDR校正，谨慎解释"
    if mediation_type == "间接传导线索":
        return "间接传导线索"
    return "不支持中介效应"


def _apply_layerwise_fdr(summary_df):
    """按层级分别进行 BH-FDR 校正。"""
    result_df = summary_df.copy()
    result_df["fdr_p"] = np.nan
    result_df["fdr_signal"] = "不适用"

    for layer, idx in result_df.groupby("layer").groups.items():
        idx = list(idx)
        pvals = result_df.loc[idx, "sobel_p"].to_numpy(dtype=float)
        valid_mask = np.isfinite(pvals)
        adjusted = np.full(len(pvals), np.nan)
        if valid_mask.any():
            adjusted_valid = multipletests(pvals[valid_mask], method="fdr_bh")[1]
            adjusted[valid_mask] = adjusted_valid
        result_df.loc[idx, "fdr_p"] = adjusted
        result_df.loc[idx, "fdr_signal"] = [
            _classify_pvalue_signal(value) if np.isfinite(value) else "无法判定"
            for value in adjusted
        ]

    return result_df

# ============================================================
# 第一部分：数据加载与清洗
# ============================================================

def load_and_clean_data(data_dir):
    """从 processed_data 文件夹加载所有数据并合并为面板数据"""
    print("=" * 70)
    print("第一部分：数据加载与清洗")
    print("=" * 70)

    # --- 1. 财务数据 (营业收入, 净利润, 总资产, 无形资产, 行业) ---
    print("\n[1/8] 加载财务数据...")
    fin_path = os.path.join(data_dir, "营业收入+净利润+总资产+无形资产+行业变量.csv")
    _assert_not_lfs_pointer(fin_path)
    df_fin = pd.read_csv(fin_path)
    df_fin.rename(columns={"id": "Symbol", "OperatingEvenue": "Revenue"}, inplace=True)
    df_fin["Symbol"] = df_fin["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_fin["Year"] = df_fin["year"]
    df_fin = df_fin[["Symbol", "Year", "IndustryCode1", "IndustryName1",
                      "TotalAssets", "IntangibleAsset", "NetProfit", "Revenue"]]
    print(f"  {len(df_fin)} 行")

    # --- 2. 两职合一 + 管理层持股 + 董事会规模 + 独董比例 ---
    print("[2/8] 加载公司治理数据...")
    gov_path = os.path.join(data_dir, "两职合一+管理层持股比例+董事会规模+独立董事占比.csv")
    _assert_not_lfs_pointer(gov_path)
    df_gov = pd.read_csv(gov_path)
    df_gov["Symbol"] = df_gov["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_gov["Year"] = pd.to_datetime(df_gov["Enddate"]).dt.year
    # 去重，保留每个公司每年最后一条记录
    df_gov = df_gov.sort_values("Enddate").groupby(["Symbol", "Year"]).last().reset_index()
    df_gov = df_gov.rename(columns={"ShortName": "GovShortName"})
    df_gov = df_gov[["Symbol", "Year", "GovShortName", "ConcurrentPosition", "Mngmhldn",
                      "Boardsize", "IndDirectorRatio"]]
    print(f"  {len(df_gov)} 行")

    # --- 3. 总负债 ---
    print("[3/8] 加载负债数据...")
    debt_path = os.path.join(data_dir, "总负债.csv")
    _assert_not_lfs_pointer(debt_path)
    df_debt = pd.read_csv(debt_path)
    df_debt["Symbol"] = df_debt["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_debt["Year"] = pd.to_datetime(df_debt["EndDate"]).dt.year
    df_debt = df_debt.groupby(["Symbol", "Year"])["TotalLiability"].last().reset_index()
    print(f"  {len(df_debt)} 行")

    # --- 4. 高管前三名薪酬总合 ---
    print("[4/8] 加载薪酬数据...")
    pay_path = os.path.join(data_dir, "高管前三名薪酬总合.csv")
    _assert_not_lfs_pointer(pay_path)
    df_pay = pd.read_csv(pay_path)
    df_pay["Symbol"] = df_pay["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_pay["Year"] = pd.to_datetime(df_pay["Enddate"]).dt.year
    # StatisticalCaliber 1和2 的值相同，取其一即可
    df_pay = df_pay[df_pay["StatisticalCaliber"] == 1]
    df_pay = df_pay.dropna(subset=["Top3ManageSumSalary"])
    df_pay = df_pay.groupby(["Symbol", "Year"])["Top3ManageSumSalary"].max().reset_index()
    df_pay.rename(columns={"Top3ManageSumSalary": "Top3Salary"}, inplace=True)
    print(f"  {len(df_pay)} 行")

    # --- 5. 所在地 + 公司性质 ---
    print("[5/8] 加载企业性质与所在地数据...")
    loc_path = os.path.join(data_dir, "所在地+公司性质.csv")
    _assert_not_lfs_pointer(loc_path)
    df_loc = pd.read_csv(loc_path)
    df_loc["Symbol"] = df_loc["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_loc["Year"] = pd.to_datetime(df_loc["EndDate"]).dt.year
    df_loc = df_loc.sort_values("EndDate").groupby(["Symbol", "Year"]).last().reset_index()
    df_loc = df_loc[["Symbol", "Year", "Ownership", "City", "ADDRESS_REGISTER", "ADDRESS_OFFICE"]]
    print(f"  {len(df_loc)} 行")

    # --- 6. 第一大股东持股比例 + 实际控制人股权性质 ---
    print("[6/8] 加载股东数据...")
    holder_path = os.path.join(data_dir, "第一大股东持股比例+实际控制人股权性质.csv")
    _assert_not_lfs_pointer(holder_path)
    df_holder = pd.read_csv(holder_path)
    df_holder["Symbol"] = df_holder["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_holder["Year"] = pd.to_datetime(df_holder["EndDate"]).dt.year
    df_holder = df_holder.sort_values("EndDate").groupby(["Symbol", "Year"]).last().reset_index()
    if "ShortName" in df_holder.columns:
        df_holder = df_holder.rename(columns={"ShortName": "HolderShortName"})
        df_holder = df_holder[["Symbol", "Year", "HolderShortName", "LargestHolderRate", "ActualControllerNatureID"]]
    else:
        df_holder = df_holder[["Symbol", "Year", "LargestHolderRate", "ActualControllerNatureID"]]
    print(f"  {len(df_holder)} 行")

    # --- 7. 总经理任期年限 ---
    print("[7/8] 加载总经理任期数据...")
    tenure_path = os.path.join(data_dir, "总经理任期年限.csv")
    _assert_not_lfs_pointer(tenure_path)
    df_tenure = pd.read_csv(tenure_path)
    df_tenure.rename(columns={"Stkcd": "Symbol"}, inplace=True)
    df_tenure["Symbol"] = df_tenure["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_tenure["Year"] = pd.to_datetime(df_tenure["Reptdt"]).dt.year
    # 筛选总经理 (GTAPosition 包含 "总经理" 或 Position 包含 "总经理")
    mask_gm = df_tenure["GTAPosition"].str.contains("总经理", na=False) | \
              df_tenure["Position"].str.contains("总经理", na=False)
    df_gm = df_tenure[mask_gm].copy()
    # 取每个公司每年总经理的最大任期（月→年）
    df_gm["TenureYears"] = df_gm["Tenure"] / 12.0
    df_gm = df_gm.groupby(["Symbol", "Year"])["TenureYears"].max().reset_index()
    print(f"  {len(df_gm)} 行 (筛选总经理后)")

    # --- 8. 政府补助 ---
    print("[8/8] 加载政府补助数据...")
    sub_path = os.path.join(data_dir, "政府补助.csv")
    _assert_not_lfs_pointer(sub_path)
    df_sub = pd.read_csv(sub_path)
    df_sub.rename(columns={"Stkcd": "Symbol", "Fn05602": "SubsidyAmount"}, inplace=True)
    df_sub["Symbol"] = df_sub["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_sub["Year"] = pd.to_datetime(df_sub["Accper"]).dt.year
    # 排除"合计"行，避免重复加总
    df_sub = df_sub[df_sub["Fn05601"] != "合计"]
    # 汇总每个公司每年的所有政府补助金额
    df_sub_agg = (
        df_sub.sort_values("Accper")
        .groupby(["Symbol", "Year"], as_index=False)
        .agg({"SubsidyAmount": "sum", "ShortName": "last"})
        .rename(columns={"ShortName": "SubsidyShortName"})
    )
    print(f"  {len(df_sub_agg)} 行 (按公司-年汇总后)")

    # ============================
    # 合并所有数据
    # ============================
    print("\n合并数据...")
    # 以财务数据为基准，逐步合并
    merged = df_fin.copy()

    for name, df_right in [("治理", df_gov), ("负债", df_debt), ("薪酬", df_pay),
                              ("所在地", df_loc), ("股东", df_holder),
                              ("任期", df_gm), ("补助", df_sub_agg)]:
        before = len(merged)
        merged = pd.merge(merged, df_right, on=["Symbol", "Year"], how="left")
        print(f"  + {name}: {before} → {len(merged)} 行")

    short_name_cols = [col for col in ["GovShortName", "HolderShortName", "SubsidyShortName"] if col in merged.columns]
    if short_name_cols:
        for col in short_name_cols:
            merged[col] = merged[col].astype("string").str.strip()
        merged["ShortName"] = merged[short_name_cols].bfill(axis=1).iloc[:, 0]
        merged["IsSpecialTreatment"] = merged[short_name_cols].apply(
            lambda row: any(_is_special_treatment_name(value) for value in row),
            axis=1,
        )
    else:
        merged["ShortName"] = pd.Series(pd.NA, index=merged.index, dtype="string")
        merged["IsSpecialTreatment"] = False

    print(f"  特殊处理简称标签样本（按简称识别）: {int(merged['IsSpecialTreatment'].sum())} 行")
    print(f"\n合并后总行数: {len(merged)}")
    return merged


# ============================================================
# 第二部分：变量构造
# ============================================================

def construct_variables(df):
    """根据论文方法构造所有研究变量"""
    print("\n" + "=" * 70)
    print("第二部分：变量构造")
    print("=" * 70)

    # 1. 被解释变量：高管薪酬对数
    print("\n1. 构造 lnCEOpay = ln(高管前三名薪酬总和)")
    positive_pay = df["Top3Salary"].where(df["Top3Salary"] > 0)
    df["lnCEOpay"] = np.log(positive_pay)
    non_positive_pay = int((df["Top3Salary"] <= 0).fillna(False).sum())
    if non_positive_pay > 0:
        print(f"   高管薪酬非正值（lnCEOpay 置缺失）: {non_positive_pay}")

    # 2. 解释变量：政府补助对数
    print("2. 构造政府补助对数变量...")
    # 未披露政府补助的公司-年视为 0，保证基准样本覆盖全部可比上市公司年度。
    df["SubsidyAmount"] = df["SubsidyAmount"].fillna(0)
    negative_subsidy_n = int((df["SubsidyAmount"] < 0).fillna(False).sum())
    zero_subsidy_n = int((df["SubsidyAmount"] == 0).fillna(False).sum())
    positive_subsidy = df["SubsidyAmount"].where(df["SubsidyAmount"] > 0)
    # 基准口径：ln(1 + Subsidy)
    df[BASE_SUBSIDY_COL] = np.log1p(df["SubsidyAmount"].clip(lower=0))
    # 替代口径：仅对正补助取对数
    df[ALT_SUBSIDY_COL] = np.log(positive_subsidy)
    # 保留旧字段名以兼容既有机器学习与出图脚本
    df["lnSubsidy1p"] = df[BASE_SUBSIDY_COL]
    print(f"   零补助样本数: {zero_subsidy_n}")
    print(f"   负补助样本数（基准口径按0处理）: {negative_subsidy_n}")

    # 3. 控制变量
    print("3. 构造控制变量...")
    # Roa = 净利润 / 总资产
    df["Roa"] = df["NetProfit"] / df["TotalAssets"]
    # lnSale = ln(营业收入)，非正值视为异常置缺失
    positive_revenue = df["Revenue"].where(df["Revenue"] > 0)
    df["lnSale"] = np.log(positive_revenue)
    non_positive_rev = int((df["Revenue"] <= 0).fillna(False).sum())
    if non_positive_rev > 0:
        print(f"   营业收入非正值（lnSale 置缺失）: {non_positive_rev}")
    # IA = 无形资产 / 总资产
    df["IA"] = df["IntangibleAsset"] / df["TotalAssets"]
    # Lever = 总负债 / 总资产
    df["Lever"] = df["TotalLiability"] / df["TotalAssets"]
    # Top1 = 第一大股东持股比例 (已有)
    df["Top1"] = df["LargestHolderRate"]

    # 4. Zone 区域虚拟变量：东部=0, 中西部=1；Province 用于省级工具变量
    print("4. 构造区域变量 Zone 与 Province...")
    df["Zone"] = df.apply(
        lambda row: _classify_zone_from_location(
            row.get("City"),
            row.get("ADDRESS_REGISTER"),
            row.get("ADDRESS_OFFICE"),
        ),
        axis=1,
    )
    df["Province"] = df.apply(
        lambda row: _extract_province_from_location(
            row.get("City"),
            row.get("ADDRESS_REGISTER"),
            row.get("ADDRESS_OFFICE"),
        ),
        axis=1,
    )
    zone_missing = int(df["Zone"].isna().sum())
    if zone_missing > 0:
        print(f"   Zone 无法识别的样本数: {zone_missing}")
    province_missing = int(df["Province"].isna().sum())
    if province_missing > 0:
        print(f"   Province 无法识别的样本数: {province_missing}")

    # 5. Industry 行业分类（用于生成行业虚拟变量）
    print("5. 构造行业分类变量 IndustrySector...")
    # 取 IndustryCode1 的首字母作为行业大类 (如 C=制造业, D=电力, E=建筑, ...)
    df["IndustrySector"] = df["IndustryCode1"].str[0]
    # 同时保留制造业二分变量，用于简单描述
    df["Industry"] = (df["IndustrySector"] == "C").astype(int)
    df["RegulatedIndustry"] = df["IndustrySector"].isin(REGULATED_INDUSTRY_SECTORS).astype(int)
    print(f"   行业大类分布:\n{df['IndustrySector'].value_counts().to_string()}")
    print(f"   管制行业样本占比: {df['RegulatedIndustry'].mean():.2%}")

    # 6. 管理层权力指标
    print("6. 构造管理层权力各分指标...")
    # Dual (两职合一): ConcurrentPosition 在 CSMAR 中可能编码为 1/2 或 0/1
    # 统一转换为 0/1 二值变量（1=兼任, 0=不兼任）
    raw_dual = df["ConcurrentPosition"]
    if raw_dual.dropna().isin([1, 2]).all():
        # CSMAR 编码: 1=兼任, 2=不兼任 → 转为 1/0
        df["Dual"] = (raw_dual == 1).astype(float)
        df.loc[raw_dual.isna(), "Dual"] = np.nan
        print("   Dual: 原始编码为1/2，已转换为1=兼任/0=不兼任")
    else:
        # 已经是 0/1 编码
        df["Dual"] = raw_dual
        print("   Dual: 原始编码为0/1，直接使用")
    print(f"   Dual 分布: {df['Dual'].value_counts(dropna=False).to_dict()}")
    # Insider (内部董事比例) = 1 - 独立董事比例/100
    df["Insider"] = 1 - df["IndDirectorRatio"] / 100.0
    # Mgshder (管理层持股比例)
    df["Mgshder"] = df["Mngmhldn"]
    # Tenure (总经理任职年限)
    df["Tenure"] = df["TenureYears"]
    # Boardsize (董事会规模) - 直接使用

    # 7. 公司性质分类（基于 Ownership 字段的精确值分类）
    print("7. 构造企业性质分类...")
    # Ownership 字段的实际值：'国营或国有控股', '私营企业', '中外合资', '集体企业', '事业单位', '外商独资', '其他'
    soe_values = ["国营或国有控股"]  # 国有企业
    non_soe_values = ["私营企业", "中外合资", "外商独资"]  # 非国有企业

    def classify_ownership(own):
        if pd.isna(own):
            return np.nan
        own_str = str(own).strip()
        if own_str in soe_values:
            return "国有"
        elif own_str in non_soe_values:
            return "非国有"
        else:
            return np.nan  # 集体企业、事业单位等分类不明确，标记为缺失

    df["OwnerType"] = df["Ownership"].apply(classify_ownership)
    df["IsSOE"] = np.where(df["OwnerType"] == "国有", 1,
                           np.where(df["OwnerType"] == "非国有", 0, np.nan))
    df["IsPrivate"] = np.where(df["Ownership"] == PRIVATE_OWNERSHIP_VALUE, 1, 0)
    print(f"   企业性质分布:\n{df['OwnerType'].value_counts(dropna=False).to_string()}")
    print(f"   私营企业样本量: {int(df['IsPrivate'].sum())}")

    # 8. 央企/地方国企识别（仅在国有样本内）
    print("8. 构造央企/地方国企分组...")
    df["ControllerCode"] = df["ActualControllerNatureID"].apply(_controller_first_code)
    df["IsCentralSOE"] = np.where(
        (df["IsSOE"] == 1) & (df["ControllerCode"].isin(CENTRAL_CONTROLLER_CODES)),
        1,
        np.where(
            (df["IsSOE"] == 1) & (df["ControllerCode"].isin(LOCAL_CONTROLLER_CODES)),
            0,
            np.nan,
        ),
    )
    df["SOELevel"] = df["IsCentralSOE"].map({1.0: "央企", 0.0: "地方国企"})
    print(f"   央地分组（仅国有内识别）:\n{df['SOELevel'].value_counts(dropna=False).to_string()}")

    print("9. 构造滞后变量与工具变量...")
    # 构建滞后一期补贴变量
    df = df.sort_values(["Symbol", "Year"])
    df["SubsidyAmount_l1"] = df.groupby("Symbol")["SubsidyAmount"].shift(1)
    df[BASE_SUBSIDY_LAG_COL] = df.groupby("Symbol")[BASE_SUBSIDY_COL].shift(1)
    df[ALT_SUBSIDY_LAG_COL] = df.groupby("Symbol")[ALT_SUBSIDY_COL].shift(1)
    df[BASE_SUBSIDY_LEAD1_COL] = df.groupby("Symbol")[BASE_SUBSIDY_COL].shift(-1)
    df[BASE_SUBSIDY_LEAD2_COL] = df.groupby("Symbol")[BASE_SUBSIDY_COL].shift(-2)

    # 单工具与双工具候选：同城、同行业、同城同行业留一法平均补贴
    df = _build_leave_one_out_mean(df, ["City", "Year"], BASE_SUBSIDY_COL, "IV_city_year")
    df = _build_leave_one_out_mean(df, ["IndustrySector", "Year"], BASE_SUBSIDY_COL, "IV_industry_year")
    df = _build_leave_one_out_mean(df, ["City", "IndustrySector", "Year"], BASE_SUBSIDY_COL, "IV_city_industry_year")
    df = _build_leave_one_out_mean(df, ["Province", "Year"], BASE_SUBSIDY_COL, "IV_province_year")
    df = _build_industry_year_excluding_province_mean(df, "IV_industry_excl_province")
    df = _build_province_industry_excluding_city_mean(df, "IV_province_industry_excl_city")

    for iv_col in [
        "IV_city_year",
        "IV_industry_year",
        "IV_city_industry_year",
        "IV_province_year",
        "IV_industry_excl_province",
        "IV_province_industry_excl_city",
    ]:
        df[f"{iv_col}_l1"] = df.groupby("Symbol")[iv_col].shift(1)

    # 保留旧字段名兼容既有导出与校验逻辑
    df["IV_lnSubsidy"] = df["IV_city_year"]
    df["IV_lnSubsidy_l1"] = df["IV_city_year_l1"]

    # Heckman 选择方程用到的正补助占比（留一法）
    df = _build_leave_one_out_positive_share(df, ["City", "Year"], "SubsidyAmount", "IV_city_posshare")
    df = _build_leave_one_out_positive_share(df, ["Province", "Year"], "SubsidyAmount", "IV_province_posshare")
    df = _build_leave_one_out_positive_share(df, ["IndustrySector", "Year"], "SubsidyAmount", "IV_industry_posshare")
    for share_col in ["IV_city_posshare", "IV_province_posshare", "IV_industry_posshare"]:
        df[f"{share_col}_l1"] = df.groupby("Symbol")[share_col].shift(1)

    print(f"\n变量构造完成。当前列: {list(df.columns)}")
    return df


# ============================================================
# 第三部分：样本筛选与描述性统计
# ============================================================

def filter_and_describe(df):
    """样本筛选与描述性统计分析"""
    print("\n" + "=" * 70)
    print("第三部分：样本筛选与描述性统计")
    print("=" * 70)

    # 核心变量列表
    core_vars = ["lnCEOpay", BASE_SUBSIDY_COL, ALT_SUBSIDY_COL, "lnSale", "Roa", "IA", "Lever",
                 "Top1", "Zone", "Industry", "Dual", "Insider", "Mgshder",
                 "Tenure", "Boardsize"]

    print(f"\n筛选前总样本: {len(df)} 行")
    sample_summary = [_sample_stage_row("原始公司-年观测", df, note="合并各类原始表后的面板样本")]

    # 剔除金融行业 (IndustryCode1 以 J 开头)
    df = df[~df["IndustryCode1"].str.startswith("J", na=False)].copy()
    print(f"剔除金融行业后: {len(df)} 行")
    sample_summary.append(_sample_stage_row("剔除金融行业后", df, note="按行业代码首字母 J 识别金融行业"))

    # 剔除 ST/*ST/S*ST/SST/PT/退市整理期样本（按公司简称标签识别）
    special_treatment_mask = df["IsSpecialTreatment"].fillna(False).astype(bool)
    removed_special_rows = int(special_treatment_mask.sum())
    removed_special_companies = int(df.loc[special_treatment_mask, "Symbol"].nunique())
    df = df.loc[~special_treatment_mask].copy()
    print(
        "剔除 ST/*ST/PT/退市整理期样本后: "
        f"{len(df)} 行（剔除 {removed_special_rows} 行、{removed_special_companies} 家公司）"
    )
    sample_summary.append(
        _sample_stage_row(
            "剔除特殊处理样本后",
            df,
            note="按公司简称标签剔除 ST/*ST/S*ST/SST/PT/退市整理期样本",
        )
    )

    # 要求核心变量非空
    key_vars = ["lnCEOpay", BASE_SUBSIDY_COL, "Roa", "Lever", "Top1", "Zone", "Industry"]
    df_clean = df.dropna(subset=key_vars).copy()
    print(f"剔除关键变量缺失后: {len(df_clean)} 行")
    sample_summary.append(
        _sample_stage_row(
            "关键变量完整样本",
            df_clean,
            note="要求 lnCEOpay、lnSubsidy、Roa、Lever、Top1、Zone、Industry 同时完整",
        )
    )
    short_name_missing = int(df_clean["ShortName"].isna().sum()) if "ShortName" in df_clean.columns else 0
    if short_name_missing > 0:
        print(f"  提示：关键变量完整样本中仍有 {short_name_missing} 行简称缺失，无法进一步按简称校验。")

    # 原始薪酬数据已在源数据阶段缩尾，此处不再对 lnCEOpay / Overpay 二次缩尾
    print("\n对除薪酬外的连续变量进行 1%-99% Winsorize 缩尾处理...")
    continuous_vars = [BASE_SUBSIDY_COL, ALT_SUBSIDY_COL, "lnSale", "Roa", "IA", "Lever", "Top1"]
    for var in continuous_vars:
        if var in df_clean.columns:
            lower = df_clean[var].quantile(0.01)
            upper = df_clean[var].quantile(0.99)
            df_clean[var] = df_clean[var].clip(lower, upper)

    # 描述性统计
    print("\n" + "-" * 50)
    print("表1 主要变量描述性统计")
    print("-" * 50)
    desc = _compute_descriptive_statistics(df_clean)
    print(
        desc.rename(
            columns={
                "变量": "变量",
                "N": "N",
                "均值": "Mean",
                "标准差": "Std",
                "最小值": "Min",
                "中位数": "Median",
                "最大值": "Max",
            }
        ).to_string(index=False)
    )

    print(f"\n样本企业性质分布:")
    print(df_clean["OwnerType"].value_counts())

    print(f"\n年度分布:")
    print(df_clean["Year"].value_counts().sort_index())

    # ---- 相关性矩阵与 VIF 检验 ----
    print("\n" + "-" * 50)
    print("相关性矩阵（皮尔逊相关系数）")
    print("-" * 50)
    corr_matrix = _compute_correlation_matrix(df_clean)
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 50)
    print("方差膨胀因子（VIF）检验")
    print("-" * 50)
    try:
        vif_df = _compute_vif_table(df_clean)
        print(vif_df.to_string(index=False))
        print(f"最大VIF = {vif_df['VIF'].max():.4f}，均值 = {vif_df['VIF'].mean():.4f}")
    except Exception as e:
        print(f"VIF 计算出错: {e}")

    df_clean.attrs["sample_screening_summary"] = sample_summary
    return df_clean


# ============================================================
# 辅助函数：生成行业和年份虚拟变量
# ============================================================

def get_industry_dummies(df, col="IndustrySector", drop_first=True):
    """生成行业虚拟变量，drop_first=True 避免完全多重共线性"""
    dummies = pd.get_dummies(df[col], prefix="Ind", drop_first=drop_first, dtype=float)
    return dummies

def get_year_dummies(df, col="Year", drop_first=True):
    """生成年份虚拟变量"""
    dummies = pd.get_dummies(df[col], prefix="Year", drop_first=drop_first, dtype=float)
    return dummies


def _fit_expectation_salary_model(df):
    """按正文口径拟合模型1：OLS + 行业虚拟变量 + 年份虚拟变量。"""
    model1_vars = ["lnSale", "Roa", "IA", "Zone"]
    df_model1 = df.dropna(subset=model1_vars + ["lnCEOpay", "IndustrySector", "Year"]).copy()
    X = df_model1[model1_vars].copy()
    ind_dummies = get_industry_dummies(df_model1)
    year_dummies = get_year_dummies(df_model1)
    X = pd.concat([X, ind_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_model1["lnCEOpay"]
    model1 = sm.OLS(y, X).fit()
    return {
        "model": model1,
        "df_model1": df_model1,
        "model1_vars": model1_vars,
        "industry_dummies": ind_dummies,
        "year_dummies": year_dummies,
    }


def _compute_bp_test(stage1_model):
    """基于模型1残差执行 Breusch-Pagan 异方差检验。"""
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
        stage1_model.resid,
        stage1_model.model.exog,
    )
    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def _compute_wooldridge_serial_test(
    df,
    dep_var="Overpay",
    core_vars=None,
    entity_col="Symbol",
    time_col="Year",
):
    """按 Wooldridge(2002)/Drukker(2003) 思路近似执行面板序列相关检验。"""
    if core_vars is None:
        core_vars = [BASE_SUBSIDY_LAG_COL] + FE_CONTROL_VARS

    needed = [dep_var, entity_col, time_col] + list(core_vars)
    panel = (
        df.dropna(subset=needed)
        .copy()
        .sort_values([entity_col, time_col])
    )
    if panel.empty:
        return {
            "rho": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "pvalue": np.nan,
            "sample_size": 0,
            "n_clusters": 0,
        }

    panel["year_gap"] = panel.groupby(entity_col)[time_col].diff()
    diff_cols = []
    for col in [dep_var] + list(core_vars):
        prev = panel.groupby(entity_col)[col].shift(1)
        diff_col = f"d_{col}"
        panel[diff_col] = panel[col] - prev
        panel.loc[panel["year_gap"] != 1, diff_col] = np.nan
        diff_cols.append(diff_col)

    fd_panel = panel.dropna(subset=diff_cols).copy()
    if len(fd_panel) < 100:
        return {
            "rho": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "pvalue": np.nan,
            "sample_size": int(len(fd_panel)),
            "n_clusters": int(fd_panel[entity_col].nunique()) if not fd_panel.empty else 0,
        }

    fd_y = fd_panel[f"d_{dep_var}"]
    fd_x = fd_panel[[f"d_{col}" for col in core_vars]]
    fd_model = sm.OLS(fd_y, fd_x).fit()

    fd_panel["resid_fd"] = fd_model.resid
    fd_panel["lag_resid_fd"] = fd_panel.groupby(entity_col)["resid_fd"].shift(1)
    fd_panel["lag_year_gap"] = fd_panel.groupby(entity_col)[time_col].diff()
    fd_panel.loc[fd_panel["lag_year_gap"] != 1, "lag_resid_fd"] = np.nan

    aux = fd_panel.dropna(subset=["resid_fd", "lag_resid_fd"]).copy()
    if len(aux) < 100:
        return {
            "rho": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "pvalue": np.nan,
            "sample_size": int(len(aux)),
            "n_clusters": int(aux[entity_col].nunique()) if not aux.empty else 0,
        }

    aux_model = sm.OLS(
        aux["resid_fd"],
        aux[["lag_resid_fd"]],
    ).fit(cov_type="cluster", cov_kwds={"groups": aux[entity_col]})

    rho = float(aux_model.params["lag_resid_fd"])
    se = float(aux_model.bse["lag_resid_fd"])
    t_stat = float((rho + 0.5) / se)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    return {
        "rho": rho,
        "se": se,
        "t_stat": t_stat,
        "pvalue": pvalue,
        "sample_size": int(len(aux)),
        "n_clusters": int(aux[entity_col].nunique()),
    }


def _compute_descriptive_statistics(df):
    """按正文展示口径计算描述性统计。"""
    desc_vars_map = [
        ("Top3Salary", "高管前三名薪酬总额"),
        ("SubsidyAmount", "政府补助"),
        ("lnSubsidy", "财政补贴强度(lnSubsidy=ln(1+Subsidy))"),
        ("lnCEOpay", "高管前三名薪酬对数"),
        ("lnSale", "企业规模(lnSale)"),
        ("IA", "无形资产占比(IA)"),
        ("Overpay", "超额薪酬(Overpay)"),
        ("Power", "管理层权力(Power, FA口径)"),
        ("Roa", "业绩(Roa)"),
        ("Lever", "财务杠杆"),
        ("Top1", "第一大股东持股比例"),
        ("Zone", "地区(Zone, 中西部=1)"),
    ]
    rows = []
    for var, label in desc_vars_map:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        rows.append({
            "变量": label,
            "N": int(s.count()),
            "均值": float(s.mean()),
            "中位数": float(s.median()),
            "标准差": float(s.std()),
            "最小值": float(s.min()),
            "最大值": float(s.max()),
        })
    return pd.DataFrame(rows)


def _compute_correlation_matrix(df):
    """按正文表4-2口径计算相关系数矩阵。"""
    corr_vars = [v for v in [BASE_SUBSIDY_COL, "lnSale", "Roa", "IA", "Lever", "Top1", "Zone"] if v in df.columns]
    return df[corr_vars].dropna().corr()


def _compute_vif_table(df):
    """按正文表4-3口径计算 VIF。"""
    check_vars = [v for v in [BASE_SUBSIDY_COL, "lnSale", "Roa", "IA", "Lever", "Top1", "Zone"] if v in df.columns]
    df_vif = df[check_vars].dropna().copy()
    df_vif = sm.add_constant(df_vif)
    vif_data = []
    for i, col in enumerate(df_vif.columns):
        if col == "const":
            continue
        vif_data.append({"变量": col, "VIF": float(variance_inflation_factor(df_vif.values, i))})
    return pd.DataFrame(vif_data)


# ============================================================
# 第四部分：期望薪酬模型 → 超额薪酬 Overpay
# ============================================================

def compute_overpay(df):
    """
    模型1: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ΣYear + ε
    模型2: Overpay = ε（即模型1残差）
    """
    print("\n" + "=" * 70)
    print("第四部分：期望薪酬模型与超额薪酬 (Overpay)")
    print("=" * 70)

    fit_result = _fit_expectation_salary_model(df)
    model1 = fit_result["model"]
    df_model1 = fit_result["df_model1"]
    model1_vars = fit_result["model1_vars"]
    ind_dummies = fit_result["industry_dummies"]
    year_dummies = fit_result["year_dummies"]

    print(f"\n模型1 样本量: {len(df_model1)}")
    print(f"核心自变量: {model1_vars}")
    print(f"行业虚拟变量: {len(ind_dummies.columns)} 个")
    print(f"年份虚拟变量: {len(year_dummies.columns)} 个")

    print("\n" + "-" * 50)
    print("模型1 回归结果 (期望薪酬模型) — 仅展示核心变量")
    print("-" * 50)
    # 只打印核心变量系数，行业/年份dummy太多不全部展示
    core_result = model1.summary2().tables[1]
    core_rows = ["const"] + model1_vars
    core_result_filtered = core_result.loc[core_result.index.isin(core_rows)]
    print(core_result_filtered.to_string())
    print(f"行业固定效应: 已控制 ({len(ind_dummies.columns)} 个行业虚拟变量)")
    print(f"年份固定效应: 已控制 ({len(year_dummies.columns)} 个年份虚拟变量)")
    print(f"\nR² = {model1.rsquared:.4f}, Adj-R² = {model1.rsquared_adj:.4f}")

    # 残差 = 超额薪酬
    df_model1["Overpay"] = model1.resid
    print("\nOverpay 直接使用模型1残差，不再进行二次缩尾。")

    # 将 Overpay 合并回主数据
    df = df.merge(df_model1[["Symbol", "Year", "Overpay"]],
                  on=["Symbol", "Year"], how="left")

    print(f"\n超额薪酬 (Overpay) 描述:")
    print(df["Overpay"].describe().to_string())

    return df


# ============================================================
# 第五部分：管理层权力 Power（FA 经验性综合口径，PCA 为对照）
# ============================================================

def _calc_shared_factor_diagnostics(df_complete, power_vars):
    """计算 KMO、Bartlett 球形检验及相关矩阵诊断。"""
    standardized = df_complete[power_vars].astype(float)
    corr = np.corrcoef(standardized, rowvar=False)
    inv_corr = np.linalg.pinv(corr)

    scale = np.diag(1 / np.sqrt(np.clip(np.diag(inv_corr), 1e-12, None)))
    partial_corr = -scale @ inv_corr @ scale
    np.fill_diagonal(partial_corr, 0)

    corr_offdiag = corr.copy()
    np.fill_diagonal(corr_offdiag, 0)
    corr_sq_sum = np.sum(corr_offdiag ** 2)
    partial_sq_sum = np.sum(partial_corr ** 2)
    kmo_overall = corr_sq_sum / (corr_sq_sum + partial_sq_sum)

    kmo_per_var = {}
    for idx, var in enumerate(power_vars):
        r2 = np.sum(np.delete(corr_offdiag[:, idx] ** 2, idx))
        p2 = np.sum(np.delete(partial_corr[:, idx] ** 2, idx))
        kmo_per_var[var] = r2 / (r2 + p2)

    n_obs = len(df_complete)
    n_var = len(power_vars)
    det_corr = max(float(np.linalg.det(corr)), np.finfo(float).tiny)
    chi_square = -(n_obs - 1 - (2 * n_var + 5) / 6) * np.log(det_corr)
    dof = n_var * (n_var - 1) / 2
    bartlett_p = 1 - stats.chi2.cdf(chi_square, dof)

    eigvals = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
    explained = eigvals / eigvals.sum()

    return {
        "n_obs": int(n_obs),
        "kmo_overall": float(kmo_overall),
        "bartlett_chi2": float(chi_square),
        "bartlett_df": int(dof),
        "bartlett_p": float(bartlett_p),
        "explained_variance_ratio_pc1": float(explained[0]),
        "cum_explained_variance_pc2": float(explained[:2].sum()),
        "cum_explained_variance_pc3": float(explained[:3].sum()),
        "kmo_per_var": {var: float(value) for var, value in kmo_per_var.items()},
    }


def _orient_power_scores(scores, standardized_df):
    """统一得分方向：数值越大表示管理层权力越强。"""
    reference = standardized_df.mean(axis=1).to_numpy(dtype=float)
    if np.std(scores) < 1e-12 or np.std(reference) < 1e-12:
        return scores, 1.0
    corr = np.corrcoef(scores, reference)[0, 1]
    sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
    return scores * sign, sign


def _standardize_power_inputs(df_complete, power_vars):
    """标准化管理层权力分指标。"""
    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(df_complete[power_vars]),
        columns=power_vars,
        index=df_complete.index,
    )
    return standardized


def _compute_pca_power(standardized_df, power_vars):
    """PCA 对照口径：提取前两个主成分并按方差贡献率加权形成综合得分。"""
    n_components = min(2, len(power_vars))
    pca = PCA(n_components=n_components)
    raw_scores = pca.fit_transform(standardized_df)

    oriented_scores = raw_scores.copy()
    component_signs = []
    loadings_matrix = pca.components_.copy()
    for idx in range(n_components):
        score_vec, sign = _orient_power_scores(raw_scores[:, idx], standardized_df)
        oriented_scores[:, idx] = score_vec
        loadings_matrix[idx, :] = loadings_matrix[idx, :] * sign
        component_signs.append(sign)

    component_weights = pca.explained_variance_ratio_ / np.sum(pca.explained_variance_ratio_)
    composite_raw = oriented_scores @ component_weights
    composite_std = np.std(composite_raw, ddof=0)
    if composite_std < 1e-12:
        composite_scores = np.zeros_like(composite_raw)
    else:
        composite_scores = (composite_raw - composite_raw.mean()) / composite_std
    composite_loadings = loadings_matrix.T @ component_weights

    loadings = {var: float(val) for var, val in zip(power_vars, composite_loadings)}
    pc1_loadings = {var: float(val) for var, val in zip(power_vars, loadings_matrix[0, :])}
    pc2_loadings = {var: float(val) for var, val in zip(power_vars, loadings_matrix[1, :])} if n_components >= 2 else {var: np.nan for var in power_vars}
    return {
        "scores": pd.Series(composite_scores, index=standardized_df.index),
        "loadings": loadings,
        "pc1_loadings": pc1_loadings,
        "pc2_loadings": pc2_loadings,
        "component_weights": {f"PC{i + 1}": float(weight) for i, weight in enumerate(component_weights)},
        "n_components": int(n_components),
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_pc2": float(pca.explained_variance_ratio_[1]) if n_components >= 2 else np.nan,
        "cum_explained_variance_pc2": float(np.sum(pca.explained_variance_ratio_[:2])),
        "cum_explained_variance_pc3": float(np.sum(pca.explained_variance_ratio_[:3])),
    }


def _compute_fa_power(standardized_df, power_vars):
    """因子分析主口径：提取前两个公因子，旋转后按方差贡献率合成为综合得分。"""
    n_factors = min(2, len(power_vars))
    fa_result = Factor(standardized_df.to_numpy(), n_factor=n_factors, method="pa").fit()
    if n_factors >= 2:
        fa_result.rotate("varimax")
    loadings_matrix = np.asarray(fa_result.loadings, dtype=float)
    score_weight_matrix = np.asarray(fa_result.factor_score_params(), dtype=float)
    raw_scores = standardized_df.to_numpy() @ score_weight_matrix

    oriented_scores = raw_scores.copy()
    oriented_loadings = loadings_matrix.copy()
    oriented_score_weights = score_weight_matrix.copy()
    for idx in range(n_factors):
        score_vec, sign = _orient_power_scores(raw_scores[:, idx], standardized_df)
        oriented_scores[:, idx] = score_vec
        oriented_loadings[:, idx] = oriented_loadings[:, idx] * sign
        oriented_score_weights[:, idx] = oriented_score_weights[:, idx] * sign

    ss_loadings = np.sum(oriented_loadings ** 2, axis=0)
    if float(np.sum(ss_loadings)) < 1e-12:
        component_weights = np.ones(n_factors, dtype=float) / n_factors
    else:
        component_weights = ss_loadings / np.sum(ss_loadings)

    composite_raw = oriented_scores @ component_weights
    composite_std = np.std(composite_raw, ddof=0)
    if composite_std < 1e-12:
        composite_scores = np.zeros_like(composite_raw)
    else:
        composite_scores = (composite_raw - composite_raw.mean()) / composite_std

    composite_loadings = oriented_loadings @ component_weights
    composite_score_weights = oriented_score_weights @ component_weights
    variance_ratio = float(np.sum(ss_loadings) / len(power_vars))
    fa1_loadings = {var: float(val) for var, val in zip(power_vars, oriented_loadings[:, 0])}
    fa2_loadings = (
        {var: float(val) for var, val in zip(power_vars, oriented_loadings[:, 1])}
        if n_factors >= 2 else
        {var: np.nan for var in power_vars}
    )
    return {
        "scores": pd.Series(composite_scores, index=standardized_df.index),
        "loadings": {var: float(val) for var, val in zip(power_vars, composite_loadings)},
        "fa1_loadings": fa1_loadings,
        "fa2_loadings": fa2_loadings,
        "score_weights": {var: float(val) for var, val in zip(power_vars, composite_score_weights)},
        "communalities": {var: float(val) for var, val in zip(power_vars, fa_result.communality)},
        "uniqueness": {var: float(val) for var, val in zip(power_vars, fa_result.uniqueness)},
        "component_weights": {f"FA{i + 1}": float(weight) for i, weight in enumerate(component_weights)},
        "ss_loading": float(np.sum(ss_loadings)),
        "ss_loadings": {f"FA{i + 1}": float(val) for i, val in enumerate(ss_loadings)},
        "n_factors": int(n_factors),
        "variance_explained_ratio": float(variance_ratio),
    }


def compute_power(df, return_diagnostics=False):
    """
    管理层权力构造：
      1. FA（因子分析）作为正文采用的经验性综合口径
      2. PCA 作为对照口径：提取前两个主成分并形成综合得分
    """
    print("\n" + "=" * 70)
    print("第五部分：管理层权力综合指标（FA经验性综合口径，PCA为对照）")
    print("=" * 70)

    # Tenure 在旧口径下共同度极低，因此从 FA/PCA 综合指标中剔除。
    power_vars = ["Dual", "Boardsize", "Insider", "Mgshder"]
    for var in power_vars:
        missing = df[var].isna().sum()
        total = len(df)
        print(f"  {var}: 缺失 {missing} ({missing / total * 100:.1f}%)")

    df_power = df.dropna(subset=power_vars).copy()
    print(f"\n可用于 Power 构造的样本量: {len(df_power)}")

    diagnostics = {
        "power_vars": power_vars,
        "main_method": "FA",
        "main_power_col": "Power_FA",
        "pca": {},
        "fa": {},
    }

    if len(df_power) < 100:
        print("  WARNING: 样本量过少，跳过 Power 构造。")
        for col in ["Power", "Power_FA", "Power_PCA"]:
            df[col] = np.nan
        df.attrs["power_diagnostics"] = diagnostics
        return (df, diagnostics) if return_diagnostics else df

    standardized = _standardize_power_inputs(df_power, power_vars)
    shared_diag = _calc_shared_factor_diagnostics(standardized, power_vars)
    pca_result = _compute_pca_power(standardized, power_vars)
    fa_result = _compute_fa_power(standardized, power_vars)

    diagnostics["pca"] = {**shared_diag, **pca_result}
    diagnostics["fa"] = {
        **shared_diag,
        **fa_result,
        "average_communality": float(np.mean(list(fa_result["communalities"].values()))),
    }

    print(f"\n  共同适用性检验: KMO = {shared_diag['kmo_overall']:.4f}")
    for var in power_vars:
        print(f"    KMO[{var:10s}] = {shared_diag['kmo_per_var'][var]:.4f}")
    print(
        f"  Bartlett 球形检验: Chi² = {shared_diag['bartlett_chi2']:.4f}, "
        f"df = {shared_diag['bartlett_df']}, p = {_format_pvalue(shared_diag['bartlett_p'])}"
    )
    print(
        f"\n  PCA 前两主成分方差贡献率 = "
        f"{pca_result['explained_variance_ratio_pc1']:.4f} + {pca_result['explained_variance_ratio_pc2']:.4f}"
    )
    print(f"  PCA 前两主成分累计方差贡献率 = {pca_result['cum_explained_variance_pc2']:.4f}")
    print("  解释：PCA 对照口径使用前两个主成分加权形成综合得分，以避免单一主成分代表性不足。")
    print(f"\n  FA 前两因子累计方差解释率 = {fa_result['variance_explained_ratio']:.4f}")
    print(f"  FA 平均共同度 = {diagnostics['fa']['average_communality']:.4f}")
    print("  解释：FA 以前两个经 varimax 旋转的公因子按载荷平方和占比合成为经验性综合口径，仍需结合较低 KMO 审慎解读。")

    score_df = pd.DataFrame({
        "Symbol": df_power["Symbol"].to_numpy(),
        "Year": df_power["Year"].to_numpy(),
        "Power_PCA": pca_result["scores"].to_numpy(),
        "Power_FA": fa_result["scores"].to_numpy(),
    })
    score_df["Power"] = score_df["Power_FA"]

    df = df.merge(score_df, on=["Symbol", "Year"], how="left")
    df.attrs["power_diagnostics"] = diagnostics

    print(f"\nPower（FA口径）描述:")
    print(df["Power"].describe().to_string())

    return (df, diagnostics) if return_diagnostics else df


# ============================================================
# 第六部分：回归分析
# ============================================================

def _within_demean_twoway(df, entity_col, time_col, variables):
    """双向组内去均值（entity + time），通过交替投影迭代收敛。

    用于 bootstrap 前的数据预处理：使 bootstrap 每轮 OLS 等价于双向 FE 估计量，
    从而保证 bootstrap 分布以分析法点估计为中心，CI 具有正确的统计含义。
    """
    vals = df[variables].copy().astype(float)
    for _ in range(300):
        prev = vals.values.copy()
        # 减去 entity（公司）均值
        vals = vals - vals.groupby(df[entity_col]).transform("mean")
        # 减去 time（年份）均值
        vals = vals - vals.groupby(df[time_col]).transform("mean")
        if np.max(np.abs(vals.values - prev)) < 1e-12:
            break
    return vals


def _build_fe_matrix(df_subset, core_vars):
    """构建包含核心变量的 X 矩阵，固定效应由 PanelOLS 原生处理"""
    X = df_subset[core_vars].copy()
    X = sm.add_constant(X)
    return X, 0, 0

def _fit_with_cluster_se(y, X, groups=None):
    """PanelOLS 回归 + 公司固定效应 + 年份固定效应 + 公司层面聚类稳健标准误"""
    if isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            X.to_numpy(dtype=float),
            index=X.index,
            columns=X.columns,
        )
        X.attrs = {}
    if isinstance(y, pd.Series):
        y = pd.Series(
            y.to_numpy(dtype=float),
            index=y.index,
            name=y.name,
        )
        y.attrs = {}
    model = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    return model.fit(cov_type='clustered', cluster_entity=True)

def _get_model_fstat(model):
    """优先返回稳健 F 统计量。"""
    robust_stat = getattr(model, "f_statistic_robust", None)
    if robust_stat is not None and getattr(robust_stat, "stat", None) is not None:
        return float(robust_stat.stat), float(robust_stat.pval)
    plain_stat = getattr(model, "f_statistic", None)
    if plain_stat is not None and getattr(plain_stat, "stat", None) is not None:
        return float(plain_stat.stat), float(plain_stat.pval)
    return np.nan, np.nan


def _print_core_results(model, core_vars, n_ind, n_year, sample_size, n_clusters=None, include_fstat=False):
    """只打印核心变量的回归结果，固定效应只显示“已控制”"""
    if hasattr(model, "summary2"):
        result_table = model.summary2().tables[1]
    else:
        # For PanelEffectsResults from linearmodels
        result_table = pd.DataFrame({
            "Coef.": model.params,
            "Std.Err.": model.std_errors,
            "t": model.tstats,
            "P>|t|": model.pvalues,
        })
    core_rows = ["const"] + core_vars
    filtered = result_table.loc[result_table.index.isin(core_rows)]
    print(f"样本量: {sample_size}")
    print(filtered.to_string())
    print("公司固定效应: 已控制")
    print("年份固定效应: 已控制")
    if n_clusters:
        print(f"聚类标准误: 公司层面 ({n_clusters} 个聚类)")
    
    rsquared = getattr(model, "rsquared", np.nan)
    if pd.isna(rsquared) and hasattr(model, "rsquared_within"):
        rsquared = model.rsquared_within
    print(f"R² = {rsquared:.4f}")
    if include_fstat:
        f_stat, f_pval = _get_model_fstat(model)
        print(f"F 统计量 = {f_stat:.2f} (p={_format_pvalue(f_pval)})")


def _fit_model2(df_sub, control_vars, subsidy_col=BASE_SUBSIDY_LAG_COL):
    """在给定样本上拟合模型2主回归。"""
    core_vars = [subsidy_col] + control_vars
    needed = core_vars + ["Overpay", "IndustrySector", "Year"]
    sub = df_sub.dropna(subset=needed).copy()
    if len(sub) < 80:
        return None

    groups = sub["Symbol"]
    sub = sub.set_index(["Symbol", "Year"], drop=False)
    x, n_ind, n_year = _build_fe_matrix(sub, core_vars)
    model = _fit_with_cluster_se(sub["Overpay"], x)
    return {
        "model": model,
        "sample_size": int(len(sub)),
        "n_clusters": int(groups.nunique()),
        "n_ind": n_ind,
        "n_year": n_year,
        "df": sub,
    }


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_test_result(test_obj):
    """统一提取 linearmodels 统计检验对象结果。"""
    if test_obj is None:
        return None
    stat = getattr(test_obj, "stat", None)
    pval = getattr(test_obj, "pval", None)
    if stat is None or pval is None:
        return None
    return {
        "stat": float(stat),
        "pval": float(pval),
        "df": getattr(test_obj, "df", None),
        "name": str(getattr(test_obj, "null", "")),
    }


def _fit_iv_with_fe(
    df,
    control_vars,
    dep_var="Overpay",
    endog_col=BASE_SUBSIDY_LAG_COL,
    instrument_col="IV_lnSubsidy_l1",
):
    """以公司内去均值 + 年份虚拟变量的方式实现 FE-2SLS。"""
    instrument_cols = _as_list(instrument_col)
    needed = [dep_var, endog_col, "Symbol", "Year"] + control_vars + instrument_cols
    df_iv = df.dropna(subset=needed).copy()
    if len(df_iv) < 100:
        return None

    year_dummies = pd.get_dummies(df_iv["Year"], prefix="Year", drop_first=True, dtype=float)
    base = pd.concat(
        [
            df_iv[["Symbol", "Year", dep_var, endog_col] + control_vars + instrument_cols].reset_index(drop=True),
            year_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    value_cols = [col for col in base.columns if col not in {"Symbol", "Year"}]
    base[value_cols] = base[value_cols].astype(float)
    entity_means = base.groupby("Symbol")[value_cols].transform("mean")
    transformed = base.copy()
    transformed[value_cols] = base[value_cols] - entity_means

    exog_cols = control_vars + list(year_dummies.columns)
    clusters = df_iv["Symbol"].reset_index(drop=True)

    iv_model = IV2SLS(
        dependent=transformed[dep_var],
        exog=transformed[exog_cols],
        endog=transformed[endog_col],
        instruments=transformed[instrument_cols],
    ).fit(cov_type="clustered", clusters=clusters)

    first_stage_ols = sm.OLS(
        transformed[endog_col],
        transformed[exog_cols + instrument_cols],
    ).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    diagnostics = iv_model.first_stage.diagnostics.loc[endog_col]
    first_stage_terms = []
    for col in instrument_cols:
        first_stage_terms.append({
            "instrument": col,
            "coef": float(first_stage_ols.params.get(col, np.nan)),
            "se": float(first_stage_ols.bse.get(col, np.nan)),
            "p": float(first_stage_ols.pvalues.get(col, np.nan)),
        })
    primary_term = first_stage_terms[0] if first_stage_terms else {}
    return {
        "model": iv_model,
        "sample_size": int(len(df_iv)),
        "n_clusters": int(df_iv["Symbol"].nunique()),
        "year_fe_count": int(len(year_dummies.columns)),
        "instrument_cols": instrument_cols,
        "instrument_label": " + ".join(instrument_cols),
        "first_stage": {
            "instrument": " + ".join(instrument_cols),
            "coef": float(primary_term.get("coef", np.nan)),
            "se": float(primary_term.get("se", np.nan)),
            "p": float(primary_term.get("p", np.nan)),
            "rsquared": float(first_stage_ols.rsquared),
            "partial_r2": float(diagnostics["partial.rsquared"]),
            "f_stat": float(diagnostics["f.stat"]),
            "f_pval": float(diagnostics["f.pval"]),
        },
        "first_stage_terms": first_stage_terms,
        "overid": _extract_test_result(getattr(iv_model, "sargan", None)),
        "overid_wooldridge": _extract_test_result(getattr(iv_model, "wooldridge_overid", None)),
        "wu_hausman": _extract_test_result(iv_model.wu_hausman()),
    }


def _fit_heckman_two_step(
    df,
    outcome_controls,
    dep_var="Overpay",
    subsidy_col=ALT_SUBSIDY_LAG_COL,
    exclusion_col="IV_lnSubsidy_l1",
):
    """对仅正补助样本的替代口径实施 Heckman 两步校正。"""
    df_h = df.sort_values(["Symbol", "Year"]).copy()
    selection_lag_bases = ["Roa", "Lever", "Top1", "lnSale", "IA"]
    for col in selection_lag_bases:
        lag_col = f"{col}_l1sel"
        if lag_col not in df_h.columns:
            df_h[lag_col] = df_h.groupby("Symbol")[col].shift(1)

    df_h["SelectedPosLag"] = (~df_h[subsidy_col].isna()).astype(int)
    selection_cols = [exclusion_col, "Roa_l1sel", "Lever_l1sel", "Top1_l1sel", "lnSale_l1sel", "IA_l1sel", "Zone"]
    needed = ["SelectedPosLag", "Symbol", "Year", "IndustrySector"] + selection_cols
    selection_df = df_h.dropna(subset=needed).copy()
    if len(selection_df) < 200:
        return None

    industry_dummies = pd.get_dummies(selection_df["IndustrySector"], prefix="Ind", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(selection_df["Year"], prefix="Year", drop_first=True, dtype=float)
    x_selection = pd.concat(
        [
            selection_df[selection_cols].reset_index(drop=True),
            industry_dummies.reset_index(drop=True),
            year_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    x_selection = sm.add_constant(x_selection)
    probit_model = sm.Probit(
        selection_df["SelectedPosLag"].reset_index(drop=True),
        x_selection,
    ).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": selection_df["Symbol"].reset_index(drop=True)},
    )

    xb = x_selection @ probit_model.params
    selection_prob = np.clip(stats.norm.cdf(xb), 1e-6, 1 - 1e-6)
    selection_df = selection_df.reset_index(drop=True)
    selection_df["IMR"] = stats.norm.pdf(xb) / selection_prob

    outcome_needed = [dep_var, subsidy_col, "IMR", "Symbol", "Year"] + outcome_controls
    outcome_df = selection_df[selection_df["SelectedPosLag"] == 1].copy()
    outcome_df = outcome_df.dropna(subset=outcome_needed).copy()
    if len(outcome_df) < 200:
        return None

    outcome_df = outcome_df.set_index(["Symbol", "Year"], drop=False)
    outcome_x = outcome_df[[subsidy_col] + outcome_controls + ["IMR"]].astype(float)
    outcome_model = _fit_with_cluster_se(outcome_df[dep_var], outcome_x)
    return {
        "selection_model": probit_model,
        "outcome_model": outcome_model,
        "selection_sample_size": int(len(selection_df)),
        "outcome_sample_size": int(len(outcome_df)),
        "selection_rate": float(selection_df["SelectedPosLag"].mean()),
        "exclusion_col": exclusion_col,
        "exclusion_coef": float(probit_model.params.get(exclusion_col, np.nan)),
        "exclusion_p": float(probit_model.pvalues.get(exclusion_col, np.nan)),
        "imr_coef": float(outcome_model.params.get("IMR", np.nan)),
        "imr_p": float(outcome_model.pvalues.get("IMR", np.nan)),
        "subsidy_coef": float(outcome_model.params.get(subsidy_col, np.nan)),
        "subsidy_p": float(outcome_model.pvalues.get(subsidy_col, np.nan)),
        "subsidy_t": float(outcome_model.tstats.get(subsidy_col, np.nan)),
    }


def _run_endogeneity_checks(df, control_vars):
    """比较不同工具变量，并对正补助口径实施 Heckman 两步校正。"""
    iv_specs = [
        {
            "spec_name": "基准工具变量",
            "role": "benchmark",
            "instrument_cols": ["IV_city_year_l1"],
            "instrument_desc": "滞后一期同城同年其他企业平均补助",
        },
        {
            "spec_name": "简单替代工具变量",
            "role": "simple",
            "instrument_cols": ["IV_industry_year_l1"],
            "instrument_desc": "滞后一期同行业同年其他企业平均补助",
        },
        {
            "spec_name": "精炼双工具变量",
            "role": "refined",
            "instrument_cols": ["IV_industry_excl_province_l1", "IV_province_industry_excl_city_l1"],
            "instrument_desc": "滞后一期同行业同年排除本省平均补助 + 同省同行业同年排除本市平均补助",
        },
    ]

    iv_results = []
    comparison_rows = []
    for spec in iv_specs:
        result = _fit_iv_with_fe(df, control_vars, instrument_col=spec["instrument_cols"])
        if result is None:
            continue
        result["spec_name"] = spec["spec_name"]
        result["role"] = spec["role"]
        result["instrument_desc"] = spec["instrument_desc"]
        iv_results.append(result)
        model = result["model"]
        comparison_rows.append({
            "spec_name": spec["spec_name"],
            "role": spec["role"],
            "instrument_desc": spec["instrument_desc"],
            "sample_size": result["sample_size"],
            "n_clusters": result["n_clusters"],
            "partial_r2": float(result["first_stage"]["partial_r2"]),
            "partial_f": float(result["first_stage"]["f_stat"]),
            "second_stage_coef": float(model.params.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "second_stage_se": float(model.std_errors.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "second_stage_t": float(model.tstats.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "second_stage_p": float(model.pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "overid_p": float(result["overid"]["pval"]) if result.get("overid") else np.nan,
            "overid_wooldridge_p": float(result["overid_wooldridge"]["pval"]) if result.get("overid_wooldridge") else np.nan,
            "wu_hausman_p": float(result["wu_hausman"]["pval"]) if result.get("wu_hausman") else np.nan,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    refined_result = next((row for row in iv_results if row.get("role") == "refined"), None)
    benchmark_result = next((row for row in iv_results if row.get("role") == "benchmark"), None)
    simple_result = next((row for row in iv_results if row.get("role") == "simple"), None)
    heckman_result = _fit_heckman_two_step(df, control_vars)
    return {
        "benchmark_iv_result": benchmark_result,
        "simple_iv_result": simple_result,
        "refined_iv_result": refined_result,
        "iv_result": refined_result or simple_result or benchmark_result,
        "iv_comparison": comparison_df,
        "heckman_result": heckman_result,
    }


def _run_placebo_checks(df, control_vars):
    """使用未来补贴开展安慰剂检验。"""
    placebo_specs = [
        ("未来一期补贴安慰剂", BASE_SUBSIDY_LEAD1_COL),
        ("未来两期补贴安慰剂", BASE_SUBSIDY_LEAD2_COL),
    ]
    rows = []
    for label, subsidy_col in placebo_specs:
        fit_result = _fit_model2(df, control_vars, subsidy_col=subsidy_col)
        if fit_result is None:
            continue
        model = fit_result["model"]
        rows.append({
            "check_item": label,
            "key_var": subsidy_col,
            "coef": float(model.params.get(subsidy_col, np.nan)),
            "se": float(model.std_errors.get(subsidy_col, np.nan)),
            "t_value": float(model.tstats.get(subsidy_col, np.nan)),
            "p_value": float(model.pvalues.get(subsidy_col, np.nan)),
            "sample_size": int(fit_result["sample_size"]),
            "n_clusters": int(fit_result["n_clusters"]),
            "r_squared": float(model.rsquared),
        })
    return pd.DataFrame(rows)


def _event_term_name(k):
    return f"event_m{abs(k)}" if k < 0 else f"event_{k}"


def _wald_zero_test(model, term_names):
    """对一组系数是否同时为 0 进行 Wald 检验。"""
    available_terms = [term for term in term_names if term in model.params.index]
    if not available_terms:
        return None
    param_names = list(model.params.index)
    restriction = np.zeros((len(available_terms), len(param_names)))
    for row_idx, term in enumerate(available_terms):
        restriction[row_idx, param_names.index(term)] = 1.0
    return _extract_test_result(model.wald_test(restriction))


def _run_event_study(df, control_vars, threshold_quantile=EVENT_STUDY_THRESHOLD_QUANTILE, window=EVENT_STUDY_WINDOW):
    """围绕首次大额补贴暴露年份开展事件研究。

    事件年定义为：公司首次出现“上一年补贴金额位于正补贴样本分位阈值以上”的年份。
    该定义与主回归的滞后一期补贴口径保持一致。
    """
    positive_lagged = df.loc[df["SubsidyAmount_l1"].fillna(0) > 0, "SubsidyAmount_l1"].dropna()
    if len(positive_lagged) < 500:
        return None

    threshold = float(positive_lagged.quantile(threshold_quantile))
    event_flag = (df["SubsidyAmount_l1"] >= threshold) & (df["SubsidyAmount_l1"] > 0)
    event_year = df.loc[event_flag, ["Symbol", "Year"]].groupby("Symbol")["Year"].min()
    if event_year.empty:
        return None

    needed = ["Overpay", "Symbol", "Year"] + control_vars
    work = df.dropna(subset=needed).copy()
    work.attrs = {}
    event_year_df = event_year.rename("event_year").reset_index()
    work = work.merge(event_year_df, on="Symbol", how="left")
    work["treated_firm"] = work["event_year"].notna().astype(int)
    work["event_time"] = work["Year"] - work["event_year"]

    event_terms = []
    for k in range(-window, window + 1):
        if k == -1:
            continue
        term = _event_term_name(k)
        work[term] = ((work["treated_firm"] == 1) & (work["event_time"] == k)).astype(float)
        event_terms.append(term)

    sub = work.set_index(["Symbol", "Year"], drop=False)
    x_cols = event_terms + control_vars
    model = _fit_with_cluster_se(sub["Overpay"], sub[x_cols].astype(float))

    row_data = []
    for k in range(-window, window + 1):
        if k == -1:
            continue
        term = _event_term_name(k)
        coef = float(model.params.get(term, np.nan))
        se = float(model.std_errors.get(term, np.nan))
        row_data.append({
            "event_time": int(k),
            "term": term,
            "coef": coef,
            "se": se,
            "t_value": float(model.tstats.get(term, np.nan)),
            "p_value": float(model.pvalues.get(term, np.nan)),
            "ci_lower": float(coef - 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan,
            "ci_upper": float(coef + 1.96 * se) if np.isfinite(coef) and np.isfinite(se) else np.nan,
            "treated_obs_count": int(((work["treated_firm"] == 1) & (work["event_time"] == k)).sum()),
        })

    pre_terms = [_event_term_name(k) for k in range(-window, -1)]
    post_terms = [_event_term_name(k) for k in range(0, window + 1)]
    pretrend_test = _wald_zero_test(model, pre_terms)
    post_test = _wald_zero_test(model, post_terms)

    return {
        "model": model,
        "rows": pd.DataFrame(row_data),
        "summary": {
            "threshold_quantile": float(threshold_quantile),
            "threshold_value": threshold,
            "window": int(window),
            "treated_firms": int(event_year.shape[0]),
            "all_firms": int(work["Symbol"].nunique()),
            "sample_size": int(len(sub)),
            "n_clusters": int(work["Symbol"].nunique()),
            "pretrend_p": float(pretrend_test["pval"]) if pretrend_test else np.nan,
            "posttrend_p": float(post_test["pval"]) if post_test else np.nan,
            "pretrend_stat": float(pretrend_test["stat"]) if pretrend_test else np.nan,
            "posttrend_stat": float(post_test["stat"]) if post_test else np.nan,
            "omitted_period": -1,
        },
    }


def _summarize_mediation(label, m3, m4, m5, sample_size, n_clusters, mediator_col="Power", mediator_method="FA", subsidy_col=BASE_SUBSIDY_LAG_COL):
    """汇总 Baron & Kenny 路径系数、Sobel 与 bootstrap 所需字段。"""
    coef_c = m3.params.get(subsidy_col, np.nan)
    p_c = m3.pvalues.get(subsidy_col, np.nan)
    coef_c_prime = m4.params.get(subsidy_col, np.nan)
    p_c_prime = m4.pvalues.get(subsidy_col, np.nan)
    coef_b = m4.params.get(mediator_col, np.nan)
    p_b = m4.pvalues.get(mediator_col, np.nan)
    se_b = m4.std_errors.get(mediator_col, np.nan) if hasattr(m4, 'std_errors') else m4.bse.get(mediator_col, np.nan)
    coef_a = m5.params.get(subsidy_col, np.nan)
    p_a = m5.pvalues.get(subsidy_col, np.nan)
    se_a = m5.std_errors.get(subsidy_col, np.nan) if hasattr(m5, 'std_errors') else m5.bse.get(subsidy_col, np.nan)

    denominator = np.sqrt(coef_b ** 2 * se_a ** 2 + coef_a ** 2 * se_b ** 2)
    sobel_z = (coef_a * coef_b) / denominator if denominator and not np.isnan(denominator) else np.nan
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan
    indirect_effect = coef_a * coef_b
    mediation_ratio = indirect_effect / coef_c * 100 if (not np.isnan(coef_c) and coef_c != 0) else np.nan

    return {
        "group": label,
        "mediator_col": mediator_col,
        "mediator_method": mediator_method,
        "sample_size": int(sample_size),
        "n_clusters": int(n_clusters),
        "coef_c": float(coef_c),
        "p_c": float(p_c),
        "coef_a": float(coef_a),
        "p_a": float(p_a),
        "coef_b": float(coef_b),
        "p_b": float(p_b),
        "coef_c_prime": float(coef_c_prime),
        "p_c_prime": float(p_c_prime),
        "indirect_effect": float(indirect_effect),
        "sobel_z": float(sobel_z),
        "sobel_p": float(sobel_p),
        "mediation_ratio_pct": float(mediation_ratio),
        "sobel_signal": _classify_pvalue_signal(sobel_p),
        "indirect_direction_consistent": _effect_direction_consistent(coef_c, indirect_effect),
        "path_a_significant": bool(pd.notna(p_a) and p_a < 0.05),
        "path_b_significant": bool(pd.notna(p_b) and p_b < 0.05),
        "fdr_p": np.nan,
        "fdr_signal": "不适用",
        "bootstrap_reps": 0,
        "bootstrap_mean_indirect": np.nan,
        "bootstrap_p": np.nan,
        "bootstrap_ci_lower": np.nan,
        "bootstrap_ci_upper": np.nan,
        "bootstrap_conclusion": "未执行",
        "bootstrap_supported": False,
        "mediation_type": "未评估",
        "group_evidence_level": "未评估",
    }

def _print_mediation_summary(summary, indent="  ", subsidy_col=BASE_SUBSIDY_LAG_COL):
    """打印中介效应摘要。"""
    print(f"\n{indent}逐步回归法 (Baron & Kenny, 1986):")
    print(
        f"{indent}  第一步 (模型3) {subsidy_col}→Overpay:  "
        f"c={summary['coef_c']:.4f} (p={_format_pvalue(summary['p_c'])})"
    )
    print(
        f"{indent}  第二步 (模型4) {subsidy_col}→Power:    "
        f"a={summary['coef_a']:.4f} (p={_format_pvalue(summary['p_a'])})"
    )
    print(
        f"{indent}  第三步 (模型5) {subsidy_col}→Overpay:  "
        f"c'={summary['coef_c_prime']:.4f} (p={_format_pvalue(summary['p_c_prime'])})"
    )
    print(
        f"{indent}  第三步 (模型5) Power→Overpay:      "
        f"b={summary['coef_b']:.4f} (p={_format_pvalue(summary['p_b'])})"
    )
    print(f"\n{indent}Sobel 检验:")
    print(f"{indent}  间接效应 (a×b) = {summary['indirect_effect']:.6f}")
    print(f"{indent}  Sobel Z = {summary['sobel_z']:.4f}")
    print(f"{indent}  Sobel p = {_format_pvalue(summary['sobel_p'])} ({summary['sobel_signal']})")
    print(f"{indent}  中介效应占总效应比例 = {summary['mediation_ratio_pct']:.2f}%")
    print(f"{indent}  路径a显著: {'是' if summary['path_a_significant'] else '否'}")
    print(f"{indent}  路径b显著: {'是' if summary['path_b_significant'] else '否'}")
    print(f"{indent}  间接效应方向与总效应一致: {'是' if summary['indirect_direction_consistent'] else '否'}")
    if summary.get("bootstrap_reps", 0):
        print(f"\n{indent}Cluster Bootstrap ({summary['bootstrap_reps']} 次):")
        print(
            f"{indent}  间接效应均值 = {summary['bootstrap_mean_indirect']:.6f}, "
            f"经验 p = {_format_pvalue(summary['bootstrap_p'])}"
        )
        print(
            f"{indent}  95% CI = "
            f"[{summary['bootstrap_ci_lower']:.6f}, {summary['bootstrap_ci_upper']:.6f}]"
        )
        print(f"{indent}  Bootstrap 判定 = {summary['bootstrap_conclusion']}")
    if pd.notna(summary.get("fdr_p", np.nan)):
        print(f"{indent}  FDR 校正后 p = {_format_pvalue(summary['fdr_p'])}")
        print(f"{indent}  FDR 信号 = {summary['fdr_signal']}")
    print(f"\n{indent}最终中介类型：{summary.get('mediation_type', '未评估')}")
    print(f"{indent}综合判定：{summary.get('group_evidence_level', '未评估')}")


def _augment_summary_with_bootstrap(fit_result, reps, seed, subsidy_col=BASE_SUBSIDY_LAG_COL):
    """为中介检验结果补充 cluster bootstrap 统计量。"""
    bootstrap_result = _cluster_bootstrap_indirect_effect(
        fit_result["x4"],
        fit_result["y4"],
        fit_result["x5"],
        fit_result["y5"],
        fit_result["groups"],
        reps=reps,
        seed=seed,
        mediator_col=fit_result["summary"]["mediator_col"],
        subsidy_col=subsidy_col
    )
    fit_result["summary"].update(bootstrap_result)
    fit_result["summary"]["bootstrap_supported"] = bool(
        _ci_excludes_zero(
            fit_result["summary"]["bootstrap_ci_lower"],
            fit_result["summary"]["bootstrap_ci_upper"],
        )
    )
    fit_result["summary"]["mediation_type"] = _classify_mediation_type(fit_result["summary"])
    fit_result["summary"]["group_evidence_level"] = _classify_group_evidence(fit_result["summary"])
    return fit_result


def _run_mediation_models(df_sub, label, control_vars, dep_var="Overpay", mediator_col="Power", mediator_method="FA", subsidy_col=BASE_SUBSIDY_LAG_COL):
    """在给定样本上统一拟合模型3/4/5并汇总中介效应。"""
    core3 = [subsidy_col] + control_vars
    core4 = [subsidy_col, mediator_col] + control_vars
    core5 = [subsidy_col] + control_vars

    needed = core4 + [dep_var, "IndustrySector", "Year"]
    unified = df_sub.dropna(subset=needed).copy()
    if len(unified) < 80:
        return None

    groups = unified["Symbol"]
    # panelols needs entity and time multiindex
    unified = unified.set_index(["Symbol", "Year"], drop=False)
    
    x3, n_ind3, n_year3 = _build_fe_matrix(unified, core3)
    x4, n_ind4, n_year4 = _build_fe_matrix(unified, core4)
    x5, n_ind5, n_year5 = _build_fe_matrix(unified, core5)

    m3 = _fit_with_cluster_se(unified[dep_var], x3)
    m4 = _fit_with_cluster_se(unified[dep_var], x4)
    m5 = _fit_with_cluster_se(unified[mediator_col], x5)

    n_unified = len(unified)
    n_clusters = groups.nunique()
    summary = _summarize_mediation(
        label,
        m3,
        m4,
        m5,
        n_unified,
        n_clusters,
        mediator_col=mediator_col,
        mediator_method=mediator_method,
        subsidy_col=subsidy_col,
    )

    # 双向组内去均值，使 bootstrap OLS 与 FE 估计量等价
    # unified 的 Symbol/Year 既是索引也是普通列（drop=False），直接用普通列去均值
    unified_flat = unified.reset_index(drop=True)
    all_boot_vars = list(dict.fromkeys(core4 + [dep_var, mediator_col]))
    demeaned = _within_demean_twoway(unified_flat, "Symbol", "Year", all_boot_vars)
    x4_boot = sm.add_constant(demeaned[core4])
    x5_boot = sm.add_constant(demeaned[core5])

    return {
        "label": label,
        "unified_df": unified.copy(),
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "x4": x4_boot.reset_index(drop=True),   # 去均值后传给 bootstrap
        "x5": x5_boot.reset_index(drop=True),
        "y4": demeaned[dep_var].reset_index(drop=True),
        "y5": demeaned[mediator_col].reset_index(drop=True),
        "groups": groups.reset_index(drop=True),
        "n3": n_unified,
        "n4": n_unified,
        "n5": n_unified,
        "ind3": n_ind3,
        "year3": n_year3,
        "ind4": n_ind4,
        "year4": n_year4,
        "ind5": n_ind5,
        "year5": n_year5,
        "cl3": n_clusters,
        "cl4": n_clusters,
        "cl5": n_clusters,
        "summary": summary,
    }


def _print_subsample_model2(label, fit_result, subsidy_col=BASE_SUBSIDY_LAG_COL):
    """打印子样本的模型2核心结果。"""
    model = fit_result["model"]
    print(f"\n{'=' * 40}")
    print(f"子样本: {label}")
    print(f"{'=' * 40}")
    print(
        f"模型2 N={fit_result['sample_size']} (聚类={fit_result['n_clusters']})"
    )
    print(
        f"模型2 {subsidy_col}={model.params.get(subsidy_col, np.nan):.4f} "
        f"(p={model.pvalues.get(subsidy_col, np.nan):.4f}), "
        f"R²={model.rsquared:.4f}, F={_get_model_fstat(model)[0]:.2f}"
    )


def _build_power_method_comparison(df, control_vars, power_diagnostics=None):
    """比较 FA 与 PCA 两种 Power 构造在全样本中的中介结果。"""
    power_diagnostics = power_diagnostics or {}
    power_vars = power_diagnostics.get("power_vars", ["Dual", "Boardsize", "Insider", "Mgshder"])
    rows = []
    method_specs = [
        {"method": "FA", "role": "正文口径（经验性综合指标）", "power_col": "Power_FA", "diag_key": "fa"},
        {"method": "PCA", "role": "对照口径（前两主成分加权综合得分）", "power_col": "Power_PCA", "diag_key": "pca"},
    ]

    for spec in method_specs:
        if spec["power_col"] not in df.columns:
            continue
        df_method = df.copy()
        df_method["Power"] = df_method[spec["power_col"]]
        fit_result = _run_mediation_models(
            df_method,
            label=spec["method"],
            control_vars=control_vars,
            mediator_col="Power",
            mediator_method=spec["method"],
        )
        if fit_result is None:
            continue
        _augment_summary_with_bootstrap(
            fit_result,
            reps=MAIN_MEDIATION_BOOTSTRAP_REPS,
            seed=(
                _stable_seed_from_text("全样本")
                if spec["method"] == "FA"
                else _stable_seed_from_text(f"PowerMethod::{spec['method']}")
            ),
        )
        summary = fit_result["summary"]
        diag = power_diagnostics.get(spec["diag_key"], {})
        diag_source = fit_result["unified_df"].dropna(subset=power_vars).copy()
        if not diag_source.empty:
            standardized = _standardize_power_inputs(diag_source, power_vars)
            shared_diag = _calc_shared_factor_diagnostics(standardized, power_vars)
            if spec["method"] == "FA":
                diag = {**shared_diag, **_compute_fa_power(standardized, power_vars)}
            else:
                diag = {**shared_diag, **_compute_pca_power(standardized, power_vars)}
        rows.append({
            "method": spec["method"],
            "role": spec["role"],
            "power_col": spec["power_col"],
            "sample_size": summary["sample_size"],
            "n_clusters": summary["n_clusters"],
            "kmo": diag.get("kmo_overall", np.nan),
            "bartlett_p": diag.get("bartlett_p", np.nan),
            "primary_variance_ratio": (
                diag.get("variance_explained_ratio", np.nan)
                if spec["method"] == "FA"
                else diag.get("cum_explained_variance_pc2", np.nan)
            ),
            "coef_a": summary["coef_a"],
            "p_a": summary["p_a"],
            "coef_b": summary["coef_b"],
            "p_b": summary["p_b"],
            "coef_c_prime": summary["coef_c_prime"],
            "p_c_prime": summary["p_c_prime"],
            "coef_c": summary["coef_c"],
            "p_c": summary["p_c"],
            "sobel_p": summary["sobel_p"],
            "bootstrap_p": summary["bootstrap_p"],
            "bootstrap_ci_lower": summary["bootstrap_ci_lower"],
            "bootstrap_ci_upper": summary["bootstrap_ci_upper"],
            "bootstrap_supported": summary["bootstrap_supported"],
            "indirect_direction_consistent": summary["indirect_direction_consistent"],
            "path_a_significant": summary["path_a_significant"],
            "path_b_significant": summary["path_b_significant"],
            "mediation_type": summary["mediation_type"],
            "group_evidence_level": summary["group_evidence_level"],
        })

    return pd.DataFrame(rows)


def run_regressions(df, power_diagnostics=None):
    """
    模型2: Overpay = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    模型3: Overpay = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    模型4: Power   = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    模型5: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + 控制 + ΣIndustry + ΣYear + ε

    注意：模型2使用主回归自身完整样本；模型3-5使用中介统一样本。
    """
    print("\n" + "=" * 70)
    print("第六部分：回归分析（公司固定效应 + 年份固定效应）")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    power_diagnostics = power_diagnostics or df.attrs.get("power_diagnostics", {})

    main_result = _fit_model2(df, control_vars)
    if main_result is None:
        raise RuntimeError("全样本无法完成模型2主回归，请检查缺失值处理。")

    fit_result = _run_mediation_models(df, "全样本", control_vars, mediator_col="Power", mediator_method="FA")
    if fit_result is None:
        raise RuntimeError("全样本无法完成模型3/4/5回归，请检查缺失值处理。")
    _augment_summary_with_bootstrap(
        fit_result,
        reps=MAIN_MEDIATION_BOOTSTRAP_REPS,
        seed=_stable_seed_from_text("全样本"),
    )
    fit_result["summary"]["power_measure_method"] = "FA"
    df_unified = df.dropna(subset=[BASE_SUBSIDY_LAG_COL, "Power"] + control_vars + ["Overpay", "IndustrySector", "Year"]).copy()
    print(f"\n模型2样本量: N = {main_result['sample_size']}")
    print(f"  聚类数（公司数）: {main_result['n_clusters']}")

    # ---- 模型2：主回归 ----
    print("\n" + "-" * 50)
    print(f"模型2: Overpay = f({BASE_SUBSIDY_LAG_COL}, Controls, Firm FE, Year FE)")
    print("-" * 50)
    model2 = main_result["model"]
    core2 = [BASE_SUBSIDY_LAG_COL] + control_vars
    _print_core_results(
        model2,
        core2,
        main_result["n_ind"],
        main_result["n_year"],
        main_result["sample_size"],
        main_result["n_clusters"],
        include_fstat=True,
    )

    # ---- 内生性与选择偏差检验 ----
    print("\n" + "-" * 50)
    print("工具变量与选择偏差检验")
    print("-" * 50)
    endogeneity_checks = _run_endogeneity_checks(df, control_vars)
    iv_comparison = endogeneity_checks["iv_comparison"]
    if iv_comparison.empty:
        raise RuntimeError("IV 样本量不足，无法完成替代工具变量检验。")
    print("IV 方案比较：")
    print(iv_comparison.to_string(index=False))
    heckman_result = endogeneity_checks.get("heckman_result")
    if heckman_result is not None:
        print(
            "Heckman 两步法（正补助样本）: "
            f"选择样本量={heckman_result['selection_sample_size']}, "
            f"结果样本量={heckman_result['outcome_sample_size']}, "
            f"排除变量={heckman_result['exclusion_col']} (p={_format_pvalue(heckman_result['exclusion_p'])}), "
            f"IMR p={_format_pvalue(heckman_result['imr_p'])}, "
            f"补贴系数={heckman_result['subsidy_coef']:.4f} (p={_format_pvalue(heckman_result['subsidy_p'])})"
        )

    print("\n" + "-" * 50)
    print("未来补贴安慰剂检验")
    print("-" * 50)
    placebo_df = _run_placebo_checks(df, control_vars)
    if placebo_df.empty:
        print("安慰剂样本量不足，未能完成估计。")
    else:
        print(placebo_df.to_string(index=False))

    print("\n" + "-" * 50)
    print("首次大额补贴事件研究")
    print("-" * 50)
    event_study_result = _run_event_study(df, control_vars)
    if event_study_result is None:
        print("事件研究样本量不足，未能完成估计。")
    else:
        event_summary = event_study_result["summary"]
        print(
            "事件定义: 首次出现上一年补贴金额进入正补贴样本前"
            f"{int(EVENT_STUDY_THRESHOLD_QUANTILE * 100)}%分位以上；"
            f"阈值={event_summary['threshold_value']:.2f}"
        )
        print(
            f"事件窗口: [-{event_summary['window']}, {event_summary['window']}], "
            f"treated firms={event_summary['treated_firms']}, "
            f"N={event_summary['sample_size']}, 聚类={event_summary['n_clusters']}"
        )
        print(
            f"预趋势联合检验 p={_format_pvalue(event_summary['pretrend_p'])}; "
            f"事后系数联合检验 p={_format_pvalue(event_summary['posttrend_p'])}"
        )
        print(event_study_result["rows"].to_string(index=False))

    print(f"\n模型3-5统一样本量: N = {len(df_unified)}")
    print(f"  聚类数（公司数）: {df_unified['Symbol'].nunique()}")

    # ---- 模型3：中介总效应 ----
    print("\n" + "-" * 50)
    print(f"模型3: Overpay = f({BASE_SUBSIDY_LAG_COL}, Controls, Firm FE, Year FE)")
    print("-" * 50)
    core3 = [BASE_SUBSIDY_LAG_COL] + control_vars
    model3 = fit_result["m3"]
    _print_core_results(model3, core3, fit_result["ind3"], fit_result["year3"], len(df_unified), fit_result["cl3"])

    # ---- 模型4：补贴对管理层权力 ----
    print("\n" + "-" * 50)
    print(f"模型4: Power[FA] = f({BASE_SUBSIDY_LAG_COL}, Controls, Firm FE, Year FE)")
    print("-" * 50)
    core4 = [BASE_SUBSIDY_LAG_COL] + control_vars
    model4 = fit_result["m5"]
    _print_core_results(model4, core4, fit_result["ind5"], fit_result["year5"], len(df_unified), fit_result["cl5"])

    # ---- 模型5：加入管理层权力后的中介回归 ----
    print("\n" + "-" * 50)
    print(f"模型5: Overpay = f({BASE_SUBSIDY_LAG_COL}, Power[FA], Controls, Firm FE, Year FE)")
    print("-" * 50)
    core5 = [BASE_SUBSIDY_LAG_COL, "Power"] + control_vars
    model5 = fit_result["m4"]
    _print_core_results(model5, core5, fit_result["ind4"], fit_result["year4"], len(df_unified), fit_result["cl4"])

    # ---- 中介效应判断 (Baron & Kenny + Sobel Test) ----
    print("\n" + "=" * 50)
    print("中介效应检验")
    print("=" * 50)
    _print_mediation_summary(fit_result["summary"])

    method_comparison = _build_power_method_comparison(df, control_vars, power_diagnostics)

    return {
        "model2": model2,
        "model3": model3,
        "model4": model4,
        "model5": model5,
        "main_result": main_result,
        "iv_result": endogeneity_checks["iv_result"],
        "iv_checks": endogeneity_checks,
        "placebo_df": placebo_df,
        "event_study_result": event_study_result,
        "summary": fit_result["summary"],
        "fit_result": fit_result,
        "method_comparison": method_comparison,
    }


# ============================================================
# 第七部分：异质性分析（分产权性质）
# ============================================================

def heterogeneity_analysis(df):
    """分企业性质（国有 vs 私营）进行模型2主回归。"""
    print("\n" + "=" * 70)
    print("第七部分：异质性分析 — 分产权性质（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()

    group_specs = [
        ("国有企业", df["IsSOE"] == 1),
        ("私营企业", df["IsPrivate"] == 1),
    ]
    rows = []

    for label, mask in group_specs:
        df_sub = df[mask].copy()
        fit_result = _fit_model2(df_sub, control_vars)
        if fit_result is None:
            print(f"  样本量不足 ({len(df_sub)})，跳过。")
            continue
        _print_subsample_model2(label, fit_result)
        rows.append({
            "group_type": "产权性质",
            "group": label,
            "sample_size": fit_result["sample_size"],
            "n_clusters": fit_result["n_clusters"],
            "coef_model2": float(fit_result["model"].params.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "r2_model2": float(fit_result["model"].rsquared),
            "f_stat_model2": float(_get_model_fstat(fit_result["model"])[0]),
        })

    return pd.DataFrame(rows)


def mechanism_analysis(df):
    """
    异质性分析：
    1) 管制行业 vs 非管制行业
    2) 央企 vs 地方国企（仅在可识别国有样本中）
    所有分组仅重复估计主回归模型2。
    """
    print("\n" + "=" * 70)
    print("第八部分：异质性分析 — 行业管制与央地国企")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    rows = []

    # ---- A. 管制行业 vs 非管制行业 ----
    print("\n[行业管制] 管制行业 vs 非管制行业")
    for label, value in [("管制行业", 1), ("非管制行业", 0)]:
        df_sub = df[df["RegulatedIndustry"] == value].copy()
        fit_result = _fit_model2(df_sub, control_vars)
        if fit_result is None:
            print(f"  {label}样本不足，跳过。")
            continue
        _print_subsample_model2(label, fit_result)
        rows.append({
            "group_type": "行业管制",
            "group": label,
            "sample_size": fit_result["sample_size"],
            "n_clusters": fit_result["n_clusters"],
            "coef_model2": float(fit_result["model"].params.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "r2_model2": float(fit_result["model"].rsquared),
            "f_stat_model2": float(_get_model_fstat(fit_result["model"])[0]),
        })

    # ---- B. 央企 vs 地方国企 ----
    print("\n[央地国企] 央企 vs 地方国企（国有样本）")
    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan
    print(f"可识别央地属性比例: {identified_ratio:.2%}")

    for label, value in [("央企", 1), ("地方国企", 0)]:
        df_sub = soe_df[soe_df["IsCentralSOE"] == value].copy()
        fit_result = _fit_model2(df_sub, control_vars)
        if fit_result is None:
            print(f"  {label}样本不足，跳过。")
            continue
        _print_subsample_model2(label, fit_result)
        rows.append({
            "group_type": "央地国企",
            "group": label,
            "sample_size": fit_result["sample_size"],
            "n_clusters": fit_result["n_clusters"],
            "coef_model2": float(fit_result["model"].params.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "r2_model2": float(fit_result["model"].rsquared),
            "f_stat_model2": float(_get_model_fstat(fit_result["model"])[0]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 第九部分：相关性分析
# ============================================================

def correlation_analysis(df):
    """Pearson 相关系数矩阵"""
    print("\n" + "=" * 70)
    print("第九部分：主要变量相关系数矩阵")
    print("=" * 70)

    corr_matrix = _compute_correlation_matrix(df)
    print(f"\n样本量: {len(df[[BASE_SUBSIDY_COL, 'lnSale', 'Roa', 'IA', 'Lever', 'Top1', 'Zone']].dropna())}")
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
    return corr_matrix


# ============================================================
# 第十部分：VIF 多重共线性诊断
# ============================================================

def vif_diagnostics(df):
    """方差膨胀因子 (VIF) 诊断多重共线性"""
    print("\n" + "=" * 70)
    print("第十部分：VIF 多重共线性诊断")
    print("=" * 70)

    vif_df = _compute_vif_table(df).sort_values("VIF", ascending=False)
    print("\n  VIF < 5: 无多重共线性")
    print("  VIF 5-10: 轻度多重共线性")
    print("  VIF > 10: 严重多重共线性\n")
    for _, row in vif_df.iterrows():
        status = "✓" if row['VIF'] < 5 else ("⚠" if row['VIF'] < 10 else "✗")
        print(f"    {row['变量']:15s}: VIF = {row['VIF']:.2f}  {status}")

    return vif_df


# ============================================================
# 第十一部分：稳健性检验
# ============================================================

def robustness_checks(df):
    """稳健性检验：替换变量、样本期缩减、行业子样本。"""
    print("\n" + "=" * 70)
    print("第十一部分：稳健性检验")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    core_vars = [BASE_SUBSIDY_LAG_COL] + control_vars
    rows = []

    # --- 稳健性1：替换被解释变量（薪酬对数替代超额薪酬） ---
    print("\n  [稳健性1] 替换被解释变量: 使用高管前三名薪酬对数替代 Overpay")
    df_r1 = df.dropna(subset=core_vars + ["lnCEOpay", "Year"]).copy()
    if len(df_r1) > 100:
        df_r1 = df_r1.set_index(["Symbol", "Year"], drop=False)
        X_r1, _, _ = _build_fe_matrix(df_r1, core_vars)
        m_r1 = _fit_with_cluster_se(df_r1["lnCEOpay"], X_r1)
        coef = m_r1.params.get(BASE_SUBSIDY_LAG_COL, np.nan)
        pval = m_r1.pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)
        print(f"    {BASE_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r1.rsquared:.4f}")
        print(f"    N={len(df_r1)}, 聚类={df_r1['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "替换因变量",
            "dependent_var": "高管前三名薪酬对数",
            "key_var": BASE_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r1.tstats.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r1)),
            "n_clusters": int(df_r1["Symbol"].nunique()),
            "r_squared": float(m_r1.rsquared),
        })

    # --- 稳健性2：缩小样本期间（2010-2020） ---
    print("\n  [稳健性2] 缩小样本期间: 2010-2020年")
    df_r2 = df[(df["Year"] >= 2010) & (df["Year"] <= 2020)].copy()
    df_r2 = df_r2.dropna(subset=core_vars + ["Overpay", "Year"])
    if len(df_r2) > 100:
        df_r2 = df_r2.set_index(["Symbol", "Year"], drop=False)
        X_r2, _, _ = _build_fe_matrix(df_r2, core_vars)
        m_r2 = _fit_with_cluster_se(df_r2["Overpay"], X_r2)
        coef = m_r2.params.get(BASE_SUBSIDY_LAG_COL, np.nan)
        pval = m_r2.pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)
        print(f"    {BASE_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r2.rsquared:.4f}")
        print(f"    N={len(df_r2)}, 聚类={df_r2['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "缩小样本期(2010-2020)",
            "dependent_var": "Overpay",
            "key_var": BASE_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r2.tstats.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r2)),
            "n_clusters": int(df_r2["Symbol"].nunique()),
            "r_squared": float(m_r2.rsquared),
        })

    # --- 稳健性3：仅制造业样本 ---
    print("\n  [稳健性3] 仅制造业样本")
    df_r4 = df[df["Industry"] == 1].copy()
    df_r4 = df_r4.dropna(subset=core_vars + ["Overpay", "Year"])
    if len(df_r4) > 100:
        df_r4 = df_r4.set_index(["Symbol", "Year"], drop=False)
        X_r4, _, _ = _build_fe_matrix(df_r4, core_vars)
        m_r4 = _fit_with_cluster_se(df_r4["Overpay"], X_r4)
        coef = m_r4.params.get(BASE_SUBSIDY_LAG_COL, np.nan)
        pval = m_r4.pvalues.get(BASE_SUBSIDY_LAG_COL, np.nan)
        print(f"    {BASE_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r4.rsquared:.4f}")
        print(f"    N={len(df_r4)}, 聚类={df_r4['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "仅制造业",
            "dependent_var": "Overpay",
            "key_var": BASE_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r4.tstats.get(BASE_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r4)),
            "n_clusters": int(df_r4["Symbol"].nunique()),
            "r_squared": float(m_r4.rsquared),
        })

    # --- 稳健性4：替换核心解释变量为 ln(1+补助_l1) ---
    print(f"\n  [稳健性4] 替换解释变量: 使用 {ALT_SUBSIDY_LAG_COL}")
    core_vars_alt = [ALT_SUBSIDY_LAG_COL] + control_vars
    df_r5 = df.dropna(subset=core_vars_alt + ["Overpay", "Year"]).copy()
    if len(df_r5) > 100:
        df_r5 = df_r5.set_index(["Symbol", "Year"], drop=False)
        X_r5, _, _ = _build_fe_matrix(df_r5, core_vars_alt)
        m_r5 = _fit_with_cluster_se(df_r5["Overpay"], X_r5)
        coef = m_r5.params.get(ALT_SUBSIDY_LAG_COL, np.nan)
        pval = m_r5.pvalues.get(ALT_SUBSIDY_LAG_COL, np.nan)
        print(f"    {ALT_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r5.rsquared:.4f}")
        print(f"    N={len(df_r5)}, 聚类={df_r5['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "替换解释变量ln(补助，仅正值)",
            "dependent_var": "Overpay",
            "key_var": ALT_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r5.tstats.get(ALT_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r5)),
            "n_clusters": int(df_r5["Symbol"].nunique()),
            "r_squared": float(m_r5.rsquared),
        })

    return pd.DataFrame(rows)


def _save_event_study_plot(event_study_result, output_dir):
    """保存事件研究系数图。"""
    if event_study_result is None:
        return None
    rows = event_study_result.get("rows")
    if rows is None or rows.empty:
        return None

    plot_df = rows.sort_values("event_time").copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        plot_df["event_time"],
        plot_df["coef"],
        yerr=1.96 * plot_df["se"],
        fmt="o-",
        color="#1f4e79",
        ecolor="#7aa6c2",
        elinewidth=1.5,
        capsize=4,
        markersize=5,
    )
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("相对首次大额补贴暴露年份")
    ax.set_ylabel("对 Overpay 的系数")
    ax.set_title("首次大额补贴事件研究")
    ax.set_xticks(plot_df["event_time"].tolist())
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "event_study_plot.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# 数据集构建与结果落盘
# ============================================================

def build_analysis_dataset(data_dir):
    """统一构造用于回归和机器学习的分析数据集。"""
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    sample_summary = list(df.attrs.get("sample_screening_summary", []))
    df = compute_overpay(df)
    sample_summary.append(
        _sample_stage_row(
            "期望薪酬模型完整案例",
            df.dropna(subset=["Overpay"]).copy(),
            note="模型1完整案例，Overpay 为期望薪酬模型残差",
        )
    )
    df, power_diagnostics = compute_power(df, return_diagnostics=True)
    sample_summary.append(
        _sample_stage_row(
            "可用于构造 Power 的样本",
            df.dropna(subset=["Power_FA"]).copy(),
            note="四个管理层权力底层指标同时完整，采用 FA 口径构造 Power",
        )
    )
    df.attrs["sample_screening_summary"] = sample_summary
    return df, power_diagnostics


def save_structured_outputs(
    output_dir,
    analysis_df,
    power_diagnostics,
    main_regression,
    ownership_df,
    mechanism_df,
    robustness_df,
):
    """保存供文稿复用的结构化结果。"""
    os.makedirs(output_dir, exist_ok=True)

    power_vars = power_diagnostics.get("power_vars", ["Tenure", "Dual", "Boardsize", "Insider", "Mgshder"])
    unified_df = main_regression.get("fit_result", {}).get("unified_df")

    # 诊断口径与正文中介统一样本保持一致，避免使用更宽的 Power 可构造样本。
    if unified_df is not None and not unified_df.empty:
        diag_source = unified_df.dropna(subset=power_vars).copy()
        standardized = _standardize_power_inputs(diag_source, power_vars)
        shared_diag = _calc_shared_factor_diagnostics(standardized, power_vars)
        pca_diagnostics = {**shared_diag, **_compute_pca_power(standardized, power_vars)}
        fa_result = _compute_fa_power(standardized, power_vars)
        fa_diagnostics = {
            **shared_diag,
            **fa_result,
            "average_communality": float(np.mean(list(fa_result["communalities"].values()))),
        }
    else:
        pca_diagnostics = power_diagnostics.get("pca", {})
        fa_diagnostics = power_diagnostics.get("fa", {})
    pca_summary = pd.DataFrame([{
        "样本量": pca_diagnostics["n_obs"],
        "KMO": pca_diagnostics["kmo_overall"],
        "Bartlett_Chi2": pca_diagnostics["bartlett_chi2"],
        "Bartlett_df": pca_diagnostics["bartlett_df"],
        "Bartlett_p": pca_diagnostics["bartlett_p"],
        "PC1方差贡献率": pca_diagnostics["explained_variance_ratio_pc1"],
        "PC2方差贡献率": pca_diagnostics.get("explained_variance_ratio_pc2", np.nan),
        "前两主成分累计方差贡献率": pca_diagnostics["cum_explained_variance_pc2"],
        "前三主成分累计方差贡献率": pca_diagnostics["cum_explained_variance_pc3"],
    }])
    pca_summary.to_csv(os.path.join(output_dir, "pca_diagnostics_summary.csv"), index=False)

    pca_var = pd.DataFrame([
        {
            "变量": var,
            "KMO": pca_diagnostics["kmo_per_var"][var],
            "PC1载荷": pca_diagnostics["pc1_loadings"][var],
            "PC2载荷": pca_diagnostics["pc2_loadings"][var],
            "综合载荷": pca_diagnostics["loadings"][var],
        }
        for var in power_vars
    ])
    pca_var.to_csv(os.path.join(output_dir, "pca_variable_metrics.csv"), index=False)

    fa_diag_df = pd.DataFrame([
        {
            "变量": var,
            "KMO": fa_diagnostics["kmo_per_var"][var],
            "FA1载荷": fa_diagnostics["fa1_loadings"][var],
            "FA2载荷": fa_diagnostics["fa2_loadings"][var],
            "综合载荷": fa_diagnostics["loadings"][var],
            "共同度": fa_diagnostics["communalities"][var],
            "特殊方差": fa_diagnostics["uniqueness"][var],
            "综合得分权重": fa_diagnostics["score_weights"][var],
            "KMO总体": fa_diagnostics["kmo_overall"],
            "Bartlett_p": fa_diagnostics["bartlett_p"],
            "前两因子累计方差解释率": fa_diagnostics["variance_explained_ratio"],
        }
        for var in power_vars
    ])
    fa_diag_df.to_csv(os.path.join(output_dir, "power_fa_diagnostics.csv"), index=False)

    desc_df = _compute_descriptive_statistics(analysis_df)
    desc_df.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"), index=False)

    corr_matrix = _compute_correlation_matrix(analysis_df)
    corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"), index=True)

    vif_df = _compute_vif_table(analysis_df).sort_values("VIF", ascending=False)
    vif_df.to_csv(os.path.join(output_dir, "vif_results.csv"), index=False)

    stage1_fit = _fit_expectation_salary_model(analysis_df)
    stage1_model = stage1_fit["model"]
    bp_test = _compute_bp_test(stage1_model)
    wooldridge_test = _compute_wooldridge_serial_test(analysis_df)
    stage1_rows = []
    for var in stage1_fit["model1_vars"]:
        stage1_rows.append({
            "model": "模型1-期望薪酬",
            "key_var": var,
            "coef": float(stage1_model.params.get(var, np.nan)),
            "se": float(stage1_model.bse.get(var, np.nan)),
            "t": float(stage1_model.tvalues.get(var, np.nan)),
            "p": float(stage1_model.pvalues.get(var, np.nan)),
            "n": int(len(stage1_fit["df_model1"])),
            "r2": float(stage1_model.rsquared),
            "note": "OLS + 行业虚拟变量 + 年份虚拟变量；该模型残差直接定义为 Overpay",
        })
    pd.DataFrame(stage1_rows).to_csv(
        os.path.join(output_dir, "stage1_results.csv"),
        index=False,
    )

    sample_summary = list(analysis_df.attrs.get("sample_screening_summary", []))
    sample_summary.append(
        _sample_stage_row(
            "模型2主回归样本",
            analysis_df.dropna(subset=[BASE_SUBSIDY_LAG_COL, "Overpay"] + FE_CONTROL_VARS + ["Year"]).copy(),
            note="主回归要求 Overpay、lnSubsidy_l1 与控制变量同时完整",
        )
    )
    if unified_df is not None and not unified_df.empty:
        sample_summary.append(
            _sample_stage_row(
                "中介统一样本",
                unified_df,
                note="中介模型统一样本，要求 Overpay、Power_FA 与 lnSubsidy_l1 同时完整",
            )
        )
    if sample_summary:
        pd.DataFrame(sample_summary).to_csv(
            os.path.join(output_dir, "sample_screening_summary.csv"),
            index=False,
        )

    power_method_rows = main_regression.get("method_comparison", pd.DataFrame())
    if isinstance(power_method_rows, pd.DataFrame) and not power_method_rows.empty:
        power_method_rows.to_csv(
            os.path.join(output_dir, "power_method_comparison.csv"),
            index=False,
        )

    pd.DataFrame([main_regression["summary"]]).to_csv(
        os.path.join(output_dir, "main_mediation_summary.csv"),
        index=False
    )

    iv_result = main_regression.get("iv_result", {})
    iv_checks = main_regression.get("iv_checks", {})
    iv_comparison = iv_checks.get("iv_comparison", pd.DataFrame())
    heckman_result = iv_checks.get("heckman_result", {})
    placebo_df = main_regression.get("placebo_df", pd.DataFrame())
    event_study_result = main_regression.get("event_study_result")
    iv_model = iv_result.get("model")
    first_stage = iv_result.get("first_stage", {})
    model2 = main_regression.get("model2")
    model3 = main_regression.get("model3")
    model4 = main_regression.get("model4")
    model5 = main_regression.get("model5")
    mediation = main_regression.get("summary", {})
    main_result = main_regression.get("main_result", {})
    causal_rows = []
    if model2 is not None:
        causal_rows.append({
            "category": "双向固定效应",
            "model": "模型2-主回归",
            "key_var": "lnSubsidy_l1",
            "coef": float(model2.params.get("lnSubsidy_l1", np.nan)),
            "se": float(model2.std_errors.get("lnSubsidy_l1", np.nan)),
            "p": float(model2.pvalues.get("lnSubsidy_l1", np.nan)),
            "n": int(main_result.get("sample_size", np.nan)),
            "r2": float(model2.rsquared),
            "note": "公司固定效应+年份固定效应+公司层面聚类稳健标准误",
        })
    if model4 is not None:
        causal_rows.append({
            "category": "中介效应FA",
            "model": "模型4-路径a",
            "key_var": "lnSubsidy_l1→Power_FA",
            "coef": float(mediation.get("coef_a", np.nan)),
            "se": float(model4.std_errors.get("lnSubsidy_l1", np.nan)),
            "p": float(mediation.get("p_a", np.nan)),
            "n": int(mediation.get("sample_size", np.nan)),
            "r2": float(model4.rsquared),
            "note": "FA口径；公司固定效应+年份固定效应",
        })
    if model5 is not None:
        causal_rows.append({
            "category": "中介效应FA",
            "model": "模型5-路径b",
            "key_var": "Power_FA→Overpay",
            "coef": float(mediation.get("coef_b", np.nan)),
            "se": float(model5.std_errors.get("Power", np.nan)),
            "p": float(mediation.get("p_b", np.nan)),
            "n": int(mediation.get("sample_size", np.nan)),
            "r2": float(model5.rsquared),
            "note": "FA口径；公司固定效应+年份固定效应",
        })
        causal_rows.append({
            "category": "中介效应FA",
            "model": "模型5-直接效应",
            "key_var": "lnSubsidy_l1→Overpay|Power",
            "coef": float(mediation.get("coef_c_prime", np.nan)),
            "se": float(model5.std_errors.get("lnSubsidy_l1", np.nan)),
            "p": float(mediation.get("p_c_prime", np.nan)),
            "n": int(mediation.get("sample_size", np.nan)),
            "r2": float(model5.rsquared),
            "note": "FA口径；公司固定效应+年份固定效应",
        })
    if model3 is not None:
        causal_rows.append({
            "category": "中介效应FA",
            "model": "模型3-总效应",
            "key_var": "lnSubsidy_l1→Overpay",
            "coef": float(mediation.get("coef_c", np.nan)),
            "se": float(model3.std_errors.get("lnSubsidy_l1", np.nan)),
            "p": float(mediation.get("p_c", np.nan)),
            "n": int(mediation.get("sample_size", np.nan)),
            "r2": float(model3.rsquared),
            "note": "FA口径；公司固定效应+年份固定效应",
        })
    causal_rows.append({
        "category": "中介效应FA",
        "model": "间接效应a×b",
        "key_var": "indirect",
        "coef": float(mediation.get("indirect_effect", np.nan)),
        "se": np.nan,
        "p": float(mediation.get("sobel_p", np.nan)),
        "n": int(mediation.get("sample_size", np.nan)),
        "r2": np.nan,
        "note": (
            f"Sobel Z={mediation.get('sobel_z', np.nan):.4f}; "
            f"Bootstrap 95%CI=[{mediation.get('bootstrap_ci_lower', np.nan):.6f},"
            f"{mediation.get('bootstrap_ci_upper', np.nan):.6f}]; "
            f"中介占比={mediation.get('mediation_ratio_pct', np.nan):.2f}%"
        ),
    })
    pd.DataFrame(causal_rows).to_csv(
        os.path.join(output_dir, "causal_results.csv"),
        index=False,
    )

    if isinstance(iv_comparison, pd.DataFrame) and not iv_comparison.empty:
        iv_comparison.to_csv(
            os.path.join(output_dir, "iv_comparison_results.csv"),
            index=False,
        )
    heckman_export = (
        {k: v for k, v in heckman_result.items() if k not in {"selection_model", "outcome_model"}}
        if isinstance(heckman_result, dict) and heckman_result
        else {}
    )
    if heckman_export:
        pd.DataFrame([heckman_export]).to_csv(
            os.path.join(output_dir, "heckman_results.csv"),
            index=False,
        )
    if isinstance(placebo_df, pd.DataFrame) and not placebo_df.empty:
        placebo_df.to_csv(
            os.path.join(output_dir, "placebo_results.csv"),
            index=False,
        )
    if event_study_result is not None:
        event_rows = event_study_result.get("rows", pd.DataFrame())
        event_summary = event_study_result.get("summary", {})
        if isinstance(event_rows, pd.DataFrame) and not event_rows.empty:
            event_rows.to_csv(
                os.path.join(output_dir, "event_study_results.csv"),
                index=False,
            )
        if event_summary:
            with open(os.path.join(output_dir, "event_study_summary.json"), "w", encoding="utf-8") as f:
                json.dump(event_summary, f, ensure_ascii=False, indent=2)
        _save_event_study_plot(event_study_result, output_dir)

    diagnostics_summary = {
        "stage1": {
            "sample_size": int(len(stage1_fit["df_model1"])),
            "r2": float(stage1_model.rsquared),
            "adj_r2": float(stage1_model.rsquared_adj),
            "f_stat": float(stage1_model.fvalue),
            "f_pvalue": float(stage1_model.f_pvalue),
        },
        "model2": {
            "sample_size": int(main_result.get("sample_size", np.nan)),
            "r2": float(model2.rsquared) if model2 is not None else np.nan,
            "f_stat": float(_get_model_fstat(model2)[0]) if model2 is not None else np.nan,
            "t_stat": float(model2.tstats.get("lnSubsidy_l1", np.nan)) if model2 is not None else np.nan,
        },
        "iv": {
            "sample_size": int(iv_result.get("sample_size", np.nan)) if iv_result else np.nan,
            "first_stage_coef": float(first_stage.get("coef", np.nan)) if first_stage else np.nan,
            "partial_r2": float(first_stage.get("partial_r2", np.nan)) if first_stage else np.nan,
            "partial_f": float(first_stage.get("f_stat", np.nan)) if first_stage else np.nan,
            "second_stage_coef": float(iv_model.params.get("lnSubsidy_l1", np.nan)) if iv_model is not None else np.nan,
            "second_stage_t": float(iv_model.tstats.get("lnSubsidy_l1", np.nan)) if iv_model is not None else np.nan,
        },
        "iv_benchmark": (
            iv_comparison.loc[iv_comparison["role"] == "benchmark"].iloc[0].to_dict()
            if isinstance(iv_comparison, pd.DataFrame) and not iv_comparison.empty and (iv_comparison["role"] == "benchmark").any()
            else {}
        ),
        "iv_simple": (
            iv_comparison.loc[iv_comparison["role"] == "simple"].iloc[0].to_dict()
            if isinstance(iv_comparison, pd.DataFrame) and not iv_comparison.empty and (iv_comparison["role"] == "simple").any()
            else {}
        ),
        "iv_refined": (
            iv_comparison.loc[iv_comparison["role"] == "refined"].iloc[0].to_dict()
            if isinstance(iv_comparison, pd.DataFrame) and not iv_comparison.empty and (iv_comparison["role"] == "refined").any()
            else {}
        ),
        "heckman": heckman_export,
        "placebo": (
            placebo_df.set_index("check_item").to_dict(orient="index")
            if isinstance(placebo_df, pd.DataFrame) and not placebo_df.empty
            else {}
        ),
        "event_study": event_study_result.get("summary", {}) if event_study_result else {},
        "vif": {
            "max": float(vif_df["VIF"].max()),
            "mean": float(vif_df["VIF"].mean()),
        },
        "power_fa": {
            "kmo_overall": float(fa_diagnostics.get("kmo_overall", np.nan)),
            "variance_explained_ratio": float(fa_diagnostics.get("variance_explained_ratio", np.nan)),
        },
        "bp_test": bp_test,
        "wooldridge_test": wooldridge_test,
    }
    with open(os.path.join(output_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics_summary, f, ensure_ascii=False, indent=2)

    grouped_path = os.path.join(output_dir, "grouped_mediation_results.csv")
    if os.path.exists(grouped_path):
        os.remove(grouped_path)
    ownership_df.to_csv(
        os.path.join(output_dir, "heterogeneity_ownership_results.csv"),
        index=False,
    )
    mechanism_df.to_csv(
        os.path.join(output_dir, "heterogeneity_mechanism_results.csv"),
        index=False,
    )
    robustness_df.to_csv(
        os.path.join(output_dir, "robustness_results.csv"),
        index=False,
    )


# ============================================================
# 主流程
# ============================================================

def main():
    data_dir = os.path.join(ROOT_DIR, "processed_data")
    output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 1-5. 加载数据并构造分析变量
    df, power_diagnostics = build_analysis_dataset(data_dir)

    # 6. 回归分析 (含 Sobel 中介效应检验)
    main_regression = run_regressions(df, power_diagnostics)

    # 7. 异质性分析（分产权）
    ownership_df = heterogeneity_analysis(df)

    # 8. 异质性分析（行业管制、央企/地方国企）
    mechanism_df = mechanism_analysis(df)

    # 9. 相关性分析
    correlation_analysis(df)

    # 10. VIF 多重共线性检验
    vif_diagnostics(df)

    # 11. 稳健性检验
    robustness_df = robustness_checks(df)

    # 保存最终数据集
    output_file = os.path.join(output_dir, "regression_dataset.csv")
    df.to_csv(output_file, index=False)
    save_structured_outputs(
        output_dir,
        df,
        power_diagnostics,
        main_regression,
        ownership_df,
        mechanism_df,
        robustness_df,
    )
    print(f"\n最终分析数据集已保存至: {output_file}")
    print(f"总行数: {len(df)}, 总列数: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
