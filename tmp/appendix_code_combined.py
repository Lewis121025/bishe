from pathlib import Path
import re

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import xgboost as xgb

from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity,
        calculate_kmo,
    )
except ImportError:
    calculate_bartlett_sphericity = None
    calculate_kmo = None


ROOT_DIR = Path.cwd()
RAW_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "processed_data"
OUTPUT_DIR = ROOT_DIR / "output"

ALT_SUBSIDY_LAG_COL = "lnSubsidy_pos_l1"
MAIN_SUBSIDY_LAG_COL = "lnSubsidy_pos_l1"
FE_CONTROL_VARS = ["Roa", "Lever", "Top1"]
MAIN_MEDIATION_BOOTSTRAP_REPS = 300
BOOTSTRAP_RANDOM_SEED = 20260414

EAST_PROVINCES = {
    "北京", "天津", "河北", "上海", "江苏", "浙江", "福建", "山东", "广东", "海南", "辽宁",
}
ALL_PROVINCES = {
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
}
REGULATED_INDUSTRIES = {"电力", "燃气", "自来水", "铁路", "航空", "电信", "金融"}
SPECIAL_TREATMENT_MARKERS = ("ST", "*ST", "S*ST", "SST", "PT", "退")


def _to_symbol(value) -> str:
    """统一股票代码为 6 位字符串。"""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    digits = re.sub(r"\\D", "", value)
    if not digits:
        return np.nan
    return f"{int(digits):06d}"


def _winsorize_series(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """按分位数缩尾。"""
    clean = series.dropna()
    if clean.empty:
        return series
    lower_q = clean.quantile(lower)
    upper_q = clean.quantile(upper)
    return series.clip(lower=lower_q, upper=upper_q)


def _classify_pvalue_signal(pvalue: float) -> str:
    """将 p 值转成论文里常用的显著性表述。"""
    if pd.isna(pvalue):
        return "缺失"
    if pvalue < 0.01:
        return "1% 水平显著"
    if pvalue < 0.05:
        return "5% 水平显著"
    if pvalue < 0.10:
        return "10% 水平显著"
    return "不显著"


def _ci_excludes_zero(ci_lower: float, ci_upper: float) -> bool:
    """判断置信区间是否跨零。"""
    if pd.isna(ci_lower) or pd.isna(ci_upper):
        return False
    return ci_lower > 0 or ci_upper < 0


def process_dta_files(input_dir: Path, output_dir: Path) -> None:
    """将 .dta 转成 .csv，并处理“两个文件上下拼接”的情况。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dta_files = sorted(input_dir.glob("*.dta"))
    print(f"发现 {len(dta_files)} 个 .dta 文件")

    processed = set()
    for file_path in dta_files:
        filename = file_path.name
        if filename in processed:
            continue

        if "两个文件上下拼接" in filename:
            prefix = filename.split("（")[0]
            part_files = sorted(
                p for p in dta_files
                if p.name.startswith(prefix) and "两个文件上下拼接" in p.name
            )
            if not part_files:
                continue

            frames = []
            for part in part_files:
                frames.append(pd.read_stata(part))
                processed.add(part.name)

            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / f"{prefix}.csv", index=False, encoding="utf-8-sig")
            print(f"[拼接完成] {prefix}.csv")
            continue

        df = pd.read_stata(file_path)
        df.to_csv(output_dir / f"{file_path.stem}.csv", index=False, encoding="utf-8-sig")
        processed.add(filename)
        print(f"[转换完成] {file_path.stem}.csv")


if __name__ == "__main__":
    process_dta_files(RAW_DIR, PROCESSED_DIR)


def load_and_clean_data(file_map: dict[str, Path]) -> pd.DataFrame:
    """
    读取并合并 7 类原始数据。

    file_map 建议包含以下键：
      finance, governance, debt, pay, region, ownership, subsidy
    """
    required_keys = {
        "finance", "governance", "debt", "pay", "region", "ownership", "subsidy",
    }
    missing = required_keys - set(file_map)
    if missing:
        raise ValueError(f"缺少必要文件映射: {sorted(missing)}")

    # 1. 财务数据
    df_fin = pd.read_csv(file_map["finance"])
    df_fin = df_fin.rename(
        columns={
            "id": "Symbol",
            "year": "Year",
            "OperatingEvenue": "Revenue",
            "营业收入": "Revenue",
            "净利润": "NetProfit",
            "总资产": "TotalAssets",
            "无形资产": "IntangibleAsset",
            "行业代码": "IndustrySector",
        }
    )
    df_fin["Symbol"] = df_fin["Symbol"].apply(_to_symbol)
    df_fin["Year"] = pd.to_numeric(df_fin["Year"], errors="coerce")

    # 2. 治理数据
    df_gov = pd.read_csv(file_map["governance"])
    df_gov = df_gov.rename(
        columns={
            "Enddate": "EndDate",
            "董事会规模": "DirectorNumber",
            "独立董事占比": "IndependentDirectorRatio",
            "内部董事占比": "InternalDirectorProportion",
            "两职合一": "DualjointTitle",
            "管理层持股比例": "ManagementHoldingPercentage",
        }
    )
    df_gov["Symbol"] = df_gov["Symbol"].apply(_to_symbol)
    df_gov["Year"] = pd.to_datetime(df_gov["EndDate"]).dt.year
    df_gov = df_gov.sort_values(["Symbol", "Year"]).drop_duplicates(["Symbol", "Year"], keep="last")

    # 3. 负债数据
    df_debt = pd.read_csv(file_map["debt"])
    df_debt = df_debt.rename(columns={"Enddate": "EndDate", "总负债": "TotalLiability"})
    df_debt["Symbol"] = df_debt["Symbol"].apply(_to_symbol)
    df_debt["Year"] = pd.to_datetime(df_debt["EndDate"]).dt.year
    df_debt = df_debt[["Symbol", "Year", "TotalLiability"]]

    # 4. 薪酬数据
    df_pay = pd.read_csv(file_map["pay"])
    df_pay = df_pay.rename(
        columns={
            "高管前三名薪酬总合": "Top3Salary",
            "高管前三名薪酬总额": "Top3Salary",
        }
    )
    df_pay["Symbol"] = df_pay["Symbol"].apply(_to_symbol)
    df_pay["Year"] = pd.to_datetime(df_pay["Year"]).dt.year

    # 5. 地区数据
    df_region = pd.read_csv(file_map["region"])
    df_region = df_region.rename(
        columns={
            "注册地址": "ADDRESS_REGISTER",
            "办公地址": "ADDRESS_OFFICE",
            "城市": "City",
        }
    )
    df_region["Symbol"] = df_region["Symbol"].apply(_to_symbol)
    if "Year" not in df_region.columns and "EndDate" in df_region.columns:
        df_region["Year"] = pd.to_datetime(df_region["EndDate"]).dt.year

    # 6. 股东结构/产权数据
    df_owner = pd.read_csv(file_map["ownership"])
    df_owner = df_owner.rename(
        columns={
            "第一大股东持股比例": "LargestHolderRate",
            "所有制性质": "Ownership",
            "是否国有": "IsSOE",
            "是否央企": "IsCentralSOE",
        }
    )
    df_owner["Symbol"] = df_owner["Symbol"].apply(_to_symbol)
    if "Year" not in df_owner.columns and "EndDate" in df_owner.columns:
        df_owner["Year"] = pd.to_datetime(df_owner["EndDate"]).dt.year

    # 7. 政府补助
    df_subsidy = pd.read_csv(file_map["subsidy"])
    df_subsidy = df_subsidy.rename(
        columns={
            "政府补助": "SubsidyAmount",
            "财政补贴": "SubsidyAmount",
        }
    )
    df_subsidy["Symbol"] = df_subsidy["Symbol"].apply(_to_symbol)
    if "Year" not in df_subsidy.columns and "EndDate" in df_subsidy.columns:
        df_subsidy["Year"] = pd.to_datetime(df_subsidy["EndDate"]).dt.year

    merge_keys = ["Symbol", "Year"]
    merged = df_fin.copy()
    for right_df in [df_gov, df_debt, df_pay, df_region, df_owner, df_subsidy]:
        merged = merged.merge(right_df, on=merge_keys, how="left")

    merged = _apply_sample_filters(merged)
    return merged.reset_index(drop=True)


def _apply_sample_filters(df: pd.DataFrame) -> pd.DataFrame:
    """按正文口径进行样本筛选。"""
    df = df.copy()

    # 剔除金融行业
    if "IndustrySector" in df.columns:
        finance_mask = df["IndustrySector"].astype(str).str.contains("J|金融", na=False)
        df = df.loc[~finance_mask].copy()

    # 剔除 ST、*ST 等特殊处理样本
    if "ShortName" in df.columns:
        st_mask = df["ShortName"].astype(str).apply(
            lambda x: any(marker in x for marker in SPECIAL_TREATMENT_MARKERS)
        )
        df = df.loc[~st_mask].copy()

    # 基础数值清洗
    numeric_cols = [
        "Revenue", "NetProfit", "TotalAssets", "IntangibleAsset",
        "TotalLiability", "Top3Salary", "SubsidyAmount", "LargestHolderRate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SubsidyAmount"] = df["SubsidyAmount"].fillna(0)
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    """根据论文口径构造全部核心变量。"""
    df = df.copy()
    df["SubsidyAmount"] = df["SubsidyAmount"].fillna(0).clip(lower=0)

    # 1. 薪酬与补贴变量
    df["lnCEOpay"] = np.log(df["Top3Salary"].where(df["Top3Salary"] > 0))
    df["lnSubsidy"] = np.log1p(df["SubsidyAmount"].clip(lower=0))
    df["lnSubsidy_pos"] = np.log(df["SubsidyAmount"].where(df["SubsidyAmount"] > 0))

    # 2. 财务控制变量
    df["Roa"] = df["NetProfit"] / df["TotalAssets"]
    df["lnSale"] = np.log(df["Revenue"].where(df["Revenue"] > 0))
    df["IA"] = df["IntangibleAsset"] / df["TotalAssets"]
    df["Lever"] = df["TotalLiability"] / df["TotalAssets"]
    df["Top1"] = df["LargestHolderRate"]

    # 3. 区域与产权变量
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
    df["OwnerType"] = df["Ownership"].apply(classify_ownership)
    derived_soe = np.where(df["OwnerType"].isin({"中央国有企业", "地方国有企业"}), 1, 0)
    if "IsSOE" in df.columns:
        df["IsSOE"] = pd.to_numeric(df["IsSOE"], errors="coerce")
        df["IsSOE"] = df["IsSOE"].fillna(pd.Series(derived_soe, index=df.index)).astype(int)
    else:
        df["IsSOE"] = derived_soe
    df["IsPrivate"] = np.where(df["OwnerType"] == "私营企业", 1, 0)
    derived_central = np.where(df["OwnerType"] == "中央国有企业", 1, 0)
    if "IsCentralSOE" in df.columns:
        df["IsCentralSOE"] = pd.to_numeric(df["IsCentralSOE"], errors="coerce")
        df["IsCentralSOE"] = df["IsCentralSOE"].fillna(pd.Series(derived_central, index=df.index)).astype(int)
    else:
        df["IsCentralSOE"] = derived_central

    # 4. 管理层权力底层指标
    df["Dual"] = (
        df[["DualjointTitle", "BothChief"]]
        .fillna(0)
        .max(axis=1)
        .astype(float)
    )
    df["Boardsize"] = pd.to_numeric(df["DirectorNumber"], errors="coerce")
    df["Insider"] = pd.to_numeric(df["InternalDirectorProportion"], errors="coerce")
    df["Mgshder"] = pd.to_numeric(df["ManagementHoldingPercentage"], errors="coerce")

    # 5. 行业管制变量
    df["RegulatedIndustry"] = df["IndustrySector"].apply(_classify_regulated_industry)

    # 6. 滞后与领先变量
    df = df.sort_values(["Symbol", "Year"]).copy()
    grouped = df.groupby("Symbol", sort=False)
    df["SubsidyAmount_l1"] = grouped["SubsidyAmount"].shift(1)
    df["lnSubsidy_l1"] = grouped["lnSubsidy"].shift(1)
    df["lnSubsidy_pos_l1"] = grouped["lnSubsidy_pos"].shift(1)
    df["lnSubsidy_f1"] = grouped["lnSubsidy"].shift(-1)

    # 7. 描述统计与第一阶段模型所用缩尾
    for col in ["Roa", "lnSale", "IA", "Lever", "Top1", "SubsidyAmount"]:
        if col in df.columns:
            df[col] = _winsorize_series(df[col])

    return df


def _classify_zone_from_location(city, address_register, address_office) -> float:
    """东部=0，中西部=1；若无法识别则返回缺失。"""
    ordered_sources = [address_register, address_office, city]
    for source in ordered_sources:
        if pd.isna(source) or not str(source).strip():
            continue
        text = str(source).strip()
        for province in EAST_PROVINCES:
            if province in text:
                return 0.0
        return 1.0
    return 1.0


def _extract_province_from_location(city, address_register, address_office) -> str:
    """按“注册地址优先，办公地址补充，城市兜底”提取省级区域。"""
    ordered_sources = [address_register, address_office, city]
    for source in ordered_sources:
        if pd.isna(source) or not str(source).strip():
            continue
        text = str(source).strip()
        for province in ALL_PROVINCES:
            if province in text:
                return province
    return np.nan


def classify_ownership(value) -> str:
    """按文本标签归类所有制性质。"""
    if pd.isna(value):
        return "未知"
    text = str(value)
    if "中央" in text or "央企" in text:
        return "中央国有企业"
    if "国有" in text:
        return "地方国有企业"
    if "民营" in text or "私营" in text:
        return "私营企业"
    return "其他"


def _classify_regulated_industry(value) -> int:
    """识别是否属于管制行业。"""
    if pd.isna(value):
        return 0
    text = str(value)
    return int(any(keyword in text for keyword in REGULATED_INDUSTRIES))


def build_leave_one_out_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    """按组计算 leave-one-out 平均值，可用于工具变量构造。"""
    df = df.copy()
    group_sum = df.groupby(group_cols)[value_col].transform(lambda x: x.sum(skipna=True))
    group_count = df.groupby(group_cols)[value_col].transform("count")
    df[output_col] = np.where(
        group_count > 1,
        (group_sum - df[value_col].fillna(0)) / (group_count - 1),
        np.nan,
    )
    return df


def _group_minus_subgroup_mean(
    df: pd.DataFrame,
    broad_cols: list[str],
    narrow_cols: list[str],
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    """计算“大组减去小组”后的组均值，用于构造排除本省/本市的诊断工具变量。"""
    df = df.copy()
    broad_sum = df.groupby(broad_cols)[value_col].transform(lambda x: x.sum(skipna=True))
    broad_count = df.groupby(broad_cols)[value_col].transform("count")
    narrow_sum = df.groupby(narrow_cols)[value_col].transform(lambda x: x.sum(skipna=True))
    narrow_count = df.groupby(narrow_cols)[value_col].transform("count")

    valid = broad_count > narrow_count
    df[output_col] = np.where(
        valid,
        (broad_sum - narrow_sum) / (broad_count - narrow_count),
        np.nan,
    )
    return df


def build_diagnostic_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """构造正文 3.4.3 所述留一法诊断工具变量。"""
    work = df.copy()
    if "Province" not in work.columns:
        work["Province"] = work.apply(
            lambda row: _extract_province_from_location(
                row.get("City"),
                row.get("ADDRESS_REGISTER"),
                row.get("ADDRESS_OFFICE"),
            ),
            axis=1,
        )

    value_col = "lnSubsidy_l1"
    work = build_leave_one_out_mean(work, ["City", "Year"], value_col, "IV_city_year_l1")
    work = build_leave_one_out_mean(work, ["IndustrySector", "Year"], value_col, "IV_industry_year_l1")
    work = _group_minus_subgroup_mean(
        work,
        ["IndustrySector", "Year"],
        ["IndustrySector", "Year", "Province"],
        value_col,
        "IV_industry_year_excl_province_l1",
    )
    work = _group_minus_subgroup_mean(
        work,
        ["Province", "IndustrySector", "Year"],
        ["Province", "IndustrySector", "Year", "City"],
        value_col,
        "IV_province_industry_year_excl_city_l1",
    )
    return work


def compute_overpay(df: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """估计期望薪酬模型，并将残差定义为超额薪酬 Overpay。"""
    model_vars = ["lnSale", "Roa", "IA", "Zone", "lnCEOpay", "IndustrySector", "Year"]
    df_model = df.dropna(subset=model_vars).copy()

    X = df_model[["lnSale", "Roa", "IA", "Zone"]].astype(float)
    industry_dummies = pd.get_dummies(df_model["IndustrySector"], prefix="ind", drop_first=True)
    year_dummies = pd.get_dummies(df_model["Year"].astype(int), prefix="year", drop_first=True)
    X = pd.concat([X, industry_dummies, year_dummies], axis=1)
    X = sm.add_constant(X.astype(float), has_constant="add")
    y = df_model["lnCEOpay"].astype(float)

    model = sm.OLS(y, X).fit()
    df_model["Overpay"] = model.resid

    # 回填至主表
    out = df.copy()
    out = out.merge(
        df_model[["Symbol", "Year", "Overpay"]],
        on=["Symbol", "Year"],
        how="left",
    )
    return out, model


def compute_power_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """基于 Dual、Boardsize、Insider、Mgshder 构造管理层权力 PCA 综合指数。"""
    power_vars = ["Dual", "Boardsize", "Insider", "Mgshder"]
    df_complete = df.dropna(subset=power_vars).copy()

    standardized = _standardize_power_inputs(df_complete, power_vars)
    diagnostics = _calc_shared_factor_diagnostics(standardized)
    pca_result = _compute_pca_power(standardized, power_vars)

    df_complete["Power"] = pca_result["scores"]
    out = df.merge(df_complete[["Symbol", "Year", "Power"]], on=["Symbol", "Year"], how="left")
    return out, diagnostics, pca_result


def _standardize_power_inputs(df_complete: pd.DataFrame, power_vars: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(df_complete[power_vars].astype(float))
    return pd.DataFrame(standardized, columns=power_vars, index=df_complete.index)


def _calc_shared_factor_diagnostics(standardized_df: pd.DataFrame) -> dict:
    """计算 KMO 与 Bartlett 检验；若缺少 factor_analyzer，则返回缺失值。"""
    if calculate_kmo is None or calculate_bartlett_sphericity is None:
        return {
            "kmo_overall": np.nan,
            "kmo_per_var": {col: np.nan for col in standardized_df.columns},
            "bartlett_chi2": np.nan,
            "bartlett_p": np.nan,
        }

    kmo_per_var, kmo_overall = calculate_kmo(standardized_df)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(standardized_df)
    return {
        "kmo_overall": float(kmo_overall),
        "kmo_per_var": {
            col: float(kmo_per_var[idx])
            for idx, col in enumerate(standardized_df.columns)
        },
        "bartlett_chi2": float(bartlett_chi2),
        "bartlett_p": float(bartlett_p),
    }


def _compute_pca_power(standardized_df: pd.DataFrame, power_vars: list[str]) -> dict:
    """提取前两个主成分，并按方差贡献率加权得到综合得分。"""
    pca = PCA(n_components=2)
    raw_scores = pca.fit_transform(standardized_df)
    loadings = pca.components_.copy()

    oriented_scores = raw_scores.copy()
    anchor = standardized_df[power_vars].sum(axis=1).to_numpy()
    for idx in range(raw_scores.shape[1]):
        score_vec, sign = _orient_component(raw_scores[:, idx], anchor)
        oriented_scores[:, idx] = score_vec
        loadings[idx, :] = loadings[idx, :] * sign

    weights = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
    composite_raw = oriented_scores @ weights
    composite_scores = (composite_raw - composite_raw.mean()) / composite_raw.std(ddof=0)
    composite_loadings = loadings.T @ weights

    return {
        "scores": composite_scores,
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_pc2": float(pca.explained_variance_ratio_[1]),
        "cum_explained_variance_pc2": float(pca.explained_variance_ratio_[:2].sum()),
        "component_weights": {f"PC{i + 1}": float(w) for i, w in enumerate(weights)},
        "loadings": {var: float(composite_loadings[i]) for i, var in enumerate(power_vars)},
    }


def _orient_component(scores: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, int]:
    """按与锚点变量的相关方向统一主成分符号。"""
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        sign = 1
    else:
        sign = 1 if corr >= 0 else -1
    return scores * sign, sign


def get_main_regression_sample(df: pd.DataFrame) -> pd.DataFrame:
    """获取主回归样本：Overpay、滞后一期正补助和控制变量同时非缺失。"""
    needed = ["Overpay", MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS + ["Symbol", "Year"]
    out = df.dropna(subset=needed).copy()
    out = out.sort_values(["Symbol", "Year"])
    return out


def fit_panel_ols(
    df: pd.DataFrame,
    dependent: str,
    exog_cols: list[str],
    entity_col: str = "Symbol",
    time_col: str = "Year",
    entity_effects: bool = True,
    time_effects: bool = True,
    other_effects: str | None = None,
):
    """统一封装 PanelOLS 固定效应回归。"""
    needed = [entity_col, time_col, dependent] + exog_cols
    if other_effects is not None:
        needed.append(other_effects)
    subset = df.dropna(subset=needed).copy()
    subset = subset.sort_values([entity_col, time_col])
    subset = subset.set_index([entity_col, time_col])

    y = subset[dependent].astype(float)
    X = subset[exog_cols].astype(float)

    other = None
    if other_effects is not None:
        other = subset[[other_effects]]

    model = PanelOLS(
        y,
        X,
        entity_effects=entity_effects,
        time_effects=time_effects,
        other_effects=other,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return result, subset.reset_index()


def run_main_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """模型2：Overpay 对滞后一期正补助强度的固定效应回归。"""
    df_main = get_main_regression_sample(df)
    exog_cols = [MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS
    result, used_sample = fit_panel_ols(df_main, "Overpay", exog_cols)

    summary = {
        "coef": float(result.params[MAIN_SUBSIDY_LAG_COL]),
        "stderr": float(result.std_errors[MAIN_SUBSIDY_LAG_COL]),
        "t_stat": float(result.tstats[MAIN_SUBSIDY_LAG_COL]),
        "pvalue": float(result.pvalues[MAIN_SUBSIDY_LAG_COL]),
        "r2": float(result.rsquared),
        "nobs": int(result.nobs),
        "n_clusters": int(used_sample["Symbol"].nunique()),
    }
    return df_main, summary


def run_heckman_two_step(df: pd.DataFrame) -> dict:
    """Heckman 两步法：校正“进入正补助样本”的选择偏差。"""
    work = build_diagnostic_instruments(df)
    work["Selected"] = np.where(work[MAIN_SUBSIDY_LAG_COL].notna(), 1, 0)

    select_vars = ["IV_city_year_l1", "Roa", "Lever", "Top1", "Year", "Selected"]
    first_stage = work.dropna(subset=select_vars).copy()

    year_dummies = pd.get_dummies(first_stage["Year"].astype(int), prefix="year", drop_first=True)
    X_select = pd.concat(
        [first_stage[["IV_city_year_l1", "Roa", "Lever", "Top1"]].astype(float), year_dummies],
        axis=1,
    )
    X_select = sm.add_constant(X_select, has_constant="add")
    probit = sm.Probit(first_stage["Selected"].astype(float), X_select).fit(disp=False)

    xb = np.asarray(X_select @ probit.params)
    cdf = stats.norm.cdf(xb)
    pdf = stats.norm.pdf(xb)
    first_stage["IMR"] = np.where(
        first_stage["Selected"] == 1,
        pdf / np.clip(cdf, 1e-8, None),
        -pdf / np.clip(1 - cdf, 1e-8, None),
    )

    second_stage = work.merge(
        first_stage[["Symbol", "Year", "IMR"]],
        on=["Symbol", "Year"],
        how="left",
    )
    second_stage = second_stage[second_stage["Selected"] == 1].copy()

    result, used = fit_panel_ols(
        second_stage,
        dependent="Overpay",
        exog_cols=[MAIN_SUBSIDY_LAG_COL, "IMR"] + FE_CONTROL_VARS,
    )

    return {
        "first_stage_n": int(len(first_stage)),
        "second_stage_n": int(result.nobs),
        "selection_coef": float(probit.params["IV_city_year_l1"]),
        "selection_pvalue": float(probit.pvalues["IV_city_year_l1"]),
        "imr_coef": float(result.params["IMR"]),
        "imr_pvalue": float(result.pvalues["IMR"]),
        "subsidy_coef": float(result.params[MAIN_SUBSIDY_LAG_COL]),
        "subsidy_pvalue": float(result.pvalues[MAIN_SUBSIDY_LAG_COL]),
        "n_clusters": int(used["Symbol"].nunique()),
    }


def _two_way_demean(df: pd.DataFrame, cols: list[str], entity_col="Symbol", time_col="Year") -> pd.DataFrame:
    """对变量做公司和年份双向去均值，以实现 FE-2SLS 的 within 变换。"""
    values = df[cols].astype(float)
    entity_mean = df.groupby(entity_col)[cols].transform("mean")
    time_mean = df.groupby(time_col)[cols].transform("mean")
    overall_mean = values.mean()
    return values - entity_mean - time_mean + overall_mean


def _build_wald_matrix(param_names: list[str], target_names: list[str]) -> np.ndarray:
    """为联合显著性检验构造限制矩阵。"""
    R = np.zeros((len(target_names), len(param_names)))
    for i, name in enumerate(target_names):
        if name in param_names:
            R[i, param_names.index(name)] = 1.0
    return R


def run_iv_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """留一法 IV 诊断：同城、同行业和双工具 FE-2SLS。"""
    work = build_diagnostic_instruments(df)
    base = get_main_regression_sample(work)

    specs = {
        "同城同年其他企业平均补助": ["IV_city_year_l1"],
        "同行业同年其他企业平均补助": ["IV_industry_year_l1"],
        "精炼双工具": [
            "IV_industry_year_excl_province_l1",
            "IV_province_industry_year_excl_city_l1",
        ],
    }

    rows = []
    needed_base = ["Overpay", MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS + ["Symbol", "Year"]
    for spec_name, instrument_cols in specs.items():
        sample = base.dropna(subset=needed_base + instrument_cols).copy()
        if len(sample) < 100:
            continue

        dm = _two_way_demean(sample, ["Overpay", MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS + instrument_cols)
        y_dm = dm["Overpay"]
        x_dm = dm[FE_CONTROL_VARS]
        endog_dm = dm[[MAIN_SUBSIDY_LAG_COL]]
        z_dm = dm[instrument_cols]

        iv_res = IV2SLS(y_dm, x_dm, endog_dm, z_dm).fit(
            cov_type="clustered",
            clusters=sample["Symbol"],
        )

        first_stage_X = pd.concat([x_dm, z_dm], axis=1)
        first_stage_res = sm.OLS(endog_dm.iloc[:, 0], first_stage_X).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["Symbol"]},
        )
        R = _build_wald_matrix(list(first_stage_res.params.index), instrument_cols)
        first_stage_test = first_stage_res.wald_test(R)

        rows.append({
            "specification": spec_name,
            "coef": float(iv_res.params[MAIN_SUBSIDY_LAG_COL]),
            "p_value": float(iv_res.pvalues[MAIN_SUBSIDY_LAG_COL]),
            "first_stage_f": float(np.squeeze(first_stage_test.statistic)),
            "first_stage_p": float(first_stage_test.pvalue),
            "sample_size": int(len(sample)),
        })

    return pd.DataFrame(rows)


def run_event_study_dynamic(df: pd.DataFrame, max_pre=3, max_post=3) -> dict:
    """事件研究式动态检验：考察补贴强度跃升前后的 Overpay 变化。"""
    work = get_main_regression_sample(df).copy()
    work = work.dropna(subset=["SubsidyAmount_l1"]).copy()

    positive_subsidy = work.loc[work["SubsidyAmount_l1"] > 0, "SubsidyAmount_l1"]
    threshold = float(positive_subsidy.quantile(0.75))
    work["HighSubsidyEvent"] = work["SubsidyAmount_l1"] >= threshold

    first_event_year = (
        work.loc[work["HighSubsidyEvent"]]
        .groupby("Symbol")["Year"]
        .min()
    )
    work["EventYear"] = work["Symbol"].map(first_event_year)
    work["EventTime"] = work["Year"] - work["EventYear"]

    event_cols = []
    for k in range(-max_pre, max_post + 1):
        if k == -1:
            continue
        name = f"event_m{abs(k)}" if k < 0 else ("event_0" if k == 0 else f"event_p{k}")
        work[name] = np.where(work["EventTime"] == k, 1.0, 0.0)
        event_cols.append(name)

    result, used = fit_panel_ols(
        work,
        dependent="Overpay",
        exog_cols=event_cols + FE_CONTROL_VARS,
    )

    pre_cols = [col for col in event_cols if col.startswith("event_m")]
    pre_R = _build_wald_matrix(list(result.params.index), pre_cols)
    pretrend_test = result.wald_test(pre_R)

    coef_table = []
    for col in event_cols:
        coef_table.append({
            "term": col,
            "coef": float(result.params[col]),
            "p_value": float(result.pvalues[col]),
        })

    treated_firms = int(work["EventYear"].notna().groupby(work["Symbol"]).max().sum())
    return {
        "threshold": threshold,
        "treated_firms": treated_firms,
        "sample_size": int(result.nobs),
        "pretrend_pvalue": float(pretrend_test.pval),
        "coefficients": coef_table,
        "n_clusters": int(used["Symbol"].nunique()),
    }


def analyze_mediation_effects(df_main: pd.DataFrame) -> dict:
    """中介效应检验：路径 a、路径 b、Sobel 与 Bootstrap。"""
    mediation_vars = [MAIN_SUBSIDY_LAG_COL, "Overpay", "Power"] + FE_CONTROL_VARS
    df_mediation = df_main.dropna(subset=mediation_vars).copy()

    # 路径 a: Subsidy -> Power
    result_a, _ = fit_panel_ols(
        df_mediation,
        dependent="Power",
        exog_cols=[MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS,
    )

    # 路径 b 与直接效应: Subsidy + Power -> Overpay
    result_b, _ = fit_panel_ols(
        df_mediation,
        dependent="Overpay",
        exog_cols=[MAIN_SUBSIDY_LAG_COL, "Power"] + FE_CONTROL_VARS,
    )

    coef_a = float(result_a.params[MAIN_SUBSIDY_LAG_COL])
    se_a = float(result_a.std_errors[MAIN_SUBSIDY_LAG_COL])
    coef_b = float(result_b.params["Power"])
    se_b = float(result_b.std_errors["Power"])
    coef_direct = float(result_b.params[MAIN_SUBSIDY_LAG_COL])

    indirect = coef_a * coef_b
    se_indirect = np.sqrt((coef_b ** 2) * (se_a ** 2) + (coef_a ** 2) * (se_b ** 2))
    z_sobel = indirect / se_indirect if se_indirect > 0 else np.nan
    p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel))) if pd.notna(z_sobel) else np.nan

    bootstrap = bootstrap_mediation(
        df_mediation,
        subsidy_var=MAIN_SUBSIDY_LAG_COL,
        control_vars=FE_CONTROL_VARS,
        reps=MAIN_MEDIATION_BOOTSTRAP_REPS,
    )

    return {
        "path_a_coef": coef_a,
        "path_a_pvalue": float(result_a.pvalues[MAIN_SUBSIDY_LAG_COL]),
        "path_b_coef": coef_b,
        "path_b_pvalue": float(result_b.pvalues["Power"]),
        "direct_effect": coef_direct,
        "direct_pvalue": float(result_b.pvalues[MAIN_SUBSIDY_LAG_COL]),
        "indirect_effect": indirect,
        "sobel_z": z_sobel,
        "sobel_pvalue": p_sobel,
        "bootstrap": bootstrap,
        "bootstrap_significant": _ci_excludes_zero(
            bootstrap["ci_lower"],
            bootstrap["ci_upper"],
        ),
    }


def bootstrap_mediation(
    df: pd.DataFrame,
    subsidy_var: str,
    control_vars: list[str],
    reps: int = 300,
) -> dict:
    """公司层面 Bootstrap，估计间接效应区间。"""
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    effects = []

    entities = df["Symbol"].dropna().unique()
    for _ in range(reps):
        sampled_entities = rng.choice(entities, size=len(entities), replace=True)
        boot_df = _resample_panel_by_entity(df, sampled_entities)

        try:
            result_a, _ = fit_panel_ols(
                boot_df,
                dependent="Power",
                exog_cols=[subsidy_var] + control_vars,
                entity_col="BootEntity",
            )
            result_b, _ = fit_panel_ols(
                boot_df,
                dependent="Overpay",
                exog_cols=[subsidy_var, "Power"] + control_vars,
                entity_col="BootEntity",
            )
        except Exception:
            continue

        effects.append(float(result_a.params[subsidy_var]) * float(result_b.params["Power"]))

    if len(effects) < 30:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_success": len(effects),
        }

    effects = np.asarray(effects)
    p_empirical = 2 * min(np.mean(effects <= 0), np.mean(effects >= 0))
    return {
        "mean": float(effects.mean()),
        "ci_lower": float(np.percentile(effects, 2.5)),
        "ci_upper": float(np.percentile(effects, 97.5)),
        "p_value": float(min(p_empirical, 1.0)),
        "n_success": int(len(effects)),
    }


def _resample_panel_by_entity(df: pd.DataFrame, sampled_entities: np.ndarray) -> pd.DataFrame:
    """对面板数据做公司层面重抽样，并重置实体编号。"""
    parts = []
    for i, entity in enumerate(sampled_entities):
        part = df[df["Symbol"] == entity].copy()
        part["BootEntity"] = f"boot_{i:05d}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_model2(
    df: pd.DataFrame,
    dependent: str = "Overpay",
    subsidy_col: str = MAIN_SUBSIDY_LAG_COL,
    fe_mode: str = "entity_time",
):
    """模型2的统一估计接口。"""
    exog_cols = [subsidy_col] + FE_CONTROL_VARS
    work = df.copy()

    if fe_mode == "entity_time":
        result, used = fit_panel_ols(work, dependent=dependent, exog_cols=exog_cols)
        return result, used

    if fe_mode == "entity_industry_year":
        work["IndustryYearFE"] = (
            work["IndustrySector"].astype(str) + "_" + work["Year"].astype(int).astype(str)
        )
        result, used = fit_panel_ols(
            work,
            dependent=dependent,
            exog_cols=exog_cols,
            time_effects=False,
            other_effects="IndustryYearFE",
        )
        return result, used

    raise ValueError(f"未知固定效应模式: {fe_mode}")


def robustness_checks(df: pd.DataFrame) -> pd.DataFrame:
    """稳健性检验：替换因变量、收缩样本期、制造业子样本、行业×年份固定效应。"""
    df_main = get_main_regression_sample(df)
    rows = []
    df_logpay = df.dropna(
        subset=["lnCEOpay", MAIN_SUBSIDY_LAG_COL] + FE_CONTROL_VARS + ["Symbol", "Year"]
    ).copy()

    checks = [
        (
            "替换因变量",
            df_logpay,
            {"dependent": "lnCEOpay", "fe_mode": "entity_time"},
        ),
        (
            "缩小样本期(2010-2020)",
            df_main[df_main["Year"].between(2010, 2020)].copy(),
            {"dependent": "Overpay", "fe_mode": "entity_time"},
        ),
        (
            "仅制造业",
            df_main[df_main["IndustrySector"].astype(str).str.startswith("C", na=False)].copy(),
            {"dependent": "Overpay", "fe_mode": "entity_time"},
        ),
        (
            "公司 + 行业×年份固定效应",
            df_main.dropna(subset=["IndustrySector"]).copy(),
            {"dependent": "Overpay", "fe_mode": "entity_industry_year"},
        ),
    ]

    for check_name, sample_df, kwargs in checks:
        if len(sample_df) < 100:
            continue
        result, used = fit_model2(sample_df, **kwargs)
        rows.append({
            "check_item": check_name,
            "coef": float(result.params[MAIN_SUBSIDY_LAG_COL]),
            "p_value": float(result.pvalues[MAIN_SUBSIDY_LAG_COL]),
            "r_squared": float(result.rsquared),
            "sample_size": int(result.nobs),
            "n_clusters": int(used["Symbol"].nunique()),
        })

    return pd.DataFrame(rows)


def heterogeneity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """产权性质、行业管制、央地国企分组回归。"""
    df_main = get_main_regression_sample(df)
    rows = []

    group_specs = [
        ("产权性质", "国有企业", df_main["IsSOE"] == 1),
        ("产权性质", "私营企业", df_main["IsPrivate"] == 1),
        ("行业管制", "管制行业", df_main["RegulatedIndustry"] == 1),
        ("行业管制", "非管制行业", df_main["RegulatedIndustry"] == 0),
        ("央地国企", "央企", (df_main["IsSOE"] == 1) & (df_main["IsCentralSOE"] == 1)),
        ("央地国企", "地方国企", (df_main["IsSOE"] == 1) & (df_main["IsCentralSOE"] == 0)),
    ]

    for group_type, label, mask in group_specs:
        sample_df = df_main.loc[mask].copy()
        if len(sample_df) < 100:
            continue

        result, used = fit_model2(sample_df)
        rows.append({
            "group_type": group_type,
            "group": label,
            "coef_model2": float(result.params[MAIN_SUBSIDY_LAG_COL]),
            "p_model2": float(result.pvalues[MAIN_SUBSIDY_LAG_COL]),
            "r2_model2": float(result.rsquared),
            "sample_size": int(result.nobs),
            "n_clusters": int(used["Symbol"].nunique()),
        })

    return pd.DataFrame(rows)


def machine_learning_verification(df: pd.DataFrame, output_dir: Path) -> dict:
    """随机森林、SHAP、Lasso、XGBoost 的统一入口。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_data = prepare_ml_data(df)
    splits = create_holdout_splits(ml_data["X"], ml_data["y"], ml_data["df_ml"])

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    groups_train = splits["groups_train"]
    feature_names = ml_data["feature_names"]

    rf_result = random_forest_analysis(
        X_train, X_test, y_train, y_test, groups_train, feature_names,
    )
    shap_result = rf_shap_analysis(rf_result["model"], X_train, feature_names)
    lasso_result = lasso_analysis(
        X_train, X_test, y_train, y_test, groups_train, feature_names,
    )
    xgb_result = xgboost_analysis(X_train, X_test, y_train, y_test, feature_names)

    summary = {
        "random_forest": rf_result,
        "shap": shap_result,
        "lasso": lasso_result,
        "xgboost": xgb_result,
    }

    summary_df = pd.DataFrame({
        "方法": ["随机森林", "SHAP", "Lasso", "XGBoost"],
        "补贴变量排名": [
            rf_result["subsidy_rank"],
            shap_result["subsidy_rank"],
            "被保留" if lasso_result["subsidy_retained"] else "被压缩",
            xgb_result["subsidy_rank"],
        ],
        "方向/系数": [
            f"重要性={rf_result['subsidy_importance']:.4f}",
            f"SHAP={shap_result['subsidy_importance']:.4f}",
            f"系数={lasso_result['subsidy_coef']:.4f} ({lasso_result['subsidy_sign']})",
            f"重要性={xgb_result['subsidy_importance']:.4f}",
        ],
        "测试集R²": [
            rf_result["test_metrics"]["R2"],
            np.nan,
            lasso_result["test_metrics"]["R2"],
            xgb_result["test_metrics"]["R2"],
        ],
    })
    summary_df.to_csv(output_dir / "ml_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def prepare_ml_data(df: pd.DataFrame) -> dict:
    """保持与主回归一致的样本边界和特征集合。"""
    feature_cols = [ALT_SUBSIDY_LAG_COL, "Roa", "Lever", "Top1"]
    target_col = "Overpay"
    use_cols = feature_cols + [target_col, "Symbol", "Year"]
    df_ml = df.dropna(subset=use_cols).copy()

    return {
        "df_ml": df_ml,
        "X": df_ml[feature_cols].copy(),
        "y": df_ml[target_col].copy(),
        "feature_names": feature_cols,
    }


def create_holdout_splits(X, y, df_ml) -> dict:
    """按企业分组划分训练集与测试集。"""
    groups = df_ml["Symbol"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return {
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "groups_train": groups.iloc[train_idx].copy(),
        "groups_test": groups.iloc[test_idx].copy(),
    }


def random_forest_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names) -> dict:
    """随机森林：考察变量重要性排序。"""
    cv = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 10],
            "min_samples_leaf": [5, 10, 20],
            "max_features": ["sqrt", 0.6],
        },
        n_iter=10,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": model.feature_importances_,
    }).sort_values("重要性", ascending=False)

    subsidy_row = importance_df.loc[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1

    return {
        "model": model,
        "test_metrics": {"R2": float(r2_score(y_test, y_pred))},
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["重要性"]),
        "importance_table": importance_df.to_dict(orient="records"),
    }


def rf_shap_analysis(rf_model, X_reference, feature_names) -> dict:
    """SHAP：考察财政补贴变量的平均边际贡献。"""
    shap_sample = X_reference.sample(n=min(2000, len(X_reference)), random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "平均绝对SHAP": mean_abs,
    }).sort_values("平均绝对SHAP", ascending=False)

    subsidy_row = importance_df.loc[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1

    return {
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["平均绝对SHAP"]),
        "importance_table": importance_df.to_dict(orient="records"),
    }


def lasso_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names) -> dict:
    """Lasso：检查补贴变量是否被保留，以及符号是否与正文一致。"""
    alpha_grid = np.logspace(-4, 1, 50)
    cv = GroupKFold(n_splits=5)

    best_alpha = None
    best_score = -np.inf
    for alpha in alpha_grid:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000)),
        ])
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="r2",
        )
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = alpha

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, max_iter=20000)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coefs = model.named_steps["model"].coef_
    coef_df = pd.DataFrame({"特征": feature_names, "系数": coefs})
    subsidy_coef = float(coef_df.loc[coef_df["特征"] == ALT_SUBSIDY_LAG_COL, "系数"].iloc[0])

    return {
        "best_alpha": float(best_alpha),
        "test_metrics": {"R2": float(r2_score(y_test, y_pred))},
        "subsidy_coef": subsidy_coef,
        "subsidy_retained": bool(abs(subsidy_coef) > 1e-6),
        "subsidy_sign": "正" if subsidy_coef > 0 else ("负" if subsidy_coef < 0 else "零"),
        "coef_table": coef_df.to_dict(orient="records"),
    }


def xgboost_analysis(X_train, X_test, y_train, y_test, feature_names) -> dict:
    """XGBoost：考察特征重要性与部分依赖趋势。"""
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": model.feature_importances_,
    }).sort_values("重要性", ascending=False)
    subsidy_row = importance_df.loc[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1

    subsidy_idx = feature_names.index(ALT_SUBSIDY_LAG_COL)
    pdp = partial_dependence(model, X_test, features=[subsidy_idx], grid_resolution=50)
    pd_trend = "递增" if pdp["average"][0][-1] > pdp["average"][0][0] else "递减"

    return {
        "model": model,
        "test_metrics": {"R2": float(r2_score(y_test, y_pred))},
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["重要性"]),
        "pdp_trend": pd_trend,
        "importance_table": importance_df.to_dict(orient="records"),
    }
