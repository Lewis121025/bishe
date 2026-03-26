"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 实证分析脚本
=============================================================================

参考论文：刘剑民(2019)《政府补助、管理层权力与国有企业高管超额薪酬》
开题报告：基于数据挖掘的上市公司财政补贴与高管超额薪酬研究

研究模型：
  模型1（期望薪酬）: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ΣYear + ε
  模型2（超额薪酬）: Overpay = lnCEOpay_actual - lnCEOpay_expected（直接使用模型1残差）
  模型3（基准回归）: Overpay = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + α₅·Zone + ΣIndustry + ΣYear + ε
  模型4（中介效应）: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + α₃·Roa + α₄·Lever + α₅·Top1 + α₆·Zone + ΣIndustry + ΣYear + ε
  模型5（权力回归）: Power  = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + α₅·Zone + ΣIndustry + ΣYear + ε
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# 行业管制口径（参考文献常见口径并结合现有行业编码字母）
# B: 采矿业, D: 电力热力燃气及水生产和供应业, G: 交通运输仓储和邮政业
# I: 信息传输软件和信息技术服务业, N: 水利环境和公共设施管理业
REGULATED_INDUSTRY_SECTORS = {"B", "D", "G", "I", "N"}

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


def _format_pvalue(pvalue):
    """格式化 p 值展示。"""
    if pd.isna(pvalue):
        return "nan"
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.4f}"


def _classify_mediation_pvalue(pvalue):
    """按统一阈值给出中介效应判定。"""
    if pd.isna(pvalue):
        return "无法判定"
    if pvalue < 0.05:
        return "存在中介效应"
    if pvalue < 0.10:
        return "边际中介效应"
    return "不存在中介效应"


def _stable_seed_from_text(label, base_seed=BOOTSTRAP_RANDOM_SEED):
    """为每个分组生成稳定的 bootstrap 随机种子。"""
    return int(base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(label))))


def _cluster_bootstrap_indirect_effect(x4, y_overpay, x5, y_power, groups, reps, seed):
    """公司层面 cluster bootstrap 估计间接效应 a×b 的分布。"""
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

    power_idx = x4.columns.get_loc("Power")
    subsidy_idx = x5.columns.get_loc("lnSubsidy")

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


def _derive_rigor_conclusion(summary):
    """综合原始 Sobel、FDR 和 bootstrap 给出更谨慎的研究结论。"""
    bootstrap_supported = summary.get("bootstrap_conclusion") == "95%CI不含0"
    fdr_p = summary.get("fdr_p", np.nan)
    sobel_p = summary.get("sobel_p", np.nan)

    if pd.notna(fdr_p):
        if fdr_p < 0.05 and bootstrap_supported:
            return "校正后稳健支持"
        if fdr_p < 0.10 and bootstrap_supported:
            return "探索性稳健支持"
        if sobel_p < 0.10:
            return "仅原始Sobel支持"
        return "无稳健证据"

    if sobel_p < 0.05 and bootstrap_supported:
        return "稳健支持"
    if sobel_p < 0.10:
        return "仅原始Sobel支持"
    return "无稳健证据"


def _apply_layerwise_fdr(summary_df):
    """按层级分别进行 BH-FDR 校正。"""
    result_df = summary_df.copy()
    result_df["fdr_p"] = np.nan
    result_df["fdr_conclusion"] = "不适用"

    for layer, idx in result_df.groupby("layer").groups.items():
        idx = list(idx)
        pvals = result_df.loc[idx, "sobel_p"].to_numpy(dtype=float)
        valid_mask = np.isfinite(pvals)
        adjusted = np.full(len(pvals), np.nan)
        if valid_mask.any():
            adjusted_valid = multipletests(pvals[valid_mask], method="fdr_bh")[1]
            adjusted[valid_mask] = adjusted_valid
        result_df.loc[idx, "fdr_p"] = adjusted
        result_df.loc[idx, "fdr_conclusion"] = [
            _classify_mediation_pvalue(value) if np.isfinite(value) else "无法判定"
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
    df_gov = df_gov[["Symbol", "Year", "ConcurrentPosition", "Mngmhldn",
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
    df_loc = df_loc[["Symbol", "Year", "Ownership", "City"]]
    print(f"  {len(df_loc)} 行")

    # --- 6. 第一大股东持股比例 + 实际控制人股权性质 ---
    print("[6/8] 加载股东数据...")
    holder_path = os.path.join(data_dir, "第一大股东持股比例+实际控制人股权性质.csv")
    _assert_not_lfs_pointer(holder_path)
    df_holder = pd.read_csv(holder_path)
    df_holder["Symbol"] = df_holder["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_holder["Year"] = pd.to_datetime(df_holder["EndDate"]).dt.year
    df_holder = df_holder.sort_values("EndDate").groupby(["Symbol", "Year"]).last().reset_index()
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
    df_sub_agg = df_sub.groupby(["Symbol", "Year"])["SubsidyAmount"].sum().reset_index()
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
    print("2. 构造 lnSubsidy = ln(政府补助)")
    # 基准口径：仅对正补助取对数，非正值置为缺失（避免将负值机械截断为1）
    positive_subsidy = df["SubsidyAmount"].where(df["SubsidyAmount"] > 0)
    df["lnSubsidy"] = np.log(positive_subsidy)
    # 备选口径：ln(1+补助)，用于稳健性检验
    df["lnSubsidy1p"] = np.log1p(df["SubsidyAmount"].clip(lower=0))
    non_positive_n = int((df["SubsidyAmount"] <= 0).fillna(False).sum())
    print(f"   非正补助样本（lnSubsidy 置缺失）: {non_positive_n}")

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

    # 4. Zone 区域虚拟变量：东部=0, 中西部=1
    print("4. 构造区域变量 Zone...")
    eastern_cities = [
        "北京市", "天津市", "上海市",  # 注：重庆属于西部地区，不纳入东部
        "石家庄市", "唐山市", "秦皇岛市", "邯郸市", "保定市", "沧州市", "廊坊市", "衡水市",  # 河北
        "沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市", "丹东市", "锦州市", "营口市",  # 辽宁
        "南京市", "无锡市", "徐州市", "常州市", "苏州市", "南通市", "连云港市", "淮安市",  # 江苏
        "盐城市", "扬州市", "镇江市", "泰州市", "宿迁市",
        "杭州市", "宁波市", "温州市", "嘉兴市", "湖州市", "绍兴市", "金华市", "台州市",  # 浙江
        "福州市", "厦门市", "莆田市", "泉州市", "漳州市", "龙岩市", "三明市", "南平市",  # 福建
        "济南市", "青岛市", "烟台市", "威海市", "潍坊市", "淄博市", "枣庄市", "东营市",  # 山东
        "济宁市", "泰安市", "临沂市", "德州市", "聊城市", "滨州市", "菏泽市",
        "广州市", "深圳市", "珠海市", "汕头市", "佛山市", "东莞市", "中山市", "惠州市",  # 广东
        "江门市", "湛江市", "茂名市", "肇庆市", "梅州市", "揭阳市",
        "海口市", "三亚市",  # 海南
    ]
    # 也可以基于省份来判断
    eastern_provinces = ["北京", "天津", "河北", "辽宁", "上海", "江苏", "浙江",
                         "福建", "山东", "广东", "海南"]

    def classify_zone(city):
        if pd.isna(city):
            return np.nan
        for prov in eastern_provinces:
            if prov in str(city):
                return 0  # 东部
        for c in eastern_cities:
            if str(city) in c or c in str(city):
                return 0
        return 1  # 中西部

    df["Zone"] = df["City"].apply(classify_zone)

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
    core_vars = ["lnCEOpay", "lnSubsidy", "lnSale", "Roa", "IA", "Lever",
                 "Top1", "Zone", "Industry", "Dual", "Insider", "Mgshder",
                 "Tenure", "Boardsize"]

    print(f"\n筛选前总样本: {len(df)} 行")

    # 剔除金融行业 (IndustryCode1 以 J 开头)
    df = df[~df["IndustryCode1"].str.startswith("J", na=False)].copy()
    print(f"剔除金融行业后: {len(df)} 行")

    # 要求核心变量非空
    key_vars = ["lnCEOpay", "lnSubsidy", "Roa", "Lever", "Top1", "Zone", "Industry"]
    df_clean = df.dropna(subset=key_vars).copy()
    print(f"剔除关键变量缺失后: {len(df_clean)} 行")

    # 原始薪酬数据已在源数据阶段缩尾，此处不再对 lnCEOpay / Overpay 二次缩尾
    print("\n对除薪酬外的连续变量进行 1%-99% Winsorize 缩尾处理...")
    continuous_vars = ["lnSubsidy", "lnSale", "Roa", "IA", "Lever", "Top1"]
    for var in continuous_vars:
        if var in df_clean.columns:
            lower = df_clean[var].quantile(0.01)
            upper = df_clean[var].quantile(0.99)
            df_clean[var] = df_clean[var].clip(lower, upper)

    # 描述性统计
    print("\n" + "-" * 50)
    print("表1 主要变量描述性统计")
    print("-" * 50)

    desc_vars = ["Top3Salary", "SubsidyAmount", "Roa", "Lever", "Top1",
                 "lnCEOpay", "lnSubsidy"]
    available_desc = [v for v in desc_vars if v in df_clean.columns]
    desc = df_clean[available_desc].describe().T
    desc = desc[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["N", "Mean", "Std", "Min", "Median", "Max"]
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n样本企业性质分布:")
    print(df_clean["OwnerType"].value_counts())

    print(f"\n年度分布:")
    print(df_clean["Year"].value_counts().sort_index())

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


# ============================================================
# 第四部分：期望薪酬模型 → 超额薪酬 Overpay
# ============================================================

def compute_overpay(df):
    """
    模型1: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ΣYear + ε
    模型2: Overpay = lnCEOpay_actual - lnCEOpay_expected (即残差)
    """
    print("\n" + "=" * 70)
    print("第四部分：期望薪酬模型与超额薪酬 (Overpay)")
    print("=" * 70)

    # 准备自变量
    model1_vars = ["lnSale", "Roa", "IA", "Zone"]
    df_model1 = df.dropna(subset=model1_vars + ["lnCEOpay", "IndustrySector"]).copy()

    # 构建 X: 核心变量 + 行业虚拟变量 + 年份虚拟变量
    X = df_model1[model1_vars].copy()
    ind_dummies = get_industry_dummies(df_model1)
    year_dummies = get_year_dummies(df_model1)
    X = pd.concat([X, ind_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_model1["lnCEOpay"]

    print(f"\n模型1 样本量: {len(df_model1)}")
    print(f"核心自变量: {model1_vars}")
    print(f"行业虚拟变量: {len(ind_dummies.columns)} 个")
    print(f"年份虚拟变量: {len(year_dummies.columns)} 个")

    # OLS 回归
    model1 = sm.OLS(y, X).fit()
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
# 第五部分：管理层权力 Power (PCA主成分分析)
# ============================================================

def _calc_pca_diagnostics(df_complete, power_vars):
    """计算 KMO、Bartlett 球形检验和 PCA 方差贡献率。"""
    corr = np.corrcoef(df_complete[power_vars].astype(float), rowvar=False)
    inv_corr = np.linalg.inv(corr)

    # Partial correlation matrix
    scale = np.diag(1 / np.sqrt(np.diag(inv_corr)))
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
    chi_square = -(n_obs - 1 - (2 * n_var + 5) / 6) * np.log(np.linalg.det(corr))
    dof = n_var * (n_var - 1) / 2
    bartlett_p = 1 - stats.chi2.cdf(chi_square, dof)

    standardized = (df_complete[power_vars] - df_complete[power_vars].mean()) / df_complete[power_vars].std(ddof=0)
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


def compute_power(df, return_diagnostics=False):
    """
    借鉴 Finkelstein(1992)、刘剑民(2019) 方法:
    管理层权力 = PCA(Tenure, Dual, Boardsize, Insider, Mgshder)
    """
    print("\n" + "=" * 70)
    print("第五部分：管理层权力综合指标 (Power - PCA)")
    print("=" * 70)

    power_vars = ["Tenure", "Dual", "Boardsize", "Insider", "Mgshder"]

    # 检查缺失情况
    for v in power_vars:
        missing = df[v].isna().sum()
        total = len(df)
        print(f"  {v}: 缺失 {missing} ({missing/total*100:.1f}%)")

    df_pca = df.dropna(subset=power_vars).copy()
    print(f"\n可用于 PCA 的样本量: {len(df_pca)}")

    if len(df_pca) < 100:
        print("  WARNING: 样本量过少，跳过 PCA。")
        df["Power"] = np.nan
        diagnostics = {
            "n_obs": int(len(df_pca)),
            "kmo_overall": np.nan,
            "bartlett_chi2": np.nan,
            "bartlett_df": np.nan,
            "bartlett_p": np.nan,
            "explained_variance_ratio_pc1": np.nan,
            "cum_explained_variance_pc2": np.nan,
            "cum_explained_variance_pc3": np.nan,
            "kmo_per_var": {var: np.nan for var in power_vars},
            "loadings": {var: np.nan for var in power_vars},
        }
        return (df, diagnostics) if return_diagnostics else df

    diagnostics = _calc_pca_diagnostics(df_pca, power_vars)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca[power_vars])

    pca = PCA(n_components=1)
    power_scores = pca.fit_transform(X_scaled)
    loadings = {var: float(loading) for var, loading in zip(power_vars, pca.components_[0])}
    diagnostics["loadings"] = loadings

    print(f"\n  KMO = {diagnostics['kmo_overall']:.4f}")
    for var in power_vars:
        print(f"    KMO[{var:10s}] = {diagnostics['kmo_per_var'][var]:.4f}")
    print(
        f"  Bartlett 球形检验: Chi² = {diagnostics['bartlett_chi2']:.4f}, "
        f"df = {diagnostics['bartlett_df']}, p = {_format_pvalue(diagnostics['bartlett_p'])}"
    )
    print(f"  第一主成分解释方差比: {diagnostics['explained_variance_ratio_pc1']:.4f}")
    print(f"  前两主成分累计解释方差比: {diagnostics['cum_explained_variance_pc2']:.4f}")
    print(f"  前三主成分累计解释方差比: {diagnostics['cum_explained_variance_pc3']:.4f}")
    print("  各变量载荷:")
    for var, loading in loadings.items():
        print(f"    {var:15s}: {loading:.4f}")

    df_pca["Power"] = power_scores.flatten()

    df = df.merge(df_pca[["Symbol", "Year", "Power"]],
                  on=["Symbol", "Year"], how="left")

    print(f"\n管理层权力 (Power) 描述:")
    print(df["Power"].describe().to_string())

    return (df, diagnostics) if return_diagnostics else df


# ============================================================
# 第六部分：回归分析
# ============================================================

def _build_fe_matrix(df_subset, core_vars):
    """构建包含核心变量 + 行业固定效应 + 年份固定效应的 X 矩阵"""
    X = df_subset[core_vars].copy()
    ind_dummies = get_industry_dummies(df_subset)
    year_dummies = get_year_dummies(df_subset)
    X = pd.concat([X, ind_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    return X, len(ind_dummies.columns), len(year_dummies.columns)

def _fit_with_cluster_se(y, X, groups):
    """OLS 回归 + 公司层面聚类稳健标准误"""
    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': groups}
    )
    return model

def _print_core_results(model, core_vars, n_ind, n_year, sample_size, n_clusters=None):
    """只打印核心变量的回归结果，固定效应只显示'已控制'"""
    result_table = model.summary2().tables[1]
    core_rows = ["const"] + core_vars
    filtered = result_table.loc[result_table.index.isin(core_rows)]
    print(f"样本量: {sample_size}")
    print(filtered.to_string())
    print(f"行业固定效应: 已控制 ({n_ind} 个)")
    print(f"年份固定效应: 已控制 ({n_year} 个)")
    if n_clusters:
        print(f"聚类标准误: 公司层面 ({n_clusters} 个聚类)")
    print(f"R² = {model.rsquared:.4f}")


def _summarize_mediation(label, m3, m4, m5, sample_size, n_clusters):
    """汇总 Baron & Kenny 路径系数和 Sobel 检验结果。"""
    coef_c = m3.params.get("lnSubsidy", np.nan)
    p_c = m3.pvalues.get("lnSubsidy", np.nan)
    coef_c_prime = m4.params.get("lnSubsidy", np.nan)
    p_c_prime = m4.pvalues.get("lnSubsidy", np.nan)
    coef_b = m4.params.get("Power", np.nan)
    p_b = m4.pvalues.get("Power", np.nan)
    se_b = m4.bse.get("Power", np.nan)
    coef_a = m5.params.get("lnSubsidy", np.nan)
    p_a = m5.pvalues.get("lnSubsidy", np.nan)
    se_a = m5.bse.get("lnSubsidy", np.nan)

    denominator = np.sqrt(coef_b ** 2 * se_a ** 2 + coef_a ** 2 * se_b ** 2)
    sobel_z = (coef_a * coef_b) / denominator if denominator and not np.isnan(denominator) else np.nan
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan
    indirect_effect = coef_a * coef_b
    mediation_ratio = indirect_effect / coef_c * 100 if (not np.isnan(coef_c) and coef_c != 0) else np.nan

    return {
        "group": label,
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
        "conclusion": _classify_mediation_pvalue(sobel_p),
        "fdr_p": np.nan,
        "fdr_conclusion": "不适用",
        "bootstrap_reps": 0,
        "bootstrap_mean_indirect": np.nan,
        "bootstrap_p": np.nan,
        "bootstrap_ci_lower": np.nan,
        "bootstrap_ci_upper": np.nan,
        "bootstrap_conclusion": "未执行",
        "rigor_conclusion": "未评估",
    }


def _print_mediation_summary(summary, indent="  "):
    """打印中介效应摘要。"""
    print(f"\n{indent}逐步回归法 (Baron & Kenny, 1986):")
    print(
        f"{indent}  第一步 (模型3) lnSubsidy→Overpay:  "
        f"c={summary['coef_c']:.4f} (p={_format_pvalue(summary['p_c'])})"
    )
    print(
        f"{indent}  第二步 (模型5) lnSubsidy→Power:    "
        f"a={summary['coef_a']:.4f} (p={_format_pvalue(summary['p_a'])})"
    )
    print(
        f"{indent}  第三步 (模型4) lnSubsidy→Overpay:  "
        f"c'={summary['coef_c_prime']:.4f} (p={_format_pvalue(summary['p_c_prime'])})"
    )
    print(
        f"{indent}  第三步 (模型4) Power→Overpay:      "
        f"b={summary['coef_b']:.4f} (p={_format_pvalue(summary['p_b'])})"
    )
    print(f"\n{indent}Sobel 检验:")
    print(f"{indent}  间接效应 (a×b) = {summary['indirect_effect']:.6f}")
    print(f"{indent}  Sobel Z = {summary['sobel_z']:.4f}")
    print(f"{indent}  Sobel p = {_format_pvalue(summary['sobel_p'])}")
    print(f"{indent}  中介效应占总效应比例 = {summary['mediation_ratio_pct']:.2f}%")
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
        print(f"{indent}  FDR 判定 = {summary['fdr_conclusion']}")
    print(f"\n{indent}原始结论：{summary['conclusion']}")
    print(f"{indent}严格口径：{summary.get('rigor_conclusion', '未评估')}")


def _augment_summary_with_bootstrap(fit_result, reps, seed):
    """为中介检验结果补充 cluster bootstrap 统计量。"""
    bootstrap_result = _cluster_bootstrap_indirect_effect(
        fit_result["x4"],
        fit_result["y4"],
        fit_result["x5"],
        fit_result["y5"],
        fit_result["groups"],
        reps=reps,
        seed=seed,
    )
    fit_result["summary"].update(bootstrap_result)
    fit_result["summary"]["rigor_conclusion"] = _derive_rigor_conclusion(fit_result["summary"])
    return fit_result


def _run_mediation_models(df_sub, label, control_vars, dep_var="Overpay"):
    """在给定样本上统一拟合模型3/4/5并汇总中介效应。"""
    core3 = ["lnSubsidy"] + control_vars
    core4 = ["lnSubsidy", "Power"] + control_vars
    core5 = ["lnSubsidy"] + control_vars

    needed = core4 + [dep_var, "IndustrySector"]
    unified = df_sub.dropna(subset=needed).copy()
    if len(unified) < 80:
        return None

    groups = unified["Symbol"]
    x3, n_ind3, n_year3 = _build_fe_matrix(unified, core3)
    x4, n_ind4, n_year4 = _build_fe_matrix(unified, core4)
    x5, n_ind5, n_year5 = _build_fe_matrix(unified, core5)

    m3 = _fit_with_cluster_se(unified[dep_var], x3, groups)
    m4 = _fit_with_cluster_se(unified[dep_var], x4, groups)
    m5 = _fit_with_cluster_se(unified["Power"], x5, groups)

    n_unified = len(unified)
    n_clusters = groups.nunique()
    summary = _summarize_mediation(label, m3, m4, m5, n_unified, n_clusters)

    return {
        "label": label,
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "x4": x4,
        "x5": x5,
        "y4": unified[dep_var],
        "y5": unified["Power"],
        "groups": groups,
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


def _fit_model3_and_4(df_sub, control_vars):
    """在给定子样本上拟合模型3和模型4（统一样本）。"""
    core3 = ["lnSubsidy"] + control_vars
    core4 = ["lnSubsidy", "Power"] + control_vars

    # 统一样本：取模型4所需的全部变量（是模型3的超集）
    unified = df_sub.dropna(subset=core4 + ["Overpay", "IndustrySector"]).copy()

    if len(unified) < 80:
        return None

    groups = unified["Symbol"]

    x3, n_ind3, n_year3 = _build_fe_matrix(unified, core3)
    y3 = unified["Overpay"]
    m3 = _fit_with_cluster_se(y3, x3, groups)

    x4, n_ind4, n_year4 = _build_fe_matrix(unified, core4)
    y4 = unified["Overpay"]
    m4 = _fit_with_cluster_se(y4, x4, groups)

    n_unified = len(unified)
    n_cl = groups.nunique()

    return {
        "m3": m3,
        "m4": m4,
        "n3": n_unified,
        "n4": n_unified,
        "ind3": n_ind3,
        "year3": n_year3,
        "ind4": n_ind4,
        "year4": n_year4,
        "cl3": n_cl,
        "cl4": n_cl,
    }


def _print_subsample_model34(label, fit_result):
    """打印子样本的模型3/4核心结果。"""
    m3 = fit_result["m3"]
    m4 = fit_result["m4"]
    print(f"\n{'=' * 40}")
    print(f"子样本: {label}")
    print(f"{'=' * 40}")
    print(
        f"模型3 N={fit_result['n3']} (聚类={fit_result['cl3']}), "
        f"模型4 N={fit_result['n4']} (聚类={fit_result['cl4']})"
    )
    print(
        f"模型3 lnSubsidy={m3.params.get('lnSubsidy', np.nan):.4f} "
        f"(p={m3.pvalues.get('lnSubsidy', np.nan):.4f}), R²={m3.rsquared:.4f}"
    )
    print(
        f"模型4 lnSubsidy={m4.params.get('lnSubsidy', np.nan):.4f} "
        f"(p={m4.pvalues.get('lnSubsidy', np.nan):.4f}), "
        f"Power={m4.params.get('Power', np.nan):.4f} "
        f"(p={m4.pvalues.get('Power', np.nan):.4f}), R²={m4.rsquared:.4f}"
    )


def run_regressions(df):
    """
    模型3: Overpay = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    模型4: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + 控制 + ΣIndustry + ΣYear + ε
    模型5: Power   = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε

    注意：三个模型使用 **同一批样本** 以确保中介效应系数对比有效。
    """
    print("\n" + "=" * 70)
    print("第六部分：回归分析（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]

    fit_result = _run_mediation_models(df, "全样本", control_vars)
    if fit_result is None:
        raise RuntimeError("全样本无法完成模型3/4/5回归，请检查缺失值处理。")
    _augment_summary_with_bootstrap(
        fit_result,
        reps=MAIN_MEDIATION_BOOTSTRAP_REPS,
        seed=_stable_seed_from_text("全样本"),
    )
    df_unified = df.dropna(subset=["lnSubsidy", "Power"] + control_vars + ["Overpay", "IndustrySector"]).copy()
    print(f"\n统一样本量（模型3/4/5共用）: N = {len(df_unified)}")
    print(f"  聚类数（公司数）: {df_unified['Symbol'].nunique()}")

    # ---- 模型3：基准回归 ----
    print("\n" + "-" * 50)
    print("模型3: Overpay = f(lnSubsidy, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)
    core3 = ["lnSubsidy"] + control_vars
    model3 = fit_result["m3"]
    _print_core_results(model3, core3, fit_result["ind3"], fit_result["year3"], len(df_unified), fit_result["cl3"])

    # ---- 模型4：加入管理层权力（中介效应） ----
    print("\n" + "-" * 50)
    print("模型4: Overpay = f(lnSubsidy, Power, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)
    core4 = ["lnSubsidy", "Power"] + control_vars
    model4 = fit_result["m4"]
    _print_core_results(model4, core4, fit_result["ind4"], fit_result["year4"], len(df_unified), fit_result["cl4"])

    # ---- 模型5：政府补助对管理层权力 ----
    print("\n" + "-" * 50)
    print("模型5: Power = f(lnSubsidy, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)
    core5 = ["lnSubsidy"] + control_vars
    model5 = fit_result["m5"]
    _print_core_results(model5, core5, fit_result["ind5"], fit_result["year5"], len(df_unified), fit_result["cl5"])

    # ---- 中介效应判断 (Baron & Kenny + Sobel Test) ----
    print("\n" + "=" * 50)
    print("中介效应检验")
    print("=" * 50)
    _print_mediation_summary(fit_result["summary"])

    return {
        "model3": model3,
        "model4": model4,
        "model5": model5,
        "summary": fit_result["summary"],
        "fit_result": fit_result,
    }


# ============================================================
# 第七部分：异质性分析（分产权性质）
# ============================================================

def heterogeneity_analysis(df):
    """分企业性质（国有 vs 私营）进行回归，含行业和年份固定效应"""
    print("\n" + "=" * 70)
    print("第七部分：异质性分析 — 分产权性质（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]

    group_specs = [
        ("国有企业", df["IsSOE"] == 1),
        ("私营企业", df["IsPrivate"] == 1),
    ]

    for label, mask in group_specs:
        df_sub = df[mask].copy()
        fit_result = _fit_model3_and_4(df_sub, control_vars)
        if fit_result is None:
            print(f"  样本量不足 ({len(df_sub)})，跳过。")
            continue
        _print_subsample_model34(label, fit_result)


def mechanism_analysis(df):
    """
    机制检验：
    1) 管制行业 vs 非管制行业
    2) 央企 vs 地方国企（仅在可识别国有样本中）
    """
    print("\n" + "=" * 70)
    print("第八部分：机制检验 — 行业管制与央地国企")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]

    # ---- A. 管制行业 vs 非管制行业 ----
    print("\n[机制A] 管制行业 vs 非管制行业")
    for label, value in [("管制行业", 1), ("非管制行业", 0)]:
        df_sub = df[df["RegulatedIndustry"] == value].copy()
        fit_result = _fit_model3_and_4(df_sub, control_vars)
        if fit_result is None:
            print(f"  {label}样本不足，跳过。")
            continue
        _print_subsample_model34(label, fit_result)

    # ---- B. 央企 vs 地方国企 ----
    print("\n[机制B] 央企 vs 地方国企（国有样本）")
    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan
    print(f"可识别央地属性比例: {identified_ratio:.2%}")

    for label, value in [("央企", 1), ("地方国企", 0)]:
        df_sub = soe_df[soe_df["IsCentralSOE"] == value].copy()
        fit_result = _fit_model3_and_4(df_sub, control_vars)
        if fit_result is None:
            print(f"  {label}样本不足，跳过。")
            continue
        _print_subsample_model34(label, fit_result)


def get_grouped_mediation_specs(df):
    """预设需报告的分组中介样本。"""
    return [
        {"layer": "产权直分", "label": "国有", "mask": df["IsSOE"] == 1},
        {"layer": "产权直分", "label": "私营", "mask": df["IsPrivate"] == 1},
        {"layer": "产权直分", "label": "央企", "mask": df["IsCentralSOE"] == 1},
        {"layer": "产权直分", "label": "地方国企", "mask": df["IsCentralSOE"] == 0},
        {"layer": "补充分组", "label": "东部", "mask": df["Zone"] == 0},
        {"layer": "补充分组", "label": "中西部", "mask": df["Zone"] == 1},
        {"layer": "补充分组", "label": "2003-2012", "mask": (df["Year"] >= 2003) & (df["Year"] <= 2012)},
        {"layer": "补充分组", "label": "2013-2024", "mask": (df["Year"] >= 2013) & (df["Year"] <= 2024)},
        {"layer": "补充分组", "label": "国有×东部", "mask": (df["IsSOE"] == 1) & (df["Zone"] == 0)},
        {"layer": "补充分组", "label": "地方国企×东部", "mask": (df["IsCentralSOE"] == 0) & (df["Zone"] == 0)},
        {"layer": "补充分组", "label": "地方国企×后期", "mask": (df["IsCentralSOE"] == 0) & (df["Year"] >= 2013)},
    ]


def grouped_mediation_analysis(df):
    """对预设分组逐一执行完整中介检验。"""
    print("\n" + "=" * 70)
    print("第九部分：分组中介效应检验")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    fit_results = []
    current_layer = None

    for spec in get_grouped_mediation_specs(df):
        if spec["layer"] != current_layer:
            current_layer = spec["layer"]
            print(f"\n[{current_layer}]")

        sub_df = df[spec["mask"]].copy()
        fit_result = _run_mediation_models(sub_df, spec["label"], control_vars)
        if fit_result is None:
            print(f"  {spec['label']}: 样本量不足，跳过。")
            continue

        fit_result["summary"]["layer"] = spec["layer"]
        fit_results.append(fit_result)
        summary = fit_result["summary"]

        print(
            f"  {spec['label']}: N={summary['sample_size']}, 聚类={summary['n_clusters']}, "
            f"a={summary['coef_a']:.4f} (p={_format_pvalue(summary['p_a'])}), "
            f"b={summary['coef_b']:.4f} (p={_format_pvalue(summary['p_b'])}), "
            f"Sobel p={_format_pvalue(summary['sobel_p'])} -> {summary['conclusion']}"
        )

    if not fit_results:
        return pd.DataFrame()

    raw_df = pd.DataFrame([fit_result["summary"] for fit_result in fit_results])
    fdr_df = _apply_layerwise_fdr(raw_df)
    for idx, fit_result in enumerate(fit_results):
        fit_result["summary"]["fdr_p"] = float(fdr_df.iloc[idx]["fdr_p"])
        fit_result["summary"]["fdr_conclusion"] = fdr_df.iloc[idx]["fdr_conclusion"]

    print("\n[分层 FDR 校正]")
    for fit_result in fit_results:
        summary = fit_result["summary"]
        print(
            f"  {summary['group']}: 原始 p={_format_pvalue(summary['sobel_p'])}, "
            f"FDR p={_format_pvalue(summary['fdr_p'])} -> {summary['fdr_conclusion']}"
        )

    print(f"\n[Cluster Bootstrap 复核：仅对原始 Sobel p < {GROUPED_BOOTSTRAP_TRIGGER_P:.2f} 的分组执行]")
    for fit_result in fit_results:
        summary = fit_result["summary"]
        if pd.notna(summary["sobel_p"]) and summary["sobel_p"] < GROUPED_BOOTSTRAP_TRIGGER_P:
            _augment_summary_with_bootstrap(
                fit_result,
                reps=GROUPED_MEDIATION_BOOTSTRAP_REPS,
                seed=_stable_seed_from_text(summary["group"]),
            )
            print(
                f"  {summary['group']}: 95% CI = "
                f"[{summary['bootstrap_ci_lower']:.6f}, {summary['bootstrap_ci_upper']:.6f}] "
                f"-> {summary['bootstrap_conclusion']}；严格口径：{summary['rigor_conclusion']}"
            )
        else:
            summary["rigor_conclusion"] = _derive_rigor_conclusion(summary)
            print(f"  {summary['group']}: 未执行 bootstrap；严格口径：{summary['rigor_conclusion']}")

    ordered_cols = [
        "layer", "group", "sample_size", "n_clusters",
        "coef_c", "p_c", "coef_a", "p_a", "coef_b", "p_b",
        "coef_c_prime", "p_c_prime", "indirect_effect",
        "sobel_z", "sobel_p", "fdr_p", "mediation_ratio_pct", "conclusion", "fdr_conclusion",
        "bootstrap_reps", "bootstrap_mean_indirect", "bootstrap_p",
        "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_conclusion", "rigor_conclusion",
    ]
    return pd.DataFrame([fit_result["summary"] for fit_result in fit_results])[ordered_cols]


# ============================================================
# 第十部分：相关性分析
# ============================================================

def correlation_analysis(df):
    """Pearson 相关系数矩阵"""
    print("\n" + "=" * 70)
    print("第十部分：主要变量相关系数矩阵")
    print("=" * 70)

    corr_vars = ["Overpay", "lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone", "Industry"]
    available = [v for v in corr_vars if v in df.columns]
    df_corr = df[available].dropna()

    corr_matrix = df_corr.corr()
    print(f"\n样本量: {len(df_corr)}")
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))


# ============================================================
# 第十一部分：VIF 多重共线性诊断
# ============================================================

def vif_diagnostics(df):
    """方差膨胀因子 (VIF) 诊断多重共线性"""
    print("\n" + "=" * 70)
    print("第十一部分：VIF 多重共线性诊断")
    print("=" * 70)

    check_vars = ["lnSubsidy", "Roa", "Lever", "Top1", "Zone", "Industry"]
    df_vif = df[check_vars].dropna().copy()
    df_vif = sm.add_constant(df_vif)

    vif_data = []
    for i, col in enumerate(df_vif.columns):
        if col == 'const':
            continue
        vif_val = variance_inflation_factor(df_vif.values, i)
        vif_data.append({"变量": col, "VIF": vif_val})
    
    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    print("\n  VIF < 5: 无多重共线性")
    print("  VIF 5-10: 轻度多重共线性")
    print("  VIF > 10: 严重多重共线性\n")
    for _, row in vif_df.iterrows():
        status = "✓" if row['VIF'] < 5 else ("⚠" if row['VIF'] < 10 else "✗")
        print(f"    {row['变量']:15s}: VIF = {row['VIF']:.2f}  {status}")

    return vif_df


# ============================================================
# 第十二部分：稳健性检验
# ============================================================

def robustness_checks(df):
    """稳健性检验：替换变量、样本期缩减、极值处理、行业子样本、滞后项。"""
    print("\n" + "=" * 70)
    print("第十二部分：稳健性检验")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    core_vars = ["lnSubsidy"] + control_vars

    # --- 稳健性1：替换被解释变量（薪酬对数替代超额薪酬） ---
    print("\n  [稳健性1] 替换被解释变量: 使用 lnCEOpay 替代 Overpay")
    df_r1 = df.dropna(subset=core_vars + ["lnCEOpay", "IndustrySector"]).copy()
    X_r1, n_ind, n_year = _build_fe_matrix(df_r1, core_vars)
    m_r1 = _fit_with_cluster_se(df_r1["lnCEOpay"], X_r1, df_r1["Symbol"])
    coef = m_r1.params.get("lnSubsidy", np.nan)
    pval = m_r1.pvalues.get("lnSubsidy", np.nan)
    print(f"    lnSubsidy 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r1.rsquared:.4f}")
    print(f"    N={len(df_r1)}, 聚类={df_r1['Symbol'].nunique()}个")
    print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")

    # --- 稳健性2：缩小样本期间（2010-2020） ---
    print("\n  [稳健性2] 缩小样本期间: 2010-2020年")
    df_r2 = df[(df["Year"] >= 2010) & (df["Year"] <= 2020)].copy()
    df_r2 = df_r2.dropna(subset=core_vars + ["Overpay", "IndustrySector"])
    if len(df_r2) > 100:
        X_r2, _, _ = _build_fe_matrix(df_r2, core_vars)
        m_r2 = _fit_with_cluster_se(df_r2["Overpay"], X_r2, df_r2["Symbol"])
        coef = m_r2.params.get("lnSubsidy", np.nan)
        pval = m_r2.pvalues.get("lnSubsidy", np.nan)
        print(f"    lnSubsidy 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r2.rsquared:.4f}")
        print(f"    N={len(df_r2)}, 聚类={df_r2['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")

    # --- 稳健性3：剔除极端补助值（上下5%） ---
    print("\n  [稳健性3] 剔除极端补助值（5%/95%分位数外）")
    q05 = df["lnSubsidy"].quantile(0.05)
    q95 = df["lnSubsidy"].quantile(0.95)
    df_r3 = df[(df["lnSubsidy"] >= q05) & (df["lnSubsidy"] <= q95)].copy()
    df_r3 = df_r3.dropna(subset=core_vars + ["Overpay", "IndustrySector"])
    if len(df_r3) > 100:
        X_r3, _, _ = _build_fe_matrix(df_r3, core_vars)
        m_r3 = _fit_with_cluster_se(df_r3["Overpay"], X_r3, df_r3["Symbol"])
        coef = m_r3.params.get("lnSubsidy", np.nan)
        pval = m_r3.pvalues.get("lnSubsidy", np.nan)
        print(f"    lnSubsidy 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r3.rsquared:.4f}")
        print(f"    N={len(df_r3)}, 聚类={df_r3['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")

    # --- 稳健性4：仅制造业样本 ---
    print("\n  [稳健性4] 仅制造业样本")
    df_r4 = df[df["Industry"] == 1].copy()
    df_r4 = df_r4.dropna(subset=core_vars + ["Overpay", "IndustrySector"])
    if len(df_r4) > 100:
        X_r4, _, _ = _build_fe_matrix(df_r4, core_vars)
        m_r4 = _fit_with_cluster_se(df_r4["Overpay"], X_r4, df_r4["Symbol"])
        coef = m_r4.params.get("lnSubsidy", np.nan)
        pval = m_r4.pvalues.get("lnSubsidy", np.nan)
        print(f"    lnSubsidy 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r4.rsquared:.4f}")
        print(f"    N={len(df_r4)}, 聚类={df_r4['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")

    # --- 稳健性5：替换核心解释变量为 ln(1+补助) ---
    print("\n  [稳健性5] 替换解释变量: 使用 lnSubsidy1p")
    core_vars_alt = ["lnSubsidy1p"] + control_vars
    df_r5 = df.dropna(subset=core_vars_alt + ["Overpay", "IndustrySector"]).copy()
    if len(df_r5) > 100:
        X_r5, _, _ = _build_fe_matrix(df_r5, core_vars_alt)
        m_r5 = _fit_with_cluster_se(df_r5["Overpay"], X_r5, df_r5["Symbol"])
        coef = m_r5.params.get("lnSubsidy1p", np.nan)
        pval = m_r5.pvalues.get("lnSubsidy1p", np.nan)
        print(f"    lnSubsidy1p 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r5.rsquared:.4f}")
        print(f"    N={len(df_r5)}, 聚类={df_r5['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")

    # --- 稳健性6：滞后一期政府补助 ---
    print("\n  [稳健性6] 使用滞后一期补助变量 lnSubsidy_l1")
    df_lag = df.sort_values(["Symbol", "Year"]).copy()
    df_lag["lnSubsidy_l1"] = df_lag.groupby("Symbol")["lnSubsidy"].shift(1)
    core_vars_lag = ["lnSubsidy_l1"] + control_vars
    df_r6 = df_lag.dropna(subset=core_vars_lag + ["Overpay", "IndustrySector"]).copy()
    if len(df_r6) > 100:
        X_r6, _, _ = _build_fe_matrix(df_r6, core_vars_lag)
        m_r6 = _fit_with_cluster_se(df_r6["Overpay"], X_r6, df_r6["Symbol"])
        coef = m_r6.params.get("lnSubsidy_l1", np.nan)
        pval = m_r6.pvalues.get("lnSubsidy_l1", np.nan)
        print(f"    lnSubsidy_l1 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r6.rsquared:.4f}")
        print(f"    N={len(df_r6)}, 聚类={df_r6['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")


# ============================================================
# 数据集构建与结果落盘
# ============================================================

def build_analysis_dataset(data_dir):
    """统一构造用于回归和机器学习的分析数据集。"""
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    df = compute_overpay(df)
    df, pca_diagnostics = compute_power(df, return_diagnostics=True)
    return df, pca_diagnostics


def save_structured_outputs(output_dir, pca_diagnostics, main_regression, grouped_mediation_df):
    """保存供文稿复用的结构化结果。"""
    os.makedirs(output_dir, exist_ok=True)

    pca_summary = pd.DataFrame([{
        "样本量": pca_diagnostics["n_obs"],
        "KMO": pca_diagnostics["kmo_overall"],
        "Bartlett_Chi2": pca_diagnostics["bartlett_chi2"],
        "Bartlett_df": pca_diagnostics["bartlett_df"],
        "Bartlett_p": pca_diagnostics["bartlett_p"],
        "PC1方差贡献率": pca_diagnostics["explained_variance_ratio_pc1"],
        "前两主成分累计方差贡献率": pca_diagnostics["cum_explained_variance_pc2"],
        "前三主成分累计方差贡献率": pca_diagnostics["cum_explained_variance_pc3"],
    }])
    pca_summary.to_csv(os.path.join(output_dir, "pca_diagnostics_summary.csv"), index=False)

    pca_var = pd.DataFrame([
        {
            "变量": var,
            "KMO": pca_diagnostics["kmo_per_var"][var],
            "载荷": pca_diagnostics["loadings"][var],
        }
        for var in ["Tenure", "Dual", "Boardsize", "Insider", "Mgshder"]
    ])
    pca_var.to_csv(os.path.join(output_dir, "pca_variable_metrics.csv"), index=False)

    pd.DataFrame([main_regression["summary"]]).to_csv(
        os.path.join(output_dir, "main_mediation_summary.csv"),
        index=False
    )

    grouped_mediation_df.to_csv(
        os.path.join(output_dir, "grouped_mediation_results.csv"),
        index=False
    )


# ============================================================
# 主流程
# ============================================================

def main():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "processed_data")
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 1-5. 加载数据并构造分析变量
    df, pca_diagnostics = build_analysis_dataset(data_dir)

    # 6. 回归分析 (含 Sobel 中介效应检验)
    main_regression = run_regressions(df)

    # 7. 异质性分析（分产权）
    heterogeneity_analysis(df)

    # 8. 机制检验（管制/非管制、央企/地方国企）
    mechanism_analysis(df)

    # 9. 分组中介效应
    grouped_mediation_df = grouped_mediation_analysis(df)

    # 10. 相关性分析
    correlation_analysis(df)

    # 11. VIF 多重共线性检验
    vif_diagnostics(df)

    # 12. 稳健性检验
    robustness_checks(df)

    # 保存最终数据集
    output_file = os.path.join(output_dir, "regression_dataset.csv")
    df.to_csv(output_file, index=False)
    save_structured_outputs(output_dir, pca_diagnostics, main_regression, grouped_mediation_df)
    print(f"\n最终分析数据集已保存至: {output_file}")
    print(f"总行数: {len(df)}, 总列数: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
