"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 实证分析脚本
=============================================================================

参考论文：刘剑民(2019)《政府补助、管理层权力与国有企业高管超额薪酬》
开题报告：基于数据挖掘的上市公司财政补贴与高管超额薪酬研究

研究模型：
  模型1（期望薪酬）: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ε
  模型2（超额薪酬）: Overpay = lnCEOpay_actual - lnCEOpay_expected
  模型3（基准回归）: Overpay = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + α₅·Zone + ΣIndustry + ΣYear + ε
  模型4（中介效应）: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + α₃·Roa + α₄·Lever + α₅·Top1 + α₆·Zone + ΣIndustry + ΣYear + ε
  模型5（权力回归）: Power  = α₀ + α₁·lnSubsidy + α₂·Roa + α₃·Lever + α₄·Top1 + α₅·Zone + ΣIndustry + ΣYear + ε
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

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
    df_fin = pd.read_csv(os.path.join(data_dir, "营业收入+净利润+总资产+无形资产+行业变量.csv"))
    # id 是数字编号，需要找到真正的股票代码
    # 这里 id 实际上不是股票代码，而是公司编号。需要检查
    # 从其他文件（如所在地）可以看到 Symbol 是 000001 格式
    # 营业收入文件中 id=1 对应 Symbol=000001
    df_fin.rename(columns={"id": "Symbol", "OperatingEvenue": "Revenue"}, inplace=True)
    df_fin["Symbol"] = df_fin["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_fin["Year"] = df_fin["year"]
    df_fin = df_fin[["Symbol", "Year", "IndustryCode1", "IndustryName1",
                      "TotalAssets", "IntangibleAsset", "NetProfit", "Revenue"]]
    print(f"  {len(df_fin)} 行")

    # --- 2. 两职合一 + 管理层持股 + 董事会规模 + 独董比例 ---
    print("[2/8] 加载公司治理数据...")
    df_gov = pd.read_csv(os.path.join(data_dir, "两职合一+管理层持股比例+董事会规模+独立董事占比.csv"))
    df_gov["Symbol"] = df_gov["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_gov["Year"] = pd.to_datetime(df_gov["Enddate"]).dt.year
    # 去重，保留每个公司每年最后一条记录
    df_gov = df_gov.sort_values("Enddate").groupby(["Symbol", "Year"]).last().reset_index()
    df_gov = df_gov[["Symbol", "Year", "ConcurrentPosition", "Mngmhldn",
                      "Boardsize", "IndDirectorRatio"]]
    print(f"  {len(df_gov)} 行")

    # --- 3. 总负债 ---
    print("[3/8] 加载负债数据...")
    df_debt = pd.read_csv(os.path.join(data_dir, "总负债.csv"))
    df_debt["Symbol"] = df_debt["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_debt["Year"] = pd.to_datetime(df_debt["EndDate"]).dt.year
    df_debt = df_debt.groupby(["Symbol", "Year"])["TotalLiability"].last().reset_index()
    print(f"  {len(df_debt)} 行")

    # --- 4. 高管前三名薪酬总合 ---
    print("[4/8] 加载薪酬数据...")
    df_pay = pd.read_csv(os.path.join(data_dir, "高管前三名薪酬总合.csv"))
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
    df_loc = pd.read_csv(os.path.join(data_dir, "所在地+公司性质.csv"))
    df_loc["Symbol"] = df_loc["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_loc["Year"] = pd.to_datetime(df_loc["EndDate"]).dt.year
    df_loc = df_loc.sort_values("EndDate").groupby(["Symbol", "Year"]).last().reset_index()
    df_loc = df_loc[["Symbol", "Year", "Ownership", "City"]]
    print(f"  {len(df_loc)} 行")

    # --- 6. 第一大股东持股比例 + 实际控制人股权性质 ---
    print("[6/8] 加载股东数据...")
    df_holder = pd.read_csv(os.path.join(data_dir, "第一大股东持股比例+实际控制人股权性质.csv"))
    df_holder["Symbol"] = df_holder["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_holder["Year"] = pd.to_datetime(df_holder["EndDate"]).dt.year
    df_holder = df_holder.sort_values("EndDate").groupby(["Symbol", "Year"]).last().reset_index()
    df_holder = df_holder[["Symbol", "Year", "LargestHolderRate", "ActualControllerNatureID"]]
    print(f"  {len(df_holder)} 行")

    # --- 7. 总经理任期年限 ---
    print("[7/8] 加载总经理任期数据...")
    df_tenure = pd.read_csv(os.path.join(data_dir, "总经理任期年限.csv"))
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
    df_sub = pd.read_csv(os.path.join(data_dir, "政府补助.csv"))
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
    df["lnCEOpay"] = np.log(df["Top3Salary"])

    # 2. 解释变量：政府补助对数
    print("2. 构造 lnSubsidy = ln(政府补助)")
    # 政府补助为0或负的处理：取正值后取对数
    df["lnSubsidy"] = np.log(df["SubsidyAmount"].clip(lower=1))

    # 3. 控制变量
    print("3. 构造控制变量...")
    # Roa = 净利润 / 总资产
    df["Roa"] = df["NetProfit"] / df["TotalAssets"]
    # lnSale = ln(营业收入)
    df["lnSale"] = np.log(df["Revenue"].clip(lower=1))
    # IA = 无形资产 / 总资产
    df["IA"] = df["IntangibleAsset"] / df["TotalAssets"]
    # Lever = 总负债 / 总资产
    df["Lever"] = df["TotalLiability"] / df["TotalAssets"]
    # Top1 = 第一大股东持股比例 (已有)
    df["Top1"] = df["LargestHolderRate"]

    # 4. Zone 区域虚拟变量：东部=0, 中西部=1
    print("4. 构造区域变量 Zone...")
    eastern_cities = [
        "北京市", "天津市", "上海市", "重庆市",
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
    print(f"   行业大类分布:\n{df['IndustrySector'].value_counts().to_string()}")

    # 6. 管理层权力指标
    print("6. 构造管理层权力各分指标...")
    # Dual (两职合一): ConcurrentPosition (0=否, 1=是? 需要检查原始值)
    df["Dual"] = df["ConcurrentPosition"]
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
    print(f"   企业性质分布:\n{df['OwnerType'].value_counts(dropna=False).to_string()}")

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

    # 对连续变量进行 1%/99% Winsorize 缩尾处理
    print("\n对连续变量进行 1%-99% Winsorize 缩尾处理...")
    continuous_vars = ["lnCEOpay", "lnSubsidy", "lnSale", "Roa", "IA", "Lever", "Top1"]
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
    模型1: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ε
    模型2: Overpay = lnCEOpay_actual - lnCEOpay_expected (即残差)
    """
    print("\n" + "=" * 70)
    print("第四部分：期望薪酬模型与超额薪酬 (Overpay)")
    print("=" * 70)

    # 准备自变量
    model1_vars = ["lnSale", "Roa", "IA", "Zone"]
    df_model1 = df.dropna(subset=model1_vars + ["lnCEOpay", "IndustrySector"]).copy()

    # 构建 X: 核心变量 + 行业虚拟变量（ΣIndustry）
    X = df_model1[model1_vars].copy()
    ind_dummies = get_industry_dummies(df_model1)
    X = pd.concat([X, ind_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_model1["lnCEOpay"]

    print(f"\n模型1 样本量: {len(df_model1)}")
    print(f"核心自变量: {model1_vars}")
    print(f"行业虚拟变量: {len(ind_dummies.columns)} 个")

    # OLS 回归
    model1 = sm.OLS(y, X).fit()
    print("\n" + "-" * 50)
    print("模型1 回归结果 (期望薪酬模型) — 仅展示核心变量")
    print("-" * 50)
    # 只打印核心变量系数，行业dummy太多不全部展示
    core_result = model1.summary2().tables[1]
    core_rows = ["const"] + model1_vars
    core_result_filtered = core_result.loc[core_result.index.isin(core_rows)]
    print(core_result_filtered.to_string())
    print(f"行业固定效应: 已控制 ({len(ind_dummies.columns)} 个行业虚拟变量)")
    print(f"\nR² = {model1.rsquared:.4f}, Adj-R² = {model1.rsquared_adj:.4f}")

    # 残差 = 超额薪酬
    df_model1["Overpay"] = model1.resid

    # 将 Overpay 合并回主数据
    df = df.merge(df_model1[["Symbol", "Year", "Overpay"]],
                  on=["Symbol", "Year"], how="left")

    print(f"\n超额薪酬 (Overpay) 描述:")
    print(df["Overpay"].describe().to_string())

    return df


# ============================================================
# 第五部分：管理层权力 Power (PCA主成分分析)
# ============================================================

def compute_power(df):
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

    # 对有完整权力指标的样本进行 PCA
    df_pca = df.dropna(subset=power_vars).copy()
    print(f"\n可用于 PCA 的样本量: {len(df_pca)}")

    if len(df_pca) < 100:
        print("  WARNING: 样本量过少，跳过 PCA。")
        df["Power"] = np.nan
        return df

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca[power_vars])

    # PCA 提取第一主成分
    pca = PCA(n_components=1)
    power_scores = pca.fit_transform(X_scaled)

    print(f"\n  第一主成分解释方差比: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  各变量载荷:")
    for var, loading in zip(power_vars, pca.components_[0]):
        print(f"    {var:15s}: {loading:.4f}")

    df_pca["Power"] = power_scores.flatten()

    # 合并回主数据
    df = df.merge(df_pca[["Symbol", "Year", "Power"]],
                  on=["Symbol", "Year"], how="left")

    print(f"\n管理层权力 (Power) 描述:")
    print(df["Power"].describe().to_string())

    return df


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


def run_regressions(df):
    """
    模型3: Overpay = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    模型4: Overpay = α₀ + α₁·lnSubsidy + α₂·Power + 控制 + ΣIndustry + ΣYear + ε
    模型5: Power   = α₀ + α₁·lnSubsidy + 控制 + ΣIndustry + ΣYear + ε
    """
    print("\n" + "=" * 70)
    print("第六部分：回归分析（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]

    # ---- 模型3：基准回归 ----
    print("\n" + "-" * 50)
    print("模型3: Overpay = f(lnSubsidy, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)

    core3 = ["lnSubsidy"] + control_vars
    df3 = df.dropna(subset=core3 + ["Overpay", "IndustrySector"]).copy()
    X3, n_ind3, n_year3 = _build_fe_matrix(df3, core3)
    y3 = df3["Overpay"]
    groups3 = df3["Symbol"]
    model3 = _fit_with_cluster_se(y3, X3, groups3)
    n_cl3 = groups3.nunique()
    _print_core_results(model3, core3, n_ind3, n_year3, len(df3), n_cl3)

    # ---- 模型4：加入管理层权力（中介效应） ----
    print("\n" + "-" * 50)
    print("模型4: Overpay = f(lnSubsidy, Power, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)

    core4 = ["lnSubsidy", "Power"] + control_vars
    df4 = df.dropna(subset=core4 + ["Overpay", "IndustrySector"]).copy()
    X4, n_ind4, n_year4 = _build_fe_matrix(df4, core4)
    y4 = df4["Overpay"]
    groups4 = df4["Symbol"]
    model4 = _fit_with_cluster_se(y4, X4, groups4)
    n_cl4 = groups4.nunique()
    _print_core_results(model4, core4, n_ind4, n_year4, len(df4), n_cl4)

    # ---- 模型5：政府补助对管理层权力 ----
    print("\n" + "-" * 50)
    print("模型5: Power = f(lnSubsidy, Controls, ΣIndustry, ΣYear)")
    print("-" * 50)

    core5 = ["lnSubsidy"] + control_vars
    df5 = df.dropna(subset=core5 + ["Power", "IndustrySector"]).copy()
    X5, n_ind5, n_year5 = _build_fe_matrix(df5, core5)
    y5 = df5["Power"]
    groups5 = df5["Symbol"]
    model5 = _fit_with_cluster_se(y5, X5, groups5)
    n_cl5 = groups5.nunique()
    _print_core_results(model5, core5, n_ind5, n_year5, len(df5), n_cl5)

    # ---- 中介效应判断 (Baron & Kenny + Sobel Test) ----
    print("\n" + "=" * 50)
    print("中介效应检验")
    print("=" * 50)
    coef3_sub = model3.params.get("lnSubsidy", np.nan)
    pval3_sub = model3.pvalues.get("lnSubsidy", np.nan)
    coef4_sub = model4.params.get("lnSubsidy", np.nan)
    pval4_sub = model4.pvalues.get("lnSubsidy", np.nan)
    coef4_pow = model4.params.get("Power", np.nan)
    pval4_pow = model4.pvalues.get("Power", np.nan)
    se4_pow = model4.bse.get("Power", np.nan)
    coef5_sub = model5.params.get("lnSubsidy", np.nan)
    pval5_sub = model5.pvalues.get("lnSubsidy", np.nan)
    se5_sub = model5.bse.get("lnSubsidy", np.nan)

    print("\n  逐步回归法 (Baron & Kenny, 1986):")
    print(f"    第一步 (模型3) lnSubsidy→Overpay:  α₁={coef3_sub:.4f} (p={pval3_sub:.4f})")
    print(f"    第二步 (模型5) lnSubsidy→Power:    c₁={coef5_sub:.4f} (p={pval5_sub:.4f})")
    print(f"    第三步 (模型4) lnSubsidy→Overpay:  α₁'={coef4_sub:.4f} (p={pval4_sub:.4f})")
    print(f"    第三步 (模型4) Power→Overpay:      b₁={coef4_pow:.4f} (p={pval4_pow:.4f})")

    # Sobel 检验
    # Z_sobel = (c₁ * b₁) / sqrt(b₁² * se_c₁² + c₁² * se_b₁²)
    from scipy import stats
    a = coef5_sub   # c₁: Subsidy → Power
    b = coef4_pow   # b₁: Power → Overpay
    se_a = se5_sub
    se_b = se4_pow
    sobel_z = (a * b) / np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    indirect_effect = a * b
    total_effect = coef3_sub
    mediation_ratio = indirect_effect / total_effect * 100 if total_effect != 0 else np.nan

    print(f"\n  Sobel 检验:")
    print(f"    间接效应 (a×b) = {indirect_effect:.6f}")
    print(f"    Sobel Z = {sobel_z:.4f}")
    print(f"    Sobel p = {sobel_p:.4f}")
    print(f"    中介效应占总效应比例 = {mediation_ratio:.2f}%")

    if sobel_p < 0.05:
        if pval4_sub > 0.1:
            print("\n  结论：管理层权力具有【完全中介效应】(Sobel检验显著, α₁'不显著)")
        else:
            print("\n  结论：管理层权力具有【部分中介效应】(Sobel检验显著, α₁'仍显著)")
    else:
        print("\n  结论：Sobel检验不显著，中介效应不成立")

    return model3, model4, model5


# ============================================================
# 第七部分：异质性分析（分产权性质）
# ============================================================

def heterogeneity_analysis(df):
    """分企业性质（国有 vs 非国有）进行回归，含行业和年份固定效应"""
    print("\n" + "=" * 70)
    print("第七部分：异质性分析 — 分产权性质（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    core_vars_no_power = ["lnSubsidy"] + control_vars
    core_vars_with_power = ["lnSubsidy", "Power"] + control_vars

    for label, mask_val in [("国有企业", 1), ("非国有企业", 0)]:
        print(f"\n{'='*40}")
        print(f"子样本: {label}")
        print(f"{'='*40}")

        df_sub = df[df["IsSOE"] == mask_val].copy()
        df_sub = df_sub.dropna(subset=core_vars_with_power + ["Overpay", "IndustrySector"])

        if len(df_sub) < 50:
            print(f"  样本量不足 ({len(df_sub)})，跳过。")
            continue

        # 模型3 (无 Power) + 行业/年份FE
        X3, n_ind3, n_year3 = _build_fe_matrix(df_sub, core_vars_no_power)
        y3 = df_sub["Overpay"]
        m3 = _fit_with_cluster_se(y3, X3, df_sub["Symbol"])

        # 模型4 (有 Power) + 行业/年份FE
        X4, n_ind4, n_year4 = _build_fe_matrix(df_sub, core_vars_with_power)
        y4 = df_sub["Overpay"]
        m4 = _fit_with_cluster_se(y4, X4, df_sub["Symbol"])

        print(f"\n  样本量: {len(df_sub)}")
        print(f"  行业FE: {n_ind3} 个 | 年份FE: {n_year3} 个")
        print(f"\n  模型3 (无Power):")
        print(f"    lnSubsidy: {m3.params['lnSubsidy']:.4f} (p={m3.pvalues['lnSubsidy']:.4f})")
        print(f"    R² = {m3.rsquared:.4f}")
        print(f"\n  模型4 (有Power):")
        print(f"    lnSubsidy: {m4.params['lnSubsidy']:.4f} (p={m4.pvalues['lnSubsidy']:.4f})")
        print(f"    Power:     {m4.params['Power']:.4f} (p={m4.pvalues['Power']:.4f})")
        print(f"    R² = {m4.rsquared:.4f}")


# ============================================================
# 第八部分：相关性分析
# ============================================================

def correlation_analysis(df):
    """Pearson 相关系数矩阵"""
    print("\n" + "=" * 70)
    print("第八部分：主要变量相关系数矩阵")
    print("=" * 70)

    corr_vars = ["Overpay", "lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone", "Industry"]
    available = [v for v in corr_vars if v in df.columns]
    df_corr = df[available].dropna()

    corr_matrix = df_corr.corr()
    print(f"\n样本量: {len(df_corr)}")
    print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))


# ============================================================
# VIF 多重共线性诊断
# ============================================================

def vif_diagnostics(df):
    """方差膨胀因子 (VIF) 诊断多重共线性"""
    print("\n" + "=" * 70)
    print("VIF 多重共线性诊断")
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
# 稳健性检验
# ============================================================

def robustness_checks(df):
    """稳健性检验：(1)替换被解释变量 (2)子样本检验 (3)滞后变量"""
    print("\n" + "=" * 70)
    print("稳健性检验")
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


# ============================================================
# 主流程
# ============================================================

def main():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "processed_data")
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    df = load_and_clean_data(data_dir)

    # 2. 构造变量
    df = construct_variables(df)

    # 3. 筛选与描述性统计
    df = filter_and_describe(df)

    # 4. 计算超额薪酬
    df = compute_overpay(df)

    # 5. 计算管理层权力
    df = compute_power(df)

    # 6. VIF 多重共线性检验
    vif_diagnostics(df)

    # 7. 相关性分析
    correlation_analysis(df)

    # 8. 回归分析 (含 Sobel 中介效应检验)
    model3, model4, model5 = run_regressions(df)

    # 9. 异质性分析
    heterogeneity_analysis(df)

    # 10. 稳健性检验
    robustness_checks(df)

    # 保存最终数据集
    output_file = os.path.join(output_dir, "regression_dataset.csv")
    df.to_csv(output_file, index=False)
    print(f"\n最终分析数据集已保存至: {output_file}")
    print(f"总行数: {len(df)}, 总列数: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
