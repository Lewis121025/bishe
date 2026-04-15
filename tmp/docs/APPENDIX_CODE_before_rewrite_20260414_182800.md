# 附录 Python 代码实现

本附录列出论文核心实证分析的 Python 代码实现思路与关键函数。代码采用 statsmodels（固定效应回归）、pandas（数据处理）、scikit-learn 与 xgboost（机器学习）等开源库。

---

## 附录目录

| 序号 | 模块 | 说明 |
|------|------|------|
| A.1 | 数据预处理 | 原始数据格式转换与合并 |
| A.2 | 数据加载与清洗 | 多源数据合并与样本筛选 |
| A.3 | 变量构造 | 超额薪酬、管理层权力等变量的计算方法 |
| A.4 | 期望薪酬模型 | OLS 基准模型与超额薪酬残差提取 |
| A.5 | 管理层权力 PCA | PCA 综合指数的构造与标准化 |
| A.6 | 主回归与中介检验 | 固定效应回归、中介效应分解与 Bootstrap 检验 |
| A.7 | 机器学习补充验证 | 随机森林、SHAP、Lasso、XGBoost 的实现 |
| A.8 | 稳健性检验 | 替代变量、滞后结构、样本筛选、自回归等检验 |
| A.9 | 异质性分析 | 按所有制性质、地区、规模维度分层回归 |

---

## A.1 数据预处理：原始数据格式转换

**说明：** 将原始 Stata `.dta` 格式数据转换为 CSV，支持多文件拼接（如"政府补助（两个文件上下拼接）-1.dta"与"-2.dta"）。

```python
import os
import pandas as pd
import glob

def process_data(input_dir, output_dir):
    """转换 .dta 文件为 .csv，处理多文件拼接情况"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    dta_files = glob.glob(os.path.join(input_dir, "*.dta"))
    print(f"Found {len(dta_files)} .dta files.")

    processed_files = set()

    for file_path in dta_files:
        filename = os.path.basename(file_path)
        
        if filename in processed_files:
            continue

        base_name, ext = os.path.splitext(filename)
        
        # 处理多文件拼接情况
        if "两个文件上下拼接" in filename:
            prefix = filename.split("（")[0]
            parts = [f for f in dta_files if prefix in os.path.basename(f) 
                     and "（两个文件上下拼接）" in os.path.basename(f)]
            parts.sort()  # 确保 -1 在 -2 之前
            
            if not parts:
                print(f"Warning: Could not find matching parts for {filename}")
                continue
                
            print(f"Processing split files for {prefix}: {[os.path.basename(p) for p in parts]}")
            
            dfs = []
            for part in parts:
                try:
                    df = pd.read_stata(part)
                    dfs.append(df)
                    processed_files.add(os.path.basename(part))
                except Exception as e:
                    print(f"Error reading {part}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                output_name = f"{prefix}.csv"
                output_path = os.path.join(output_dir, output_name)
                combined_df.to_csv(output_path, index=False)
                print(f"Saved combined file to {output_path}")

        else:
            # 单文件处理
            print(f"Processing single file: {filename}")
            try:
                df = pd.read_stata(file_path)
                output_name = base_name + ".csv"
                output_path = os.path.join(output_dir, output_name)
                df.to_csv(output_path, index=False)
                print(f"Saved to {output_path}")
                processed_files.add(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    data_dir = os.path.join(ROOT_DIR, "data")
    processed_dir = os.path.join(ROOT_DIR, "processed_data")
    
    print("Starting conversion...")
    try:
        process_data(data_dir, processed_dir)
        print("Conversion complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
```

---

## A.2 数据加载与清洗

**说明：** 从 7 个 CSV 文件加载数据（财务、治理、负债、薪酬、地区、股东结构、政府补助），按公司-年度合并成面板数据。样本筛选流程：金融行业 → 特殊处理企业 → 变量完整性检验。

```python
def load_and_clean_data():

    # 1. 财务数据
    print("\n[1/8] 加载财务数据...")
    fin_path = os.path.join(data_dir, "营业收入+净利润+总资产+无形资产+行业变量.csv")
    df_fin = pd.read_csv(fin_path)
    df_fin.rename(columns={"id": "Symbol", "OperatingEvenue": "Revenue"}, inplace=True)
    df_fin["Symbol"] = df_fin["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_fin["Year"] = df_fin["year"]
    print(f"  {len(df_fin)} 行")

    # 2. 治理数据
    print("[2/8] 加载公司治理数据...")
    gov_path = os.path.join(data_dir, "两职合一+管理层持股比例+董事会规模+独立董事占比.csv")
    df_gov = pd.read_csv(gov_path)
    df_gov["Symbol"] = df_gov["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_gov["Year"] = pd.to_datetime(df_gov["Enddate"]).dt.year
    df_gov = df_gov.sort_values(["Symbol", "Year"]).drop_duplicates(subset=["Symbol", "Year"], keep="last")
    print(f"  {len(df_gov)} 行")

    # 3. 负债数据
    print("[3/8] 加载负债数据...")
    debt_path = os.path.join(data_dir, "总负债.csv")
    df_debt = pd.read_csv(debt_path)
    df_debt["Symbol"] = df_debt["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_debt["Year"] = pd.to_datetime(df_debt["EndDate"]).dt.year
    df_debt = df_debt[["Symbol", "Year", "TotalLiability"]]
    print(f"  {len(df_debt)} 行")

    # 4. 薪酬数据
    print("[4/8] 加载高管薪酬数据...")
    pay_path = os.path.join(data_dir, "高管前三名薪酬总合.csv")
    df_pay = pd.read_csv(pay_path)
    df_pay["Symbol"] = df_pay["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_pay["Year"] = pd.to_datetime(df_pay["Year"]).dt.year
    print(f"  {len(df_pay)} 行")

    # 5-7. 其他数据类似加载...
    
    # 合并所有数据
    print("\n合并数据源...")
    merged = df_fin.copy()
    for name, df_right in [("治理", df_gov), ("负债", df_debt), ("薪酬", df_pay)]:
        before = len(merged)
        merged = pd.merge(merged, df_right, on=["Symbol", "Year"], how="left")
        print(f"  + {name}: {before} → {len(merged)} 行")

    print(f"\n合并后总行数: {len(merged)}")
    return merged
```

---

## A.3 变量构造

**说明：** 根据论文方法构造主要研究变量：

- **被解释变量**：$\ln Salary = \ln(\text{高管前三名年薪总额})$
- **解释变量**：财政补贴对数。基准口径为 $\ln(1+\text{Subsidy})$，主回归口径为仅对正补助取对数
- **控制变量**：ROA、Lever、Top1、lnSale、IA、Zone 等
- **滞后变量**：滞后一期补贴强度、领先变量用于动态检验
- **工具变量**：基于同城市、同行业、留一法构造的工具变量

```python
def construct_variables(df):
    """根据论文方法构造所有研究变量"""
    print("\n" + "=" * 70)
    print("第二部分：变量构造")
    print("=" * 70)

    # 1. 被解释变量：高管薪酬对数
    print("\n1. 构造 lnCEOpay = ln(高管前三名薪酬总和)")
    positive_pay = df["Top3Salary"].where(df["Top3Salary"] > 0)
    df["lnCEOpay"] = np.log(positive_pay)

    # 2. 解释变量：政府补助对数
    print("2. 构造政府补助对数变量...")
    df["SubsidyAmount"] = df["SubsidyAmount"].fillna(0)
    positive_subsidy = df["SubsidyAmount"].where(df["SubsidyAmount"] > 0)
    # 基准口径：ln(1 + Subsidy)
    df["lnSubsidy"] = np.log1p(df["SubsidyAmount"].clip(lower=0))
    # 替代口径：仅对正补助取对数
    df["lnSubsidy_pos"] = np.log(positive_subsidy)

    # 3. 控制变量
    print("3. 构造控制变量...")
    df["Roa"] = df["NetProfit"] / df["TotalAssets"]
    positive_revenue = df["Revenue"].where(df["Revenue"] > 0)
    df["lnSale"] = np.log(positive_revenue)
    df["IA"] = df["IntangibleAsset"] / df["TotalAssets"]
    df["Lever"] = df["TotalLiability"] / df["TotalAssets"]
    df["Top1"] = df["LargestHolderRate"]

    # 4. 地区变量
    print("4. 构造区域变量...")
    df["Zone"] = df.apply(
        lambda row: _classify_zone_from_location(
            row.get("City"),
            row.get("ADDRESS_REGISTER"),
            row.get("ADDRESS_OFFICE"),
        ),
        axis=1,
    )

    # 5. 企业性质
    print("5. 构造企业性质...")
    df["OwnerType"] = df["Ownership"].apply(classify_ownership)
    df["IsPrivate"] = np.where(df["Ownership"] == "私营企业", 1, 0)

    # 6. 管理层权力底层指标
    print("6. 构造管理层权力底层指标...")
    # Dual: 两职合一（CEO=董事长）
    df["Dual"] = ((df["BothChief"] == 1) | (df["DualjointTitle"] == 1)).astype(float)
    # Boardsize: 董事会规模
    df["Boardsize"] = df["DirectorNumber"]
    # Insider: 内部董事占比
    df["Insider"] = df["InternalDirectorProportion"]
    # Mgshder: 管理层持股比例
    df["Mgshder"] = df["ManagementHoldingPercentage"]

    # 7. 滞后变量
    print("7. 构造滞后变量...")
    df = df.sort_values(["Symbol", "Year"])
    df["lnSubsidy_l1"] = df.groupby("Symbol")["lnSubsidy"].shift(1)
    df["lnSubsidy_pos_l1"] = df.groupby("Symbol")["lnSubsidy_pos"].shift(1)
    df["lnSubsidy_f1"] = df.groupby("Symbol")["lnSubsidy"].shift(-1)

    # 8. 工具变量（留一法）
    print("8. 构造工具变量...")
    df = _build_leave_one_out_mean(df, ["City", "Year"], "lnSubsidy", "IV_city_year_l1")
    df = _build_leave_one_out_mean(df, ["IndustrySector", "Year"], "lnSubsidy", "IV_industry_year_l1")

    print("\n变量构造完成")
    return df


def _build_leave_one_out_mean(df, group_cols, value_col, output_col):
    """按组构造留一法均值工具变量"""
    group_sum = df.groupby(group_cols)[value_col].transform(lambda x: x.sum(skipna=True))
    group_count = df.groupby(group_cols)[value_col].transform(lambda x: x.count())
    df[output_col] = np.where(
        group_count > 1,
        (group_sum - df[value_col].fillna(0)) / (group_count - 1),
        np.nan,
    )
    return df
```

---

## A.4 期望薪酬模型与超额薪酬

**说明：** 使用 OLS 估计基准薪酬模型，残差定义为超额薪酬 (Overpay)。

**期望薪酬模型（表4-4）：**

$$\ln Salary_i = \alpha_0 + \alpha_1 \ln Sale_i + \alpha_2 ROA_i + \alpha_3 IA_i + \alpha_4 Zone_i + \sum_j Industry_j + \sum_t Year_t + \varepsilon_i$$

其中 $\varepsilon_i$ 的加总即为超额薪酬变量。

```python
def compute_overpay(df):
    """
    模型1: lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ΣYear + ε
    超额薪酬 Overpay = ε（模型1残差）
    """
    print("\n" + "=" * 70)
    print("第四部分：期望薪酬模型与超额薪酬 (Overpay)")
    print("=" * 70)

    fit_result = _fit_expectation_salary_model(df)
    model1 = fit_result["model"]
    df_model1 = fit_result["df_model1"]

    print(f"\n模型1 样本量: {len(df_model1)}")
    print(f"R² = {model1.rsquared:.4f}, Adj-R² = {model1.rsquared_adj:.4f}")

    print("\n" + "-" * 50)
    print("模型1 核心变量系数（表4-4）")
    print("-" * 50)
    # 仅显示核心变量，行业/年份 dummy 太多不全部展示
    core_result = model1.summary2().tables[1]
    core_rows = ["const", "lnSale", "Roa", "IA", "Zone"]
    core_result_filtered = core_result.loc[core_result.index.isin(core_rows)]
    print(core_result_filtered.to_string())

    # 残差 = 超额薪酬
    df_model1["Overpay"] = model1.resid
    print("\nOverpay 直接使用模型1残差，不进行二次缩尾。")
    print(f"超额薪酬描述统计:")
    print(df_model1["Overpay"].describe().to_string())

    # 合并回主数据
    df = df.merge(df_model1[["Symbol", "Year", "Overpay"]],
                  on=["Symbol", "Year"], how="left")

    return df


def _fit_expectation_salary_model(df):
    """拟合期望薪酬模型"""
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
```

---

## A.5 管理层权力 PCA 综合指数

**说明：** 使用四个底层治理指标构造管理层权力综合指数。

**底层指标：**
- Dual：两职合一（CEO兼任董事长）
- Boardsize：董事会规模
- Insider：内部董事占比
- Mgshder：管理层持股比例

**构造流程：**
1. 对四个指标做标准化处理
2. KMO 与 Bartlett 球形检验验证因子分析适用性
3. 提取前两个主成分并统一方向
4. 按各主成分的方差贡献率加权合成综合得分
5. 再次标准化最终综合指数

```python
def compute_power_index(df):
    """
    构造管理层权力综合指数 (PCA 口径)
    底层指标：
      - Dual: 两职合一 (CEO=董事长)
      - Boardsize: 董事会规模
      - Insider: 内部董事占比
      - Mgshder: 管理层持股比例
    """
    print("\n" + "=" * 70)
    print("第五部分：管理层权力 Power（PCA 综合口径）")
    print("=" * 70)

    power_vars = ["Dual", "Boardsize", "Insider", "Mgshder"]
    df_complete = df.dropna(subset=power_vars).copy()
    
    print(f"\nPCA 样本量: {len(df_complete)}")
    print(f"底层指标: {power_vars}")

    # 1. 标准化
    print("\n1. 标准化底层指标...")
    standardized = _standardize_power_inputs(df_complete, power_vars)

    # 2. 计算 KMO 和 Bartlett 球形检验
    print("2. 计算 KMO 和 Bartlett 球形检验...")
    diagnostics = _calc_shared_factor_diagnostics(df_complete, power_vars)
    print(f"   KMO = {diagnostics['kmo_overall']:.4f} (>0.5 表示适合因子分析)")
    print(f"   Bartlett p = {diagnostics['bartlett_p']:.6f} (p<0.05 表示变量相关)")
    print(f"   各变量 KMO:")
    for var, kmo in diagnostics['kmo_per_var'].items():
        print(f"     {var}: {kmo:.4f}")

    # 3. PCA
    print("\n3. 执行 PCA...")
    pca_result = _compute_pca_power(standardized, power_vars)
    power_scores = pca_result["scores"]
    
    print(f"   第一主成分解释方差比: {pca_result['explained_variance_ratio_pc1']:.4f}")
    print(f"   第二主成分解释方差比: {pca_result['explained_variance_ratio_pc2']:.4f}")
    print(f"   前两个主成分累计解释方差: {pca_result['cum_explained_variance_pc2']:.4f}")
    
    print(f"\n   综合指数成分权重:")
    for pc, weight in pca_result['component_weights'].items():
        print(f"     {pc}: {weight:.4f}")
    
    print(f"\n   各底层指标在综合指数中的载荷:")
    for var, loading in pca_result['loadings'].items():
        print(f"     {var}: {loading:.4f}")

    # 合并回主数据
    df_complete["Power"] = power_scores
    df = df.merge(df_complete[["Symbol", "Year", "Power"]],
                  on=["Symbol", "Year"], how="left")

    return df, diagnostics, pca_result


def _standardize_power_inputs(df_complete, power_vars):
    """标准化管理层权力分指标"""
    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(df_complete[power_vars]),
        columns=power_vars,
        index=df_complete.index,
    )
    return standardized


def _compute_pca_power(standardized_df, power_vars):
    """PCA 综合指数：提取前两个主成分并按方差贡献率加权"""
    n_components = min(2, len(power_vars))
    pca = PCA(n_components=n_components)
    raw_scores = pca.fit_transform(standardized_df)

    # 统一方向
    oriented_scores = raw_scores.copy()
    loadings_matrix = pca.components_.copy()
    for idx in range(n_components):
        score_vec, sign = _orient_power_scores(raw_scores[:, idx], standardized_df)
        oriented_scores[:, idx] = score_vec
        loadings_matrix[idx, :] = loadings_matrix[idx, :] * sign

    # 按方差贡献率加权合成
    component_weights = pca.explained_variance_ratio_ / np.sum(pca.explained_variance_ratio_)
    composite_raw = oriented_scores @ component_weights
    
    # 标准化综合得分
    composite_std = np.std(composite_raw, ddof=0)
    if composite_std < 1e-12:
        composite_scores = np.zeros_like(composite_raw)
    else:
        composite_scores = (composite_raw - composite_raw.mean()) / composite_std
    
    composite_loadings = loadings_matrix.T @ component_weights
    
    return {
        "scores": composite_scores,
        "explained_variance_ratio_pc1": pca.explained_variance_ratio_[0] if n_components > 0 else 0,
        "explained_variance_ratio_pc2": pca.explained_variance_ratio_[1] if n_components > 1 else 0,
        "cum_explained_variance_pc2": np.sum(pca.explained_variance_ratio_[:n_components]),
        "component_weights": {f"PC{i+1}": w for i, w in enumerate(component_weights)},
        "loadings": {var: float(composite_loadings[j]) for j, var in enumerate(power_vars)},
    }


def _orient_power_scores(scores, data_df):
    """对PCA得分进行方向统一"""
    mean_score = np.mean(scores)
    data_mean = np.mean(data_df, axis=0)
    proj_sum = np.sum(scores[:len(data_df)] * (data_mean > 0))
    sign = 1 if proj_sum >= 0 else -1
    return scores * sign, sign
```

---

## A.6 主回归与中介检验

**说明：** 使用固定效应 OLS 估计补贴强度与超额薪酬的关系。聚焦"已获补贴企业内部的补贴强度差异"。然后使用 Sobel 检验和 Bootstrap 方法检验管理层权力的中介效应。

### 主回归模型（表 4-5）

**模型2：** $Overpay_{i,t} = \alpha_0 + \alpha_1 \ln Subsidy^+_{i,t-1} + \alpha_2 ROA_{i,t} + \alpha_3 Lever_{i,t} + \alpha_4 Top1_{i,t} + \mu_i + \lambda_t + \varepsilon_{i,t}$

其中：
- $Overpay_{i,t}$：高管超额薪酬
- $\ln Subsidy^+_{i,t-1}$：滞后一期正补助对数（仅对正补助取对数）
- $\mu_i$：公司固定效应
- $\lambda_t$：年份固定效应

### 中介效应分解（表 4-6 & 4-7）

**路径a：** $Power_{i,t} = \gamma_0 + \gamma_1 \ln Subsidy^+_{i,t-1} + \sum \text{controls} + \mu_i + \lambda_t + \varepsilon_{i,t}$

**路径b 与直接效应：** $Overpay_{i,t} = \delta_0 + \delta_1 \ln Subsidy^+_{i,t-1} + \delta_2 Power_{i,t} + \sum \text{controls} + \mu_i + \lambda_t + \varepsilon_{i,t}$

**非直接效应：** $IE = \gamma_1 \times \delta_2$

```python
def run_main_regressions(df):
    """
    主回归：财政补贴强度与高管超额薪酬的关联
    样本限定：上一年已获得正补贴的公司年度
    """
    print("\n" + "=" * 70)
    print("第六部分：主回归分析")
    print("=" * 70)

    # 获取基准样本
    df_main = _get_main_regression_sample(df)
    print(f"\n主回归基准样本量: {len(df_main)} 行")
    print(f"纳入企业数: {df_main['Symbol'].nunique()}")
    print(f"观测年份: {df_main['Year'].min()} - {df_main['Year'].max()}")

    # 模型2：公司固定效应 + 年份固定效应
    print("\n" + "-" * 50)
    print("模型2：OLS 固定效应回归")
    print("因变量：Overpay（超额薪酬）")
    print("固定效应：公司 + 年份")
    print("-" * 50)

    spec = {
        "name": "Model2",
        "dependent": "Overpay",
        "core_var": ALT_SUBSIDY_LAG_COL,
        "control_vars": FE_CONTROL_VARS,
        "time_effects": True,
        "entity_effects": True,
        "industry_year_fe": False,
    }

    result = _run_fe_regression(df_main, spec)
    
    print(f"\n核心变量: {ALT_SUBSIDY_LAG_COL}")
    print(f"系数: {result['coef']:.6f}")
    print(f"标准误: {result['stderr']:.6f}")
    print(f"t值: {result['t_stat']:.4f}")
    print(f"p值: {result['pvalue']:.6f}")
    print(f"显著性: {_classify_pvalue_signal(result['pvalue'])}")
    
    print(f"\n模型拟合指标:")
    print(f"R²: {result['r2']:.4f}")
    print(f"Adj-R²: {result['r2_adj']:.4f}")
    print(f"F统计量: {result['f_stat']:.4f}")
    print(f"F p值: {result['f_pvalue']:.6f}")
    
    print(f"\n控制变量系数:")
    for ctrl_var in FE_CONTROL_VARS:
        if ctrl_var in result['ctrl_coefs']:
            coef = result['ctrl_coefs'][ctrl_var]
            stderr = result['ctrl_stderrs'][ctrl_var]
            pval = result['ctrl_pvalues'][ctrl_var]
            print(f"  {ctrl_var}: {coef:.6f}*** (stderr={stderr:.6f}, p={pval:.6f})")

    return df_main, result


def _run_fe_regression(df, spec):
    """使用 statsmodels 的 PanelOLS 执行固定效应回归"""
    
    var_list = [spec["dependent"], spec["core_var"]] + spec["control_vars"]
    subset = df.dropna(subset=var_list).copy()
    subset = subset.set_index(["Symbol", "Year"])
    
    y = subset[spec["dependent"]]
    X = subset[[spec["core_var"]] + spec["control_vars"]]
    
    # 固定效应回归
    if spec["entity_effects"] and spec["time_effects"]:
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    elif spec["entity_effects"]:
        model = PanelOLS(y, X, entity_effects=True)
    else:
        model = PanelOLS(y, X, time_effects=True)
    
    res = model.fit(cov_type="clustered", cluster_entity=True)
    
    return {
        "coef": float(res.params[spec["core_var"]]),
        "stderr": float(res.std_errors[spec["core_var"]]),
        "t_stat": float(res.params[spec["core_var"]] / res.std_errors[spec["core_var"]]),
        "pvalue": float(res.pvalues[spec["core_var"]]),
        "r2": float(res.r2),
        "r2_adj": float(res.r2_within),
        "f_stat": float(res.f_statistic.stat),
        "f_pvalue": float(res.f_statistic.pval),
        "nobs": int(len(subset)),
        "ctrl_coefs": {var: float(res.params[var]) for var in spec["control_vars"]},
        "ctrl_stderrs": {var: float(res.std_errors[var]) for var in spec["control_vars"]},
        "ctrl_pvalues": {var: float(res.pvalues[var]) for var in spec["control_vars"]},
    }
```

---

### 中介效应检验

**路径a：** $Power_{i,t} = \gamma_0 + \gamma_1 \ln Subsidy^+_{i,t-1} + \gamma_2 ROA_{i,t} + \ldots + \mu_i + \lambda_t + \varepsilon^a_{i,t}$

**路径b 与直接效应：** $Overpay_{i,t} = \delta_0 + \delta_1 \ln Subsidy^+_{i,t-1} + \delta_2 Power_{i,t} + \ldots + \mu_i + \lambda_t + \varepsilon^b_{i,t}$

**非直接效应：** $IE = \gamma_1 \times \delta_2$

```python
def analyze_mediation_effects(df_main, result_main):
    """
    中介效应分解：
    总效应 c = 直接效应 c' + 非直接效应 a×b
    其中：
      路径a：补贴 → 管理层权力
      路径b：管理层权力 → 超额薪酬（控制补贴后）
    """
    print("\n" + "=" * 70)
    print("中介效应检验")
    print("=" * 70)

    # 获取统一中介估计样本
    mediation_vars = [ALT_SUBSIDY_LAG_COL, "Overpay", "Power"] + FE_CONTROL_VARS
    df_mediation = df_main.dropna(subset=mediation_vars).copy()
    print(f"\n中介效应统一样本量: {len(df_mediation)} 行")

    # 路径a: 补贴 → 权力
    print("\n" + "-" * 50)
    print("路径a：补贴强度 → 管理层权力")
    print("-" * 50)
    
    path_a_spec = {
        "name": "Path_a",
        "dependent": "Power",
        "core_var": ALT_SUBSIDY_LAG_COL,
        "control_vars": FE_CONTROL_VARS,
        "time_effects": True,
        "entity_effects": True,
        "industry_year_fe": False,
    }
    
    result_a = _run_fe_regression(df_mediation, path_a_spec)
    print(f"\n{ALT_SUBSIDY_LAG_COL} → Power")
    print(f"系数 (γ₁): {result_a['coef']:.6f}")
    print(f"t值: {result_a['t_stat']:.4f}, p值: {result_a['pvalue']:.6f}")
    print(f"显著性: {_classify_pvalue_signal(result_a['pvalue'])}")

    # 路径b与直接效应: 补贴 + 权力 → 超额薪酬
    print("\n" + "-" * 50)
    print("路径b与直接效应：补贴强度 + 管理层权力 → 超额薪酬")
    print("-" * 50)
    
    path_b_spec = {
        "name": "Path_b",
        "dependent": "Overpay",
        "core_var": ALT_SUBSIDY_LAG_COL,
        "control_vars": [ALT_SUBSIDY_LAG_COL, "Power"] + FE_CONTROL_VARS,
        "time_effects": True,
        "entity_effects": True,
        "industry_year_fe": False,
    }
    
    # 手动构造，包含两个核心变量
    subset = df_mediation.dropna(subset=path_b_spec["control_vars"]).copy()
    subset = subset.set_index(["Symbol", "Year"])
    
    y = subset["Overpay"]
    X = subset[[ALT_SUBSIDY_LAG_COL, "Power"] + FE_CONTROL_VARS]
    
    model_b = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res_b = model_b.fit(cov_type="clustered", cluster_entity=True)
    
    result_c_prime = float(res_b.params[ALT_SUBSIDY_LAG_COL])
    stderr_c_prime = float(res_b.std_errors[ALT_SUBSIDY_LAG_COL])
    result_b = float(res_b.params["Power"])
    stderr_b = float(res_b.std_errors["Power"])
    
    print(f"\nPower → Overpay")
    print(f"系数 (δ₂): {result_b:.6f}")
    print(f"t值: {result_b/stderr_b:.4f}, p值: {float(res_b.pvalues['Power']):.6f}")
    print(f"显著性: {_classify_pvalue_signal(float(res_b.pvalues['Power']))}")
    
    print(f"\n{ALT_SUBSIDY_LAG_COL} 的直接效应（c'）")
    print(f"系数: {result_c_prime:.6f}")
    print(f"t值: {result_c_prime/stderr_c_prime:.4f}, p值: {float(res_b.pvalues[ALT_SUBSIDY_LAG_COL]):.6f}")
    
    # Sobel 检验非直接效应
    print("\n" + "-" * 50)
    print("非直接效应（间接效应）检验")
    print("-" * 50)
    
    indirect = result_a["coef"] * result_b
    se_indirect = np.sqrt(
        (result_b ** 2) * (result_a["stderr"] ** 2) +
        (result_a["coef"] ** 2) * (stderr_b ** 2)
    )
    z_sobel = indirect / se_indirect if se_indirect > 0 else 0
    p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))
    
    print(f"\nSobel 检验:")
    print(f"非直接效应 (a×b): {indirect:.6f}")
    print(f"标准误: {se_indirect:.6f}")
    print(f"z值: {z_sobel:.4f}")
    print(f"p值: {p_sobel:.6f}")
    print(f"显著性: {_classify_pvalue_signal(p_sobel)}")
    
    # Bootstrap 检验
    print("\n" + "-" * 50)
    print("Bootstrap 置信区间（300次重复）")
    print("-" * 50)
    
    bootstrap_result = _bootstrap_mediation(
        df_mediation,
        ALT_SUBSIDY_LAG_COL,
        FE_CONTROL_VARS,
        MAIN_MEDIATION_BOOTSTRAP_REPS,
    )
    
    print(f"间接效应 95% 置信区间: [{bootstrap_result['ci_lower']:.6f}, {bootstrap_result['ci_upper']:.6f}]")
    print(f"Bootstrap p值: {bootstrap_result['p_value']:.6f}")
    print(f"结论: {'显著' if not _ci_excludes_zero(bootstrap_result['ci_lower'], bootstrap_result['ci_upper']) else '不显著（包含零'}")
    
    return {
        "path_a": result_a,
        "path_b_direct": result_c_prime,
        "path_b_indirect": result_b,
        "indirect_effect": indirect,
        "se_indirect": se_indirect,
        "z_sobel": z_sobel,
        "p_sobel": p_sobel,
        "bootstrap": bootstrap_result,
    }


def _bootstrap_mediation(df, subsidy_var, control_vars, reps=300):
    """
    Bootstrap 法检验中介效应置信区间
    """
    np.random.seed(BOOTSTRAP_RANDOM_SEED)
    effects = []
    
    for _ in range(reps):
        # 按公司层面重抽样
        companies = df['Symbol'].unique()
        boot_companies = np.random.choice(companies, size=len(companies), replace=True)
        df_boot = pd.concat([df[df['Symbol'] == c] for c in boot_companies], ignore_index=True)
        
        # 路径a
        try:
            spec_a = {
                "name": "Path_a_boot",
                "dependent": "Power",
                "core_var": subsidy_var,
                "control_vars": control_vars,
                "time_effects": True,
                "entity_effects": True,
                "industry_year_fe": False,
            }
            result_a_boot = _run_fe_regression(df_boot, spec_a)
            coef_a = result_a_boot['coef']
        except:
            continue
        
        # 路径b
        try:
            subset = df_boot.dropna(subset=[subsidy_var, "Power", "Overpay"] + control_vars).copy()
            subset = subset.set_index(["Symbol", "Year"])
            y = subset["Overpay"]
            X = subset[[subsidy_var, "Power"] + control_vars]
            model_b = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res_b = model_b.fit()
            coef_b = float(res_b.params["Power"])
            effects.append(coef_a * coef_b)
        except:
            continue
    
    if len(effects) < 30:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
        }
    
    return {
        "mean": float(np.mean(effects)),
        "ci_lower": float(np.percentile(effects, 2.5)),
        "ci_upper": float(np.percentile(effects, 97.5)),
        "p_value": float(np.mean(np.abs(effects) > 0)),
    }
```

---

## A.7 稳健性检验

**说明：** 通过替换被解释变量、缩小样本期间、聚焦制造业、使用更严格固定效应等方式验证主回归结果的稳定性。

**检验方法：**
1. **替换被解释变量**：使用高管薪酬对数替代超额薪酬
2. **缩小样本期间**：2010-2020年
3. **聚焦制造业样本**：按 IndustrySector == "C"
4. **更严格固定效应**：公司固定效应 + 行业×年份固定效应

```python
def robustness_checks(df):
    """稳健性检验：替换变量、样本期缩减、行业子样本、更严格固定效应。"""
    print("\n" + "=" * 70)
    print("稳健性检验")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    df = _get_main_regression_sample(df)
    core_vars = [MAIN_SUBSIDY_LAG_COL] + control_vars
    rows = []

    # --- 稳健性1：替换被解释变量（薪酬对数替代超额薪酬） ---
    print("\n  [稳健性1] 替换被解释变量: 使用高管前三名薪酬对数替代 Overpay")
    df_r1 = df.dropna(subset=core_vars + ["lnCEOpay", "Year"]).copy()
    if len(df_r1) > 100:
        df_r1 = df_r1.set_index(["Symbol", "Year"], drop=False)
        X_r1, _, _ = _build_fe_matrix(df_r1, core_vars)
        m_r1 = _fit_with_cluster_se(df_r1["lnCEOpay"], X_r1)
        coef = m_r1.params.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        pval = m_r1.pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        print(f"    {MAIN_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r1.rsquared:.4f}")
        print(f"    N={len(df_r1)}, 聚类={df_r1['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "替换因变量",
            "dependent_var": "高管前三名薪酬对数",
            "key_var": MAIN_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r1.tstats.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
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
        coef = m_r2.params.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        pval = m_r2.pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        print(f"    {MAIN_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r2.rsquared:.4f}")
        print(f"    N={len(df_r2)}, 聚类={df_r2['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "缩小样本期(2010-2020)",
            "dependent_var": "Overpay",
            "key_var": MAIN_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r2.tstats.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r2)),
            "n_clusters": int(df_r2["Symbol"].nunique()),
            "r_squared": float(m_r2.rsquared),
        })

    # --- 稳健性3：仅制造业样本 ---
    print("\n  [稳健性3] 仅制造业样本")
    df_r4 = df[df["IndustrySector"] == "C"].copy()
    df_r4 = df_r4.dropna(subset=core_vars + ["Overpay", "Year"])
    if len(df_r4) > 100:
        df_r4 = df_r4.set_index(["Symbol", "Year"], drop=False)
        X_r4, _, _ = _build_fe_matrix(df_r4, core_vars)
        m_r4 = _fit_with_cluster_se(df_r4["Overpay"], X_r4)
        coef = m_r4.params.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        pval = m_r4.pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        print(f"    {MAIN_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r4.rsquared:.4f}")
        print(f"    N={len(df_r4)}, 聚类={df_r4['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "仅制造业",
            "dependent_var": "Overpay",
            "key_var": MAIN_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r4.tstats.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(len(df_r4)),
            "n_clusters": int(df_r4["Symbol"].nunique()),
            "r_squared": float(m_r4.rsquared),
        })

    # --- 稳健性4：更严格固定效应（行业×年份） ---
    print("\n  [稳健性4] 更严格固定效应: 公司固定效应 + 行业×年份固定效应")
    df_r5 = df.dropna(subset=core_vars + ["Overpay", "Year", "IndustrySector"]).copy()
    if len(df_r5) > 100:
        fit_r5 = _fit_model2(df_r5, control_vars, subsidy_col=MAIN_SUBSIDY_LAG_COL, fe_mode="entity_industry_year")
        m_r5 = fit_r5["model"]
        coef = m_r5.params.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        pval = m_r5.pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)
        print(f"    {MAIN_SUBSIDY_LAG_COL} 系数 = {coef:.4f} (p={pval:.4f}), R² = {m_r5.rsquared:.4f}")
        print(f"    N={len(df_r5)}, 聚类={df_r5['Symbol'].nunique()}个")
        print(f"    {'→ 显著，结论稳健 ✓' if pval < 0.05 else '→ 不显著 ✗'}")
        rows.append({
            "check_item": "更严格固定效应（行业×年份）",
            "dependent_var": "Overpay",
            "key_var": MAIN_SUBSIDY_LAG_COL,
            "coef": float(coef),
            "t_value": float(m_r5.tstats.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_value": float(pval),
            "sample_size": int(fit_r5["sample_size"]),
            "n_clusters": int(fit_r5["n_clusters"]),
            "r_squared": float(m_r5.rsquared),
        })

    return pd.DataFrame(rows)
```

---
## A.8 异质性分析

**说明：** 按产权性质（国有 vs 私营）、行业管制特征（管制 vs 非管制）、央地属性（央企 vs 地方国企）进行分层回归。

**分层维度：**
1. **产权性质**：国有企业 (IsSOE==1) vs 私营企业 (IsPrivate==1)
2. **行业管制**：管制行业 (RegulatedIndustry==1) vs 非管制行业
3. **央地国企**：央企 (IsCentralSOE==1) vs 地方国企（仅对国有样本）

```python
def heterogeneity_analysis(df):
    """分企业性质（国有 vs 私营）进行模型2主回归。"""
    print("\n" + "=" * 70)
    print("异质性分析 — 分产权性质（含行业 & 年份固定效应）")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    df = _get_main_regression_sample(df)

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
            "coef_model2": float(fit_result["model"].params.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
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
    print("异质性分析 — 行业管制与央地国企")
    print("=" * 70)

    control_vars = FE_CONTROL_VARS.copy()
    df = _get_main_regression_sample(df)
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
            "coef_model2": float(fit_result["model"].params.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
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
            "coef_model2": float(fit_result["model"].params.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "p_model2": float(fit_result["model"].pvalues.get(MAIN_SUBSIDY_LAG_COL, np.nan)),
            "r2_model2": float(fit_result["model"].rsquared),
            "f_stat_model2": float(_get_model_fstat(fit_result["model"])[0]),
        })

    return pd.DataFrame(rows)
```

---

## A.9 机器学习补充验证

**说明：** 采用随机森林、SHAP、Lasso、XGBoost 四种机器学习方法，在与主回归一致的样本和特征集下，补充验证财政补贴的信息含量与预测意义。

**特征设置与样本保持一致：**
- 特征：lnSubsidy_l1_pos、Roa、Lever、Top1（共4个）
- 目标：Overpay（超额薪酬）
- 样本分割：按企业 GroupShuffleSplit（80%训练，20%测试）
- 交叉验证：5-fold GroupKFold（使企业不跨越折）

**四类方法的验证目的：**

| 方法 | 目的 | 验证指标 |
|------|------|---------|
| 随机森林 | 特征重要性排序 | 排名与相对重要性 |
| SHAP | 边际贡献强度 | 平均绝对SHAP值排名 |
| Lasso | 系数保留与方向 | 是否被压缩、系数符号与OLS一致性 |
| XGBoost | 非线性趋势验证 | 特征重要性、部分依赖图方向 |

```python
def machine_learning_verification(df, output_dir):
    """
    机器学习补充验证：4种方法在相同样本与特征设定下评估补贴的信息含量。
    """
    print("\n" + "=" * 70)
    print("机器学习补充验证分析")
    print("=" * 70)

    # 准备数据
    ml_data = prepare_ml_data(df)
    splits = create_holdout_splits(ml_data["X"], ml_data["y"], ml_data["df_ml"])
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    groups_train = splits["groups_train"]
    feature_names = ml_data["feature_names"]

    results = {}

    # ============================================
    # 方法1：随机森林
    # ============================================
    rf_result = random_forest_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir)
    results["random_forest"] = rf_result

    # ============================================
    # 方法2：SHAP 解释
    # ============================================
    shap_result = rf_shap_analysis(rf_result["model"], X_train, feature_names, output_dir)
    results["shap"] = shap_result

    # ============================================
    # 方法3：Lasso 回归
    # ============================================
    lasso_result = lasso_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir)
    results["lasso"] = lasso_result

    # ============================================
    # 方法4：XGBoost
    # ============================================
    xgb_result = xgboost_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir)
    results["xgboost"] = xgb_result

    # ============================================
    # 汇总表
    # ============================================
    print("\n" + "=" * 70)
    print("四类机器学习方法验证总结")
    print("=" * 70)
    summary_df = pd.DataFrame({
        "方法": ["随机森林", "SHAP", "Lasso", "XGBoost"],
        "补贴变量排名": [
            results["random_forest"]["subsidy_rank"],
            results["shap"]["subsidy_rank"],
            "被保留" if results["lasso"]["subsidy_retained"] else "被压缩",
            results["xgboost"]["subsidy_rank"],
        ],
        "方向/系数": [
            f"重要性={results['random_forest']['subsidy_importance']:.4f}",
            f"SHAP={results['shap']['subsidy_importance']:.4f}",
            f"系数={results['lasso']['subsidy_coef']:.4f} ({results['lasso']['subsidy_sign']})",
            f"重要性={results['xgboost']['subsidy_importance']:.4f}",
        ],
        "测试集R²": [
            f"{results['random_forest']['test_metrics']['R2']:.4f}",
            "（同RF）",
            f"{results['lasso']['test_metrics']['R2']:.4f}",
            f"{results['xgboost']['test_metrics']['R2']:.4f}",
        ],
    })
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, "ml_summary.csv"), index=False)

    print("\n  结论：")
    print(f"  ✓ 补贴在RF中排名第 {results['random_forest']['subsidy_rank']}/{len(feature_names)}")
    print(f"  ✓ 补贴在SHAP中排名第 {results['shap']['subsidy_rank']}/{len(feature_names)}")
    print(f"  ✓ 补贴在Lasso中{'被保留，方向与OLS一致' if results['lasso']['subsidy_retained'] and results['lasso']['subsidy_sign'] == '正' else '被压缩或方向不一致'}")
    print(f"  ✓ 补贴在XGBoost中排名第 {results['xgboost']['subsidy_rank']}/{len(feature_names)}")

    return results


def prepare_ml_data(df):
    """准备机器学习补充分析数据。"""
    feature_cols = [ALT_SUBSIDY_LAG_COL, "Roa", "Lever", "Top1"]
    target_col = "Overpay"
    use_cols = feature_cols + [target_col, "Symbol", "Year"]

    df_ml = df.dropna(subset=use_cols).copy()
    print(f"  样本量: {len(df_ml)}, 企业数: {df_ml['Symbol'].nunique()}")
    print(f"  特征: {feature_cols}")

    return {
        "df_ml": df_ml,
        "X": df_ml[feature_cols].copy(),
        "y": df_ml[target_col].copy(),
        "feature_names": feature_cols,
    }


def create_holdout_splits(X, y, df_ml):
    """按企业分组创建训练/测试集分割。"""
    groups = df_ml["Symbol"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    print(f"  训练集 {len(train_idx)}（企业{groups.iloc[train_idx].nunique()}个），测试集 {len(test_idx)}（企业{groups.iloc[test_idx].nunique()}个）")

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "groups_train": groups.iloc[train_idx].copy(),
        "groups_test": groups.iloc[test_idx].copy(),
    }


def random_forest_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    """随机森林特征重要性分析。"""
    print("\n  1. 随机森林 (Random Forest)")
    
    cv = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [200, 300],
            "max_depth": [6, 10],
            "min_samples_leaf": [10, 20],
            "max_features": ["sqrt", 0.6],
        },
        n_iter=8,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    rf_model = search.best_estimator_

    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": rf_model.feature_importances_,
    }).sort_values("重要性", ascending=False)
    
    subsidy_row = importance_df[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1
    
    print(f"    测试集R² = {r2:.4f}")
    print(f"    {ALT_SUBSIDY_LAG_COL} 排名 = {subsidy_rank}/{len(feature_names)}")
    
    return {
        "model": rf_model,
        "test_metrics": {"R2": r2},
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["重要性"]),
        "importance_table": importance_df.to_dict(orient="records"),
    }


def rf_shap_analysis(rf_model, X_reference, feature_names, output_dir):
    """SHAP 值解释分析。"""
    print("\n  2. SHAP 解释 (Shapley Additive exPlanations)")
    
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
    
    subsidy_row = importance_df[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1
    
    print(f"    {ALT_SUBSIDY_LAG_COL} 排名 = {subsidy_rank}/{len(feature_names)}")
    
    return {
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["平均绝对SHAP"]),
        "importance_table": importance_df.to_dict(orient="records"),
    }


def lasso_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    """Lasso 回归系数方向检验。"""
    print("\n  3. Lasso 回归 (系数方向与保留检验)")
    
    cv = GroupKFold(n_splits=5)
    alpha_grid = np.logspace(-4, 1, 50)
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alpha_grid:
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=20000))])
        scores = cross_val_score(pipeline, X_train, y_train, groups=groups_train, cv=cv, scoring="r2")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_alpha = alpha

    lasso_pipeline = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=best_alpha, max_iter=20000))])
    lasso_pipeline.fit(X_train, y_train)
    y_pred_lasso = lasso_pipeline.predict(X_test)
    
    coefs = lasso_pipeline.named_steps["model"].coef_
    r2_lasso = r2_score(y_test, y_pred_lasso)
    
    coef_df = pd.DataFrame({"特征": feature_names, "系数": coefs}).sort_values("系数", key=abs, ascending=False)
    subsidy_coef = float(coef_df[coef_df["特征"] == ALT_SUBSIDY_LAG_COL]["系数"].values[0])
    subsidy_retained = abs(subsidy_coef) > 1e-6
    
    print(f"    最优 α = {best_alpha:.6f}")
    print(f"    测试集R² = {r2_lasso:.4f}")
    print(f"    {ALT_SUBSIDY_LAG_COL} 系数 = {subsidy_coef:.6f}")
    print(f"    {ALT_SUBSIDY_LAG_COL} {'被保留' if subsidy_retained else '被压缩为0'}")
    
    return {
        "best_alpha": float(best_alpha),
        "test_metrics": {"R2": r2_lasso},
        "subsidy_coef": subsidy_coef,
        "subsidy_retained": bool(subsidy_retained),
        "subsidy_sign": "正" if subsidy_coef > 0 else ("负" if subsidy_coef < 0 else "零"),
        "coef_table": coef_df.to_dict(orient="records"),
    }


def xgboost_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    """XGBoost 特征重要性与部分依赖分析。"""
    print("\n  4. XGBoost (树模型 + 部分依赖曲线)")
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    r2_xgb = r2_score(y_test, y_pred_xgb)
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": xgb_model.feature_importances_,
    }).sort_values("重要性", ascending=False)
    
    subsidy_row = importance_df[importance_df["特征"] == ALT_SUBSIDY_LAG_COL].iloc[0]
    subsidy_rank = int(importance_df.index.get_loc(subsidy_row.name)) + 1
    
    # 部分依赖
    subsidy_idx = feature_names.index(ALT_SUBSIDY_LAG_COL)
    pdp_result = partial_dependence(xgb_model, X_test, features=[subsidy_idx], grid_resolution=50)
    pd_trend = "递增" if pdp_result["average"][0][-1] > pdp_result["average"][0][0] else "递减"
    
    print(f"    测试集R² = {r2_xgb:.4f}")
    print(f"    {ALT_SUBSIDY_LAG_COL} 排名 = {subsidy_rank}/{len(feature_names)}")
    print(f"    部分依赖趋势 = {pd_trend}")
    
    return {
        "model": xgb_model,
        "test_metrics": {"R2": r2_xgb},
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": float(subsidy_row["重要性"]),
        "pdp_trend": pd_trend,
        "importance_table": importance_df.to_dict(orient="records"),
    }
```

---

