"""
生成论文格式回归结果表（统一使用公司层面聚类稳健标准误）
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 复用 regression_analysis.py 的数据和建模逻辑
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from regression_analysis import (
    build_analysis_dataset,
    run_regressions,
    _fit_expectation_salary_model,
    _build_fe_matrix,
    _fit_with_cluster_se,
)


def format_coef(coef, pval):
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    return f"{coef:.4f}{stars}"


def format_tval(tval):
    return f"({tval:.4f})"


def format_pval(pval):
    if pd.isna(pval):
        return "nan"
    if pval < 0.001:
        return "<0.001"
    return f"{pval:.4f}"


def _get_tstat(model, var_name):
    if hasattr(model, "tstats") and var_name in model.tstats:
        return model.tstats[var_name]
    return model.tvalues[var_name]


def _get_std_error(model, var_name):
    if hasattr(model, "std_errors") and var_name in model.std_errors:
        return model.std_errors[var_name]
    return model.bse[var_name]


def _get_fstat(model):
    stat = getattr(model, "f_statistic_robust", None)
    if stat is not None and getattr(stat, "stat", None) is not None:
        return float(stat.stat)
    stat = getattr(model, "f_statistic", None)
    if stat is not None and getattr(stat, "stat", None) is not None:
        return float(stat.stat)
    return np.nan


def fit_cluster_model(df, dep_var, core_vars, df_pre=None):
    """
    dep_var: Overpay / Power / lnCEOpay
    core_vars: 核心变量（不含行业和年份 FE）
    df_pre: 可选，预筛选好的统一样本（跳过内部 dropna）
    """
    if df_pre is not None:
        sub = df_pre
    else:
        needed = core_vars + [dep_var, "IndustrySector", "Year"]
        sub = df.dropna(subset=needed).copy()
    if len(sub) == 0:
        return None, 0
    sub = sub.set_index(["Symbol", "Year"], drop=False)
    x, _, _ = _build_fe_matrix(sub, core_vars)
    y = sub[dep_var]
    model = _fit_with_cluster_se(y, x, sub["Symbol"])
    return model, len(sub)


def build_main_table(main_results):
    """表3 主回归：模型2。"""
    model2 = main_results["model2"]
    main_fit = main_results["main_result"]
    display_vars = ["lnSubsidy_l1", "Roa", "Lever", "Top1", "Zone"]

    lines = []
    lines.append("=" * 90)
    lines.append("表3  主回归结果（模型2，公司与年份固定效应）")
    lines.append("=" * 90)
    lines.append(f"{'变量':<15} {'模型2':<24}")
    lines.append(f"{'':15} {'因变量:Overpay':<24}")
    lines.append("-" * 90)

    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        coef_line += f" {format_coef(model2.params[var], model2.pvalues[var]):<24}"
        tval_line += f" {format_tval(_get_tstat(model2, var)):<24}"
        lines.append(coef_line)
        lines.append(tval_line)

    lines.append("-" * 90)
    lines.append(f"{'Firm FE':<15} {'控制':<24}")
    lines.append(f"{'Year FE':<15} {'控制':<24}")
    lines.append(f"{'N':<15} {main_fit['sample_size']:<24}")
    lines.append(f"{'R-squared':<15} {model2.rsquared:<24.4f}")
    lines.append(f"{'F-stat':<15} {_get_fstat(model2):<24.2f}")
    lines.append("=" * 90)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。F 统计量报告公司层面聚类稳健标准误下的联合显著性检验。标准误为公司层面聚类稳健标准误。")
    return "\n".join(lines)


def build_mediation_table(df):
    """表5 全样本中介效应检验。"""
    results = df if isinstance(df, dict) else run_regressions(df)
    summary = results["summary"]
    model3 = results["model3"]
    model4 = results["model4"]
    model5 = results["model5"]

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("表5  管理层权力的中介效应检验结果（模型3至模型5，FA口径）")
    lines.append("=" * 90)
    lines.append(f"{'路径':<34} {'系数':<16} {'标准误(cluster)':<18} {'Bootstrap 95%CI':<18}")
    lines.append("-" * 90)
    rows = [
        ("模型3 总效应 c：lnSubsidy_l1 → Overpay", format_coef(summary["coef_c"], summary["p_c"]), f"{_get_std_error(model3, 'lnSubsidy_l1'):.4f}", "—"),
        ("模型4 路径 a：lnSubsidy_l1 → Power（FA）", format_coef(summary["coef_a"], summary["p_a"]), f"{_get_std_error(model4, 'lnSubsidy_l1'):.4f}", "—"),
        ("模型5 路径 b：Power（FA） → Overpay", format_coef(summary["coef_b"], summary["p_b"]), f"{_get_std_error(model5, 'Power'):.4f}", "—"),
        ("模型5 直接效应 c'：lnSubsidy_l1 → Overpay", format_coef(summary["coef_c_prime"], summary["p_c_prime"]), f"{_get_std_error(model5, 'lnSubsidy_l1'):.4f}", "—"),
        ("间接效应 a×b", f"{summary['indirect_effect']:.6f}", "—", f"[{summary['bootstrap_ci_lower']:.6f}, {summary['bootstrap_ci_upper']:.6f}]"),
        ("Sobel p 值", f"{summary['sobel_p']:.4f}", "—", "—"),
        ("中介效应占比（a×b / c）", f"{summary['mediation_ratio_pct']:.2f}%", "—", "—"),
    ]
    for label, coef, se, ci in rows:
        lines.append(f"{label:<34} {coef:<16} {se:<18} {ci:<18}")
    lines.append("-" * 90)
    lines.append("=" * 90)
    lines.append("注：括号中星号表示显著性水平，* p<0.1, ** p<0.05, *** p<0.01。模型3报告总效应，模型4报告路径a，模型5同时报告路径b与直接效应。Bootstrap 为公司层面 cluster bootstrap，300 次重抽样。所有回归均控制 Roa、Lever、Top1、Zone 以及公司和年份固定效应。")
    return "\n".join(lines)


def build_causal_table(main_results):
    """表4 FE-2SLS 因果识别结果。"""
    iv_result = main_results["iv_result"]
    second_stage = iv_result["model"]
    first_stage = iv_result["first_stage"]

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("表4  工具变量（FE-2SLS）估计结果")
    lines.append("=" * 90)
    lines.append(f"{'阶段':<12} {'因变量':<18} {'系数':<16} {'统计量':<22} {'N':<10}")
    lines.append("-" * 90)
    rows = [
        ("第一阶段", "lnSubsidy_l1", f"{format_coef(first_stage['coef'], first_stage['f_pval'])}", f"Partial F = {first_stage['f_stat']:.2f}", f"{iv_result['sample_size']}"),
        ("第二阶段", "Overpay", f"{format_coef(second_stage.params['lnSubsidy_l1'], second_stage.pvalues['lnSubsidy_l1'])}", f"t = {_get_tstat(second_stage, 'lnSubsidy_l1'):.4f}", f"{iv_result['sample_size']}"),
    ]
    for stage, dep_var, coef, stat, nobs in rows:
        lines.append(f"{stage:<12} {dep_var:<18} {coef:<16} {stat:<22} {nobs:<10}")
    lines.append("=" * 90)
    lines.append(f"注：第一阶段报告工具变量系数；Partial R² 为 {first_stage['partial_r2']:.4f}，说明工具变量具有统计相关性但解释力度有限。标准误为公司层面聚类稳健标准误。")
    return "\n".join(lines)


def build_first_stage_table(df):
    """表2 第一阶段期望薪酬模型。"""
    fit_result = _fit_expectation_salary_model(df)
    model = fit_result["model"]
    n = len(fit_result["df_model1"])

    display_vars = ["lnSale", "Roa", "IA", "Zone"]
    explanations = {
        "lnSale": "企业规模越大，期望薪酬越高",
        "Roa": "盈利能力越强，期望薪酬越高",
        "IA": "无形资产占比较高时，期望薪酬相对较低",
        "Zone": "在中西部=1、东部=0的定义下，负系数表示中西部期望薪酬较低",
    }

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("表2  第一阶段期望薪酬模型估计结果")
    lines.append("=" * 90)
    lines.append(f"{'变量':<15} {'系数':<20} {'t值':<20} {'说明'}")
    lines.append("-" * 90)
    for var in display_vars:
        coef = format_coef(model.params[var], model.pvalues[var]) if model is not None else "nan"
        tval = format_tval((model.tstats[var] if hasattr(model, 'tstats') else model.tvalues[var])) if model is not None else ""
        lines.append(f"{var:<15} {coef:<20} {tval:<20} {explanations[var]}")
    lines.append("-" * 90)
    lines.append(f"{'Industry FE':<15} {'控制':<20} {'':<20} {'17 个行业虚拟变量'}")
    lines.append(f"{'Year FE':<15} {'控制':<20} {'':<20} {'21 个年份虚拟变量'}")
    lines.append(f"{'N':<15} {n:<20} {'':<20} {'—'}")
    lines.append(f"{'R-squared':<15} {(model.rsquared if model else np.nan):<20.4f} {'':<20} {'—'}")
    lines.append("=" * 90)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。该模型使用 OLS 并控制行业与年份虚拟变量，用于估计正常薪酬水平，其残差直接构造为 Overpay。")
    return "\n".join(lines)


def build_power_method_comparison_table(main_results):
    """表5 Power 构造方法比较。"""
    method_df = main_results["method_comparison"].copy()
    if method_df.empty:
        return ""

    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("表5  `Power` 不同构造口径下的全样本中介结果比较")
    lines.append("=" * 110)
    lines.append(f"{'方法':<12} {'角色':<18} {'路径b':<18} {'bootstrap 95%CI':<34} {'结论'}")
    lines.append("-" * 110)
    for _, row in method_df.iterrows():
        coef_b = format_coef(row["coef_b"], row["p_b"])
        ci = f"[{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}]"
        lines.append(
            f"{row['method']:<12} {row['role']:<18} {coef_b:<18} {ci:<34} {row['mediation_type']}"
        )
    lines.append("=" * 110)
    lines.append("注：FA 为修订后主测度，PCA 为上一版原始方案对照，熵值法为稳健性替代。bootstrap 95%CI 为公司层面 cluster bootstrap（300 次）得到的 95% 置信区间。")
    return "\n".join(lines)


def build_subsample_table(df, table_title, group_specs):
    """
    通用分组回归表：每组仅展示模型2
    group_specs: [(label, mask_series), ...]
    """
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    display_vars = ["lnSubsidy_l1", "Roa", "Lever", "Top1", "Zone"]

    models = []
    for label, mask in group_specs:
        df_sub = df[mask].copy()
        m2, n2 = fit_cluster_model(df_sub, "Overpay", ["lnSubsidy_l1"] + control_vars)
        models.append((label, m2, n2))

    col_count = len(models)
    line_width = max(100, 18 + col_count * 22)
    lines = []
    lines.append("")
    lines.append("=" * line_width)
    lines.append(table_title)
    lines.append("=" * line_width)

    header = f"{'变量':<15}"
    subheader = f"{'':15}"
    for label, _, _ in models:
        header += f" {label:<22}"
        subheader += f" {'模型2':<22}"
    lines.append(header)
    lines.append(subheader)
    lines.append("-" * line_width)

    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        for _, model, _ in models:
            if model is not None and var in model.params:
                coef_line += f" {format_coef(model.params[var], model.pvalues[var]):<22}"
                tval_line += f" {format_tval(_get_tstat(model, var)):<22}"
            else:
                coef_line += f" {'':22}"
                tval_line += f" {'':22}"
        lines.append(coef_line)
        lines.append(tval_line)

    lines.append("-" * line_width)
    fe_line_ind = f"{'Firm FE':<15}"
    fe_line_year = f"{'Year FE':<15}"
    n_line = f"{'N':<15}"
    r2_line = f"{'R-squared':<15}"

    for _, model, nobs in models:
        fe_line_ind += f" {'控制':<22}"
        fe_line_year += f" {'控制':<22}"
        n_line += f" {nobs:<22}"
        r2_line += f" {(model.rsquared if model else np.nan):<22.4f}"

    lines.append(fe_line_ind)
    lines.append(fe_line_year)
    lines.append(n_line)
    lines.append(r2_line)
    lines.append("=" * line_width)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。")
    return "\n".join(lines)


def build_desc_table(df):
    """表1 描述性统计"""
    desc_vars_map = {
        "Top3Salary": "高管前三名薪酬总额",
        "SubsidyAmount": "政府补助",
        "lnSubsidy": "财政补贴强度(lnSubsidy)",
        "lnCEOpay": "薪酬对数(lnCEOpay)",
        "lnSale": "企业规模(lnSale)",
        "IA": "无形资产占比(IA)",
        "Overpay": "超额薪酬(Overpay)",
        "Power": "管理层权力(Power, FA)",
        "Roa": "业绩(Roa)",
        "Lever": "财务杠杆",
        "Top1": "第一大股东持股比例",
        "Zone": "地区(Zone, 中西部=1)",
    }
    desc_data = []
    for var, label in desc_vars_map.items():
        if var in df.columns:
            s = df[var].dropna()
            desc_data.append({
                "变量": label,
                "N": int(s.count()),
                "均值": f"{s.mean():.4f}" if abs(s.mean()) < 1e6 else f"{s.mean():.2e}",
                "中位数": f"{s.median():.4f}" if abs(s.median()) < 1e6 else f"{s.median():.2e}",
                "标准差": f"{s.std():.4f}" if abs(s.std()) < 1e6 else f"{s.std():.2e}",
                "最小值": f"{s.min():.4f}" if abs(s.min()) < 1e6 else f"{s.min():.2e}",
                "最大值": f"{s.max():.4f}" if abs(s.max()) < 1e6 else f"{s.max():.2e}",
            })
    desc_df = pd.DataFrame(desc_data)

    lines = []
    lines.append("=" * 90)
    lines.append("表1  主要变量描述性统计")
    lines.append("=" * 90)
    lines.append(desc_df.to_string(index=False))
    lines.append("=" * 90)
    return "\n".join(lines)


def build_robustness_table(df):
    """表6 稳健性检验汇总表"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    core_vars = ["lnSubsidy_l1"] + control_vars
    results = []

    # 稳健性1：替换被解释变量（lnCEOpay 替代 Overpay）
    m1, n1 = fit_cluster_model(df, "lnCEOpay", core_vars)
    results.append(("(1) 替换因变量\n    (lnCEOpay)", m1, n1, "lnSubsidy_l1"))

    # 稳健性2：缩小样本期间（2010-2020）
    df_r2 = df[(df["Year"] >= 2010) & (df["Year"] <= 2020)].copy()
    m2, n2 = fit_cluster_model(df_r2, "Overpay", core_vars)
    results.append(("(2) 缩小样本期\n    (2010-2020)", m2, n2, "lnSubsidy_l1"))

    # 稳健性3：仅制造业样本
    df_r4 = df[df["Industry"] == 1].copy()
    m4, n4 = fit_cluster_model(df_r4, "Overpay", core_vars)
    results.append(("(3) 仅制造业\n    ", m4, n4, "lnSubsidy_l1"))

    # 稳健性4：替换解释变量 ln(1+补助)
    df_alt = df.sort_values(["Symbol", "Year"]).copy()
    if "lnSubsidy1p_l1" not in df_alt.columns:
        df_alt["lnSubsidy1p_l1"] = df_alt.groupby("Symbol")["lnSubsidy1p"].shift(1)
    core_vars_alt = ["lnSubsidy1p_l1"] + control_vars
    m5, n5 = fit_cluster_model(df_alt, "Overpay", core_vars_alt)
    results.append(("(4) 替换解释变量\n    (ln(1+补助))", m5, n5, "lnSubsidy1p_l1"))

    # 构建表格
    col_width = 20
    n_cols = len(results)
    line_width = max(100, 15 + n_cols * (col_width + 2))
    lines = []
    lines.append("")
    lines.append("=" * line_width)
    lines.append("表6  稳健性检验结果（聚类标准误）")
    lines.append("=" * line_width)

    # 表头行1：检验编号
    header1 = f"{'':15}"
    for label, _, _, _ in results:
        short_label = label.split("\n")[0]  # 取第一行作为列头
        header1 += f" {short_label:<{col_width}}"
    lines.append(header1)

    # 表头行2：因变量
    header2 = f"{'':15}"
    for label, _, _, _ in results:
        dep = "lnCEOpay" if "因变量" in label else "Overpay"
        if "(1)" in label:
            dep = "lnCEOpay"
        else:
            dep = "Overpay"
        header2 += f" {dep:<{col_width}}"
    lines.append(header2)
    lines.append("-" * line_width)

    # 核心变量行：补助系数 + t值
    coef_line = f"{'补助变量':15}"
    tval_line = f"{'':15}"
    for _, model, _, var_name in results:
        if model is not None and var_name in model.params:
            coef_line += f" {format_coef(model.params[var_name], model.pvalues[var_name]):<{col_width}}"
            tval_line += f" {format_tval(_get_tstat(model, var_name)):<{col_width}}"
        else:
            coef_line += f" {'—':<{col_width}}"
            tval_line += f" {'':>{col_width}}"
    lines.append(coef_line)
    lines.append(tval_line)

    lines.append("-" * line_width)

    # 控制变量
    ctrl_line = f"{'Controls':15}"
    for _ in results:
        ctrl_line += f" {'控制':<{col_width}}"
    lines.append(ctrl_line)

    # 行业固定效应
    fe_ind_line = f"{'Firm FE':15}"
    for _ in results:
        fe_ind_line += f" {'控制':<{col_width}}"
    lines.append(fe_ind_line)

    # 年份固定效应
    fe_year_line = f"{'Year FE':15}"
    for _ in results:
        fe_year_line += f" {'控制':<{col_width}}"
    lines.append(fe_year_line)

    # N
    n_line = f"{'N':15}"
    for _, _, n, _ in results:
        n_line += f" {n:<{col_width}}"
    lines.append(n_line)

    # R²
    r2_line = f"{'R-squared':15}"
    for _, model, _, _ in results:
        r2 = model.rsquared if model else np.nan
        r2_line += f" {r2:<{col_width}.4f}"
    lines.append(r2_line)

    lines.append("=" * line_width)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。")
    return "\n".join(lines)


def generate_tables(df, main_results):
    """生成论文结果表并保存"""
    table1 = build_desc_table(df)
    table2 = build_first_stage_table(df)
    table3 = build_main_table(main_results)
    table4 = build_causal_table(main_results)
    table5 = build_mediation_table(main_results)
    table6 = build_robustness_table(df)

    table7 = build_subsample_table(
        df,
        "表7  分产权性质主回归结果（模型2，聚类标准误）",
        [
            ("国有", df["IsSOE"] == 1),
            ("私营", df["IsPrivate"] == 1),
        ],
    )

    table8 = build_subsample_table(
        df,
        "表8  分行业管制主回归结果（模型2，聚类标准误）",
        [
            ("管制行业", df["RegulatedIndustry"] == 1),
            ("非管制行业", df["RegulatedIndustry"] == 0),
        ],
    )

    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan
    table9 = build_subsample_table(
        df,
        f"表9  央企与地方国企主回归结果（模型2，聚类标准误，央地可识别比例={identified_ratio:.2%}）",
        [
            ("央企", df["IsCentralSOE"] == 1),
            ("地方国企", df["IsCentralSOE"] == 0),
        ],
    )

    all_tables = "\n\n".join([table1, table2, table3, table4, table5, table6, table7, table8, table9])
    print(all_tables)

    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "regression_tables.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_tables)
    print(f"\n所有表格已保存至: {output_file}")


def main():
    data_dir = os.path.join(os.getcwd(), "processed_data")
    print("加载数据...")
    df, _ = build_analysis_dataset(data_dir)
    main_results = run_regressions(df)

    print("\n\n" + "#" * 90)
    print("#  以下为论文格式回归结果表格（公司层面聚类标准误）")
    print("#" * 90 + "\n")

    generate_tables(df, main_results)


if __name__ == "__main__":
    main()
