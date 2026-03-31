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
    grouped_mediation_analysis,
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


def fit_cluster_model(df, dep_var, core_vars, df_pre=None):
    """
    dep_var: Overpay / Power / lnCEOpay
    core_vars: 核心变量（不含行业和年份 FE）
    df_pre: 可选，预筛选好的统一样本（跳过内部 dropna）
    """
    if df_pre is not None:
        sub = df_pre
    else:
        needed = core_vars + [dep_var, "IndustrySector"]
        sub = df.dropna(subset=needed).copy()
    if len(sub) == 0:
        return None, 0
    x, _, _ = _build_fe_matrix(sub, core_vars)
    y = sub[dep_var]
    model = _fit_with_cluster_se(y, x, sub["Symbol"])
    return model, len(sub)


def build_main_table(df):
    """表3 主回归：模型3/4/5（统一样本）"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]

    # 统一样本：取模型4所需全部变量的交集
    all_needed = ["lnSubsidy", "Power"] + control_vars + ["Overpay", "IndustrySector"]
    df_unified = df.dropna(subset=all_needed).copy()

    m3, n3 = fit_cluster_model(df, "Overpay", ["lnSubsidy"] + control_vars, df_pre=df_unified)
    m4, n4 = fit_cluster_model(df, "Overpay", ["lnSubsidy", "Power"] + control_vars, df_pre=df_unified)
    m5, n5 = fit_cluster_model(df, "Power", ["lnSubsidy"] + control_vars, df_pre=df_unified)

    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]

    lines = []
    lines.append("=" * 90)
    lines.append("表3  政府补助、管理层权力与高管超额薪酬的回归结果（修订后主测度：Power 为 FA）")
    lines.append("=" * 90)
    lines.append(f"{'变量':<15} {'模型(3)':<24} {'模型(4)':<24} {'模型(5)':<24}")
    lines.append(f"{'':15} {'因变量:Overpay':<24} {'因变量:Overpay':<24} {'因变量:Power(FA)':<24}")
    lines.append("-" * 90)

    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        for model in [m3, m4, m5]:
            if model is not None and var in model.params:
                coef_line += f" {format_coef(model.params[var], model.pvalues[var]):<24}"
                tval_line += f" {format_tval(model.tvalues[var]):<24}"
            else:
                coef_line += f" {'':24}"
                tval_line += f" {'':24}"
        lines.append(coef_line)
        lines.append(tval_line)

    lines.append("-" * 90)
    lines.append(f"{'Industry FE':<15} {'控制':<24} {'控制':<24} {'控制':<24}")
    lines.append(f"{'Year FE':<15} {'控制':<24} {'控制':<24} {'控制':<24}")
    lines.append(f"{'N':<15} {n3:<24} {n4:<24} {n5:<24}")
    lines.append(
        f"{'R-squared':<15} "
        f"{(m3.rsquared if m3 else np.nan):<24.4f} "
        f"{(m4.rsquared if m4 else np.nan):<24.4f} "
        f"{(m5.rsquared if m5 else np.nan):<24.4f}"
    )
    lines.append("=" * 90)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。Power 为因子分析（FA）构造的修订后主测度；PCA 结果作为上一版原始方案对照另行报告。标准误为公司层面聚类稳健标准误。")
    return "\n".join(lines)


def build_mediation_table(df):
    """表4 全样本中介效应检验。"""
    summary = df["summary"] if isinstance(df, dict) else run_regressions(df)["summary"]

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("表4  管理层权力中介效应检验（全样本，修订后主测度：Power 为 FA）")
    lines.append("=" * 90)
    lines.append(f"{'路径':<28} {'系数':<18} {'p值':<18}")
    lines.append("-" * 90)
    rows = [
        ("lnSubsidy → Overpay（总效应 c）", f"{summary['coef_c']:.4f}", format_pval(summary["p_c"])),
        ("lnSubsidy → Power（路径 a）", f"{summary['coef_a']:.4f}", format_pval(summary["p_a"])),
        ("Power → Overpay（路径 b）", f"{summary['coef_b']:.4f}", format_pval(summary["p_b"])),
        ("lnSubsidy → Overpay（直接效应 c'）", f"{summary['coef_c_prime']:.4f}", format_pval(summary["p_c_prime"])),
        ("间接效应（a×b）", f"{summary['indirect_effect']:.6f}", "—"),
        ("Sobel Z", f"{summary['sobel_z']:.4f}", "—"),
        ("Sobel p", format_pval(summary["sobel_p"]), summary["sobel_signal"]),
        ("Bootstrap p", format_pval(summary["bootstrap_p"]), "—"),
        ("Bootstrap 95%CI", f"[{summary['bootstrap_ci_lower']:.6f}, {summary['bootstrap_ci_upper']:.6f}]", "—"),
        ("路径a显著", "是" if summary["path_a_significant"] else "否", "—"),
        ("路径b显著", "是" if summary["path_b_significant"] else "否", "—"),
        ("中介效应占比(%)", f"{summary['mediation_ratio_pct']:.2f}", "—"),
    ]
    for label, coef, pval in rows:
        lines.append(f"{label:<28} {coef:<18} {pval:<18}")
    lines.append("-" * 90)
    lines.append(f"中介类型：{summary['mediation_type']}")
    lines.append("=" * 90)
    lines.append("注：常规中介效应要求路径a、路径b均显著、间接效应方向与总效应一致，且 bootstrap 95%CI 不含0；PCA 原始方案对照与熵值法稳健性结果另行比较报告。")
    return "\n".join(lines)


def build_first_stage_table(df):
    """表2 第一阶段期望薪酬模型。"""
    core_vars = ["lnSale", "Roa", "IA", "Zone"]
    model, n = fit_cluster_model(df, "lnCEOpay", core_vars)

    display_vars = ["lnSale", "Roa", "IA", "Zone"]
    explanations = {
        "lnSale": "企业规模越大，期望薪酬越高",
        "Roa": "盈利能力越强，期望薪酬越高",
        "IA": "无形资产占比较高时，期望薪酬相对较低",
        "Zone": "中西部地区样本的期望薪酬相对较低",
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
        tval = format_tval(model.tvalues[var]) if model is not None else ""
        lines.append(f"{var:<15} {coef:<20} {tval:<20} {explanations[var]}")
    lines.append("-" * 90)
    lines.append(f"{'Industry FE':<15} {'控制':<20} {'':<20} {'17 个行业虚拟变量'}")
    lines.append(f"{'Year FE':<15} {'控制':<20} {'':<20} {'21 个年份虚拟变量'}")
    lines.append(f"{'N':<15} {n:<20} {'':<20} {'—'}")
    lines.append(f"{'R-squared':<15} {(model.rsquared if model else np.nan):<20.4f} {'':<20} {'—'}")
    lines.append("=" * 90)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。该模型用于估计正常薪酬水平，并据此构造超额薪酬 Overpay。标准误为公司层面聚类稳健标准误。")
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


def build_grouped_mediation_table(df):
    """附表A 分组中介效应汇总。"""
    summary_df = df if isinstance(df, pd.DataFrame) else grouped_mediation_analysis(df)
    display_df = summary_df.copy()
    display_df.rename(columns={
        "layer": "层级",
        "group": "分组",
        "sample_size": "N",
        "n_clusters": "聚类数",
        "coef_a": "路径a",
        "coef_b": "路径b",
        "sobel_p": "原始Sobel_p",
        "fdr_p": "FDR_p",
        "path_b_significant": "路径b显著",
        "mediation_type": "中介类型",
        "group_evidence_level": "严谨口径",
    }, inplace=True)

    display_df["Bootstrap 95%CI"] = display_df.apply(
        lambda row: (
            f"[{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}]"
            if pd.notna(row["bootstrap_ci_lower"]) and pd.notna(row["bootstrap_ci_upper"])
            else "未执行"
        ),
        axis=1,
    )

    cols = ["层级", "分组", "N", "聚类数", "路径a", "路径b", "原始Sobel_p", "FDR_p", "路径b显著", "Bootstrap 95%CI", "中介类型", "严谨口径"]
    for col in ["路径a", "路径b"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    for col in ["原始Sobel_p", "FDR_p"]:
        display_df[col] = display_df[col].map(format_pval)
    display_df["路径b显著"] = display_df["路径b显著"].map(lambda x: "是" if bool(x) else "否")

    lines = []
    lines.append("")
    lines.append("=" * 180)
    lines.append("附表A  分组中介效应汇总（FA 修订后主测度，含分层FDR与bootstrap复核）")
    lines.append("=" * 180)
    lines.append(display_df[cols].to_string(index=False))
    lines.append("=" * 180)
    lines.append("注：FDR 为在“产权直分”“补充分组”两个层级内分别进行的 BH 校正；bootstrap 仅对原始 Sobel p<0.10 的分组执行。严谨口径在中介类型基础上进一步要求通过相应层内 FDR，未通过者仅可谨慎解释。")
    return "\n".join(lines)


def build_subsample_table(df, table_title, group_specs):
    """
    通用分组回归表：每组展示模型3(无Power)和模型4(有Power)
    group_specs: [(label, mask_series), ...]
    """
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]

    models = []
    for label, mask in group_specs:
        df_sub = df[mask].copy()
        # 统一样本：取模型4所需全部变量的交集
        all_needed = ["lnSubsidy", "Power"] + control_vars + ["Overpay", "IndustrySector"]
        df_sub_unified = df_sub.dropna(subset=all_needed).copy()
        m3, n3 = fit_cluster_model(df_sub, "Overpay", ["lnSubsidy"] + control_vars, df_pre=df_sub_unified)
        m4, n4 = fit_cluster_model(df_sub, "Overpay", ["lnSubsidy", "Power"] + control_vars, df_pre=df_sub_unified)
        models.append((label, m3, n3, m4, n4))

    col_count = len(models) * 2
    line_width = max(100, 18 + col_count * 22)
    lines = []
    lines.append("")
    lines.append("=" * line_width)
    lines.append(table_title)
    lines.append("=" * line_width)

    header = f"{'变量':<15}"
    subheader = f"{'':15}"
    for label, _, _, _, _ in models:
        header += f" {label+'-模型(3)':<22} {label+'-模型(4)':<22}"
        subheader += f" {'Overpay':<22} {'Overpay':<22}"
    lines.append(header)
    lines.append(subheader)
    lines.append("-" * line_width)

    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        for _, m3, _, m4, _ in models:
            for m in [m3, m4]:
                if m is not None and var in m.params:
                    coef_line += f" {format_coef(m.params[var], m.pvalues[var]):<22}"
                    tval_line += f" {format_tval(m.tvalues[var]):<22}"
                else:
                    coef_line += f" {'':22}"
                    tval_line += f" {'':22}"
        lines.append(coef_line)
        lines.append(tval_line)

    lines.append("-" * line_width)
    fe_line_ind = f"{'Industry FE':<15}"
    fe_line_year = f"{'Year FE':<15}"
    n_line = f"{'N':<15}"
    r2_line = f"{'R-squared':<15}"

    for _, m3, n3, m4, n4 in models:
        fe_line_ind += f" {'控制':<22} {'控制':<22}"
        fe_line_year += f" {'控制':<22} {'控制':<22}"
        n_line += f" {n3:<22} {n4:<22}"
        r2_line += f" {(m3.rsquared if m3 else np.nan):<22.4f} {(m4.rsquared if m4 else np.nan):<22.4f}"

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
        "Overpay": "超额薪酬(Overpay)",
        "Power": "管理层权力(Power, FA)",
        "Roa": "业绩(Roa)",
        "Revenue": "营业收入",
        "Lever": "财务杠杆",
        "Top1": "第一大股东持股比例",
    }
    desc_data = []
    for var, label in desc_vars_map.items():
        if var in df.columns:
            s = df[var].dropna()
            desc_data.append({
                "变量": label,
                "N": int(s.count()),
                "均值": f"{s.mean():.4f}" if abs(s.mean()) < 1e6 else f"{s.mean():.2e}",
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
    """表9 稳健性检验汇总表"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    core_vars = ["lnSubsidy"] + control_vars
    results = []

    # 稳健性1：替换被解释变量（lnCEOpay 替代 Overpay）
    m1, n1 = fit_cluster_model(df, "lnCEOpay", core_vars)
    results.append(("(1) 替换因变量\n    (lnCEOpay)", m1, n1, "lnSubsidy"))

    # 稳健性2：缩小样本期间（2010-2020）
    df_r2 = df[(df["Year"] >= 2010) & (df["Year"] <= 2020)].copy()
    m2, n2 = fit_cluster_model(df_r2, "Overpay", core_vars)
    results.append(("(2) 缩小样本期\n    (2010-2020)", m2, n2, "lnSubsidy"))

    # 稳健性3：剔除极端补助值（5%/95%）
    q05 = df["lnSubsidy"].quantile(0.05)
    q95 = df["lnSubsidy"].quantile(0.95)
    df_r3 = df[(df["lnSubsidy"] >= q05) & (df["lnSubsidy"] <= q95)].copy()
    m3, n3 = fit_cluster_model(df_r3, "Overpay", core_vars)
    results.append(("(3) 剔除极端补助\n    (5%-95%)", m3, n3, "lnSubsidy"))

    # 稳健性4：仅制造业样本
    df_r4 = df[df["Industry"] == 1].copy()
    m4, n4 = fit_cluster_model(df_r4, "Overpay", core_vars)
    results.append(("(4) 仅制造业\n    ", m4, n4, "lnSubsidy"))

    # 稳健性5：替换解释变量 ln(1+补助)
    core_vars_alt = ["lnSubsidy1p"] + control_vars
    m5, n5 = fit_cluster_model(df, "Overpay", core_vars_alt)
    results.append(("(5) 替换解释变量\n    (ln(1+补助))", m5, n5, "lnSubsidy1p"))

    # 稳健性6：滞后一期补助
    df_lag = df.sort_values(["Symbol", "Year"]).copy()
    df_lag["lnSubsidy_l1"] = df_lag.groupby("Symbol")["lnSubsidy"].shift(1)
    core_vars_lag = ["lnSubsidy_l1"] + control_vars
    m6, n6 = fit_cluster_model(df_lag, "Overpay", core_vars_lag)
    results.append(("(6) 滞后一期补助\n    (lnSubsidy_l1)", m6, n6, "lnSubsidy_l1"))

    # 构建表格
    col_width = 20
    n_cols = len(results)
    line_width = max(100, 15 + n_cols * (col_width + 2))
    lines = []
    lines.append("")
    lines.append("=" * line_width)
    lines.append("表9  稳健性检验结果（聚类标准误）")
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
            tval_line += f" {format_tval(model.tvalues[var_name]):<{col_width}}"
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
    fe_ind_line = f"{'Industry FE':15}"
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


def generate_tables(df, main_results, grouped_results):
    """生成论文结果表并保存"""
    table1 = build_desc_table(df)
    table2 = build_first_stage_table(df)
    table3 = build_main_table(df)
    table4 = build_mediation_table(main_results)
    table5 = build_power_method_comparison_table(main_results)

    table6 = build_subsample_table(
        df,
        "表6  分产权性质回归结果（聚类标准误）",
        [
            ("国有", df["IsSOE"] == 1),
            ("私营", df["IsPrivate"] == 1),
        ],
    )

    table7 = build_subsample_table(
        df,
        "表7  分行业管制回归结果（聚类标准误）",
        [
            ("管制行业", df["RegulatedIndustry"] == 1),
            ("非管制行业", df["RegulatedIndustry"] == 0),
        ],
    )

    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan
    table8 = build_subsample_table(
        df,
        f"表8  央企与地方国企回归结果（聚类标准误，央地可识别比例={identified_ratio:.2%}）",
        [
            ("央企", df["IsCentralSOE"] == 1),
            ("地方国企", df["IsCentralSOE"] == 0),
        ],
    )

    table9 = build_robustness_table(df)
    appendix_a = build_grouped_mediation_table(grouped_results)

    all_tables = "\n\n".join([table1, table2, table3, table4, table5, table6, table7, table8, table9, appendix_a])
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
    grouped_results = grouped_mediation_analysis(df)

    print("\n\n" + "#" * 90)
    print("#  以下为论文格式回归结果表格（公司层面聚类标准误）")
    print("#" * 90 + "\n")

    generate_tables(df, main_results, grouped_results)


if __name__ == "__main__":
    main()
