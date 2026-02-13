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
    load_and_clean_data,
    construct_variables,
    filter_and_describe,
    compute_overpay,
    compute_power,
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


def fit_cluster_model(df, dep_var, core_vars):
    """
    dep_var: Overpay / Power / lnCEOpay
    core_vars: 核心变量（不含行业和年份 FE）
    """
    needed = core_vars + [dep_var, "IndustrySector"]
    sub = df.dropna(subset=needed).copy()
    if len(sub) == 0:
        return None, 0
    x, _, _ = _build_fe_matrix(sub, core_vars)
    y = sub[dep_var]
    model = _fit_with_cluster_se(y, x, sub["Symbol"])
    return model, len(sub)


def build_main_table(df):
    """表2 主回归：模型3/4/5"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    m3, n3 = fit_cluster_model(df, "Overpay", ["lnSubsidy"] + control_vars)
    m4, n4 = fit_cluster_model(df, "Overpay", ["lnSubsidy", "Power"] + control_vars)
    m5, n5 = fit_cluster_model(df, "Power", ["lnSubsidy"] + control_vars)

    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]

    lines = []
    lines.append("=" * 90)
    lines.append("表2  政府补助、管理层权力与高管超额薪酬的回归结果（聚类标准误）")
    lines.append("=" * 90)
    lines.append(f"{'变量':<15} {'模型(3)':<24} {'模型(4)':<24} {'模型(5)':<24}")
    lines.append(f"{'':15} {'因变量:Overpay':<24} {'因变量:Overpay':<24} {'因变量:Power':<24}")
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
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。")
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
        m3, n3 = fit_cluster_model(df_sub, "Overpay", ["lnSubsidy"] + control_vars)
        m4, n4 = fit_cluster_model(df_sub, "Overpay", ["lnSubsidy", "Power"] + control_vars)
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
        "Overpay": "超额薪酬",
        "Power": "管理层权力",
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


def generate_tables(df):
    """生成论文结果表并保存"""
    table1 = build_desc_table(df)
    table2 = build_main_table(df)

    table3 = build_subsample_table(
        df,
        "表3  分产权性质回归结果（聚类标准误）",
        [
            ("国有", df["IsSOE"] == 1),
            ("非国有", df["IsSOE"] == 0),
        ],
    )

    table4 = build_subsample_table(
        df,
        "表4  分行业管制回归结果（聚类标准误）",
        [
            ("管制行业", df["RegulatedIndustry"] == 1),
            ("非管制行业", df["RegulatedIndustry"] == 0),
        ],
    )

    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan
    table5 = build_subsample_table(
        df,
        f"表5  央企与地方国企回归结果（聚类标准误，央地可识别比例={identified_ratio:.2%}）",
        [
            ("央企", df["IsCentralSOE"] == 1),
            ("地方国企", df["IsCentralSOE"] == 0),
        ],
    )

    all_tables = "\n\n".join([table1, table2, table3, table4, table5])
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
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    df = compute_overpay(df)
    df = compute_power(df)

    print("\n\n" + "#" * 90)
    print("#  以下为论文格式回归结果表格（公司层面聚类标准误）")
    print("#" * 90 + "\n")

    generate_tables(df)


if __name__ == "__main__":
    main()

