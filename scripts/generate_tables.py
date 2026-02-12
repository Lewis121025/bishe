"""
将回归结果整理为论文格式表格，保存为 txt 和 csv
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 复用 regression_analysis.py 的数据加载逻辑
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from regression_analysis import (load_and_clean_data, construct_variables,
                                  filter_and_describe, compute_overpay,
                                  compute_power, get_industry_dummies,
                                  get_year_dummies, _build_fe_matrix)


def format_coef(coef, pval):
    """格式化系数：加星号"""
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


def generate_tables(df):
    """生成论文格式的回归结果表"""
    
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    
    # === 回归模型 ===
    # 模型3
    core3 = ["lnSubsidy"] + control_vars
    df3 = df.dropna(subset=core3 + ["Overpay", "IndustrySector"]).copy()
    X3, n_ind3, n_year3 = _build_fe_matrix(df3, core3)
    model3 = sm.OLS(df3["Overpay"], X3).fit(cov_type='HC1')
    
    # 模型4
    core4 = ["lnSubsidy", "Power"] + control_vars
    df4 = df.dropna(subset=core4 + ["Overpay", "IndustrySector"]).copy()
    X4, n_ind4, n_year4 = _build_fe_matrix(df4, core4)
    model4 = sm.OLS(df4["Overpay"], X4).fit(cov_type='HC1')
    
    # 模型5
    core5 = ["lnSubsidy"] + control_vars
    df5 = df.dropna(subset=core5 + ["Power", "IndustrySector"]).copy()
    X5, n_ind5, n_year5 = _build_fe_matrix(df5, core5)
    model5 = sm.OLS(df5["Power"], X5).fit(cov_type='HC1')
    
    # === 表2：主检验结果 ===
    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("表2  政府补助、管理层权力与高管超额薪酬的回归结果")
    lines.append("=" * 80)
    lines.append(f"{'变量':<15} {'模型(3)':<20} {'模型(4)':<20} {'模型(5)':<20}")
    lines.append(f"{'':15} {'因变量:Overpay':<20} {'因变量:Overpay':<20} {'因变量:Power':<20}")
    lines.append("-" * 80)
    
    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        
        for model in [model3, model4, model5]:
            if var in model.params:
                coef_line += f" {format_coef(model.params[var], model.pvalues[var]):<20}"
                tval_line += f" {format_tval(model.tvalues[var]):<20}"
            else:
                coef_line += f" {'':20}"
                tval_line += f" {'':20}"
        
        lines.append(coef_line)
        lines.append(tval_line)
    
    lines.append("-" * 80)
    lines.append(f"{'Industry':<15} {'控制':<20} {'控制':<20} {'控制':<20}")
    lines.append(f"{'Year':<15} {'控制':<20} {'控制':<20} {'控制':<20}")
    lines.append(f"{'N':<15} {len(df3):<20} {len(df4):<20} {len(df5):<20}")
    lines.append(f"{'R-squared':<15} {model3.rsquared:<20.4f} {model4.rsquared:<20.4f} {model5.rsquared:<20.4f}")
    lines.append("=" * 80)
    lines.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为异方差稳健标准误(HC1)。")
    
    table2 = "\n".join(lines)
    print(table2)
    
    # === 表3：分产权性质检验 ===
    lines2 = []
    lines2.append("")
    lines2.append("=" * 100)
    lines2.append("表3  分产权性质回归结果")
    lines2.append("=" * 100)
    lines2.append(f"{'变量':<15} {'国有-模型(3)':<20} {'国有-模型(4)':<20} {'非国有-模型(3)':<20} {'非国有-模型(4)':<20}")
    lines2.append(f"{'':15} {'Overpay':<20} {'Overpay':<20} {'Overpay':<20} {'Overpay':<20}")
    lines2.append("-" * 100)
    
    sub_models = {}
    for label, mask_val in [("soe", 1), ("non_soe", 0)]:
        df_sub = df[df["IsSOE"] == mask_val].copy()
        
        # 模型3
        core_no_p = ["lnSubsidy"] + control_vars
        df_s3 = df_sub.dropna(subset=core_no_p + ["Overpay", "IndustrySector"]).copy()
        X_s3, _, _ = _build_fe_matrix(df_s3, core_no_p)
        m3 = sm.OLS(df_s3["Overpay"], X_s3).fit(cov_type='HC1')
        
        # 模型4
        core_w_p = ["lnSubsidy", "Power"] + control_vars
        df_s4 = df_sub.dropna(subset=core_w_p + ["Overpay", "IndustrySector"]).copy()
        X_s4, _, _ = _build_fe_matrix(df_s4, core_w_p)
        m4 = sm.OLS(df_s4["Overpay"], X_s4).fit(cov_type='HC1')
        
        sub_models[f"{label}_3"] = (m3, len(df_s3))
        sub_models[f"{label}_4"] = (m4, len(df_s4))
    
    for var in display_vars:
        coef_line = f"{var:<15}"
        tval_line = f"{'':15}"
        for key in ["soe_3", "soe_4", "non_soe_3", "non_soe_4"]:
            m, _ = sub_models[key]
            if var in m.params:
                coef_line += f" {format_coef(m.params[var], m.pvalues[var]):<20}"
                tval_line += f" {format_tval(m.tvalues[var]):<20}"
            else:
                coef_line += f" {'':20}"
                tval_line += f" {'':20}"
        lines2.append(coef_line)
        lines2.append(tval_line)
    
    lines2.append("-" * 100)
    lines2.append(f"{'Industry':<15} {'控制':<20} {'控制':<20} {'控制':<20} {'控制':<20}")
    lines2.append(f"{'Year':<15} {'控制':<20} {'控制':<20} {'控制':<20} {'控制':<20}")
    
    n_line = f"{'N':<15}"
    r2_line = f"{'R-squared':<15}"
    for key in ["soe_3", "soe_4", "non_soe_3", "non_soe_4"]:
        m, n = sub_models[key]
        n_line += f" {n:<20}"
        r2_line += f" {m.rsquared:<20.4f}"
    lines2.append(n_line)
    lines2.append(r2_line)
    lines2.append("=" * 100)
    lines2.append("注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为异方差稳健标准误(HC1)。")
    
    table3 = "\n".join(lines2)
    print(table3)

    # === 表1: 描述性统计 ===
    lines0 = []
    lines0.append("")
    lines0.append("=" * 80)
    lines0.append("表1  主要变量描述性统计")
    lines0.append("=" * 80)
    
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
    lines0.append(desc_df.to_string(index=False))
    lines0.append("=" * 80)
    table1 = "\n".join(lines0)
    print(table1)
    
    # 保存所有表格
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    all_tables = table1 + "\n\n" + table2 + "\n\n" + table3
    
    with open(os.path.join(output_dir, "regression_tables.txt"), "w", encoding="utf-8") as f:
        f.write(all_tables)
    
    print(f"\n所有表格已保存至: results/regression_tables.txt")


def main():
    data_dir = os.path.join(os.getcwd(), "processed_data")
    
    print("加载数据...")
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    df = compute_overpay(df)
    df = compute_power(df)
    
    print("\n\n" + "#" * 80)
    print("#  以下为论文格式回归结果表格")
    print("#" * 80 + "\n")
    
    generate_tables(df)


if __name__ == "__main__":
    main()
