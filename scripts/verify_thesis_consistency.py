"""
校验论文正文中的核心实证数据是否与当前 results/ 目录中的最新结果一一对应。
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
THESIS_PATH = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.md"
RESULTS_DIR = ROOT_DIR / "results"


def read_csv_rows(filename: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / filename).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_map(filename: str, key: str) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(filename)
    return {row[key]: row for row in rows}


def read_json(filename: str) -> dict:
    return json.loads((RESULTS_DIR / filename).read_text(encoding="utf-8"))


def fmt_int(value: str | int | float) -> str:
    return f"{int(round(float(value))):,}"


def fmt_fixed(value: str | int | float, digits: int = 4, unicode_minus: bool = True) -> str:
    number = float(value)
    if abs(number) < 0.5 * (10 ** (-digits)):
        number = 0.0
    prefix = "−" if unicode_minus and number < 0 else "-" if number < 0 else ""
    return f"{prefix}{abs(number):.{digits}f}"


def fmt_plain(value: str | int | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_t(value: str | int | float, digits: int = 2) -> str:
    return fmt_fixed(value, digits=digits)


def fmt_p(value: str | int | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: str | int | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}%"


def fmt_coef(value: str | int | float, p_value: str | int | float, digits: int = 4) -> str:
    coef = fmt_plain(value, digits)
    p = float(p_value)
    stars = ""
    if p < 0.01:
        stars = "***"
    elif p < 0.05:
        stars = "**"
    elif p < 0.1:
        stars = "*"
    return f"{coef}{stars}"


def fmt_scientific(value: str | int | float, digits: int = 2) -> str:
    number = float(value)
    if number == 0:
        return f"0.{'0' * digits}×10^0"
    mantissa, exp = f"{number:.{digits}e}".split("e")
    exponent = int(exp)
    return f"{mantissa}×10^{exponent}"


def expect(text: str, needle: str, label: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"[缺失] {label}: {needle}")


def expect_absent(text: str, needle: str, label: str, failures: list[str]) -> None:
    if needle in text:
        failures.append(f"[残留] {label}: {needle}")


def normalize_tables(text: str) -> str:
    """把 HTML 表格中的行提取成统一的管道文本，便于一致性校验兼容 Markdown/HTML 两种写法。"""
    rows: list[str] = []
    for tr_html in re.findall(r"<tr>(.*?)</tr>", text, flags=re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh]>(.*?)</t[dh]>", tr_html, flags=re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        cleaned = []
        for cell in cells:
            plain = re.sub(r"<.*?>", "", cell, flags=re.DOTALL)
            plain = html.unescape(re.sub(r"\s+", " ", plain)).strip()
            cleaned.append(plain)
        rows.append(f"| {' | '.join(cleaned)} |")
    return text + "\n" + "\n".join(rows)


def check_st_rows(failures: list[str]) -> None:
    prefix = re.compile(r"^(?:[A-Z]+)?(?:S\*ST|\*ST|SST|PT|ST)", re.IGNORECASE)
    suffix = re.compile(r"(?:退市整理|退市|退)$", re.IGNORECASE)
    with (RESULTS_DIR / "regression_dataset.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("ShortName") or "").replace(" ", "").replace("\u3000", "")
            if name and (prefix.search(name) or suffix.search(name)):
                failures.append(
                    f"[样本残留] regression_dataset.csv 仍包含特殊处理样本: "
                    f"{row.get('Symbol')} {row.get('ShortName')} {row.get('Year')}"
                )
                break


def main() -> int:
    thesis_raw = THESIS_PATH.read_text(encoding="utf-8")
    thesis = normalize_tables(thesis_raw)
    failures: list[str] = []

    sample_map = read_csv_map("sample_screening_summary.csv", "stage")
    desc_map = read_csv_map("descriptive_statistics.csv", "变量")
    vif_rows = read_csv_rows("vif_results.csv")
    vif_map = {row["变量"]: row for row in vif_rows}
    causal_rows = read_csv_rows("causal_results.csv")
    causal_map = {(row["category"], row["model"]): row for row in causal_rows}
    robust_map = read_csv_map("robustness_results.csv", "check_item")
    own_map = read_csv_map("heterogeneity_ownership_results.csv", "group")
    mech_rows = read_csv_rows("heterogeneity_mechanism_results.csv")
    mech_map = {(row["group_type"], row["group"]): row for row in mech_rows}
    model_map = read_csv_map("model_comparison.csv", "模型")
    clf_map = read_csv_map("classification_comparison.csv", "模型")
    ml_summary = read_json("ml_summary.json")
    ml_tuning = read_json("ml_tuning_summary.json")
    diagnostics = read_json("diagnostics_summary.json")
    shap_rows = read_csv_rows("shap_importance.csv")
    power_fa = read_csv_rows("power_fa_diagnostics.csv")[0]
    stage1_map = read_csv_map("stage1_results.csv", "key_var")

    sample_chain = (
        f"原始公司—年度观测为{fmt_int(sample_map['原始公司-年观测']['observations'])}条；"
        f"剔除金融行业后为{fmt_int(sample_map['剔除金融行业后']['observations'])}条；"
        f"再剔除特殊处理样本后为{fmt_int(sample_map['剔除特殊处理样本后']['observations'])}条；"
        f"满足关键变量完整要求的样本为{fmt_int(sample_map['关键变量完整样本']['observations'])}条；"
        f"满足期望薪酬模型完整案例要求的样本为{fmt_int(sample_map['期望薪酬模型完整案例']['observations'])}条；"
        f"纳入管理层权力底层指标后，可用于构造 Power 的样本为{fmt_int(sample_map['可用于构造 Power 的样本']['observations'])}条；"
        f"在基准回归中加入滞后一期补助变量后，模型2样本为{fmt_int(sample_map['模型2主回归样本']['observations'])}条；"
        f"在中介效应检验中要求 Overpay、Power 与滞后补助均完整后，统一样本为{fmt_int(sample_map['中介统一样本']['observations'])}条；"
        f"机器学习部分在进一步要求 `IsSOE` 等特征完整后，可用样本为{fmt_int(ml_summary['ml_sample_size'])}条。"
    )
    expect(thesis, sample_chain, "第三章样本筛选链", failures)

    expect(
        thesis,
        f"整体 KMO 约为{fmt_plain(diagnostics['power_fa']['kmo_overall'], 3)}，前两个公共因子累计方差解释率约为{float(diagnostics['power_fa']['variance_explained_ratio']) * 100:.2f}%",
        "FA 统计支撑说明",
        failures,
    )
    expect(
        thesis,
        f"最大值约为{fmt_plain(diagnostics['vif']['max'], 2)}，均值约为{fmt_plain(diagnostics['vif']['mean'], 2)}",
        "VIF 诊断说明",
        failures,
    )
    expect(
        thesis_raw,
        "期望薪酬模型的残差项 $\\varepsilon_{it}$ 直接定义为 $\\text{Overpay}_{it}$",
        "Overpay 残差定义口径",
        failures,
    )
    expect_absent(thesis_raw, "\\widehat{\\ln\\text{Salary}}_{it}", "Overpay 二次减法公式残留", failures)
    expect_absent(thesis_raw, "并结合PCA、熵值法对照结果观察机制结论对测度口径的敏感性", "PCA/熵值法正文主线残留", failures)
    expect_absent(thesis_raw, "基于FA的五维综合得分", "Power 五维定义残留", failures)
    expect_absent(thesis_raw, "PCA 与熵值法仅作对照", "熵值法对照残留", failures)

    desc_rows = [
        f"| 高管前三名薪酬总额（元） | {fmt_int(desc_map['高管前三名薪酬总额']['N'])} | {fmt_scientific(desc_map['高管前三名薪酬总额']['均值'])} | {fmt_scientific(desc_map['高管前三名薪酬总额']['中位数'])} | {fmt_scientific(desc_map['高管前三名薪酬总额']['标准差'])} | {float(desc_map['高管前三名薪酬总额']['最小值']):,.2f} | {fmt_scientific(desc_map['高管前三名薪酬总额']['最大值'])} |",
        f"| 政府补助（元） | {fmt_int(desc_map['政府补助']['N'])} | {fmt_scientific(desc_map['政府补助']['均值'])} | {fmt_scientific(desc_map['政府补助']['中位数'])} | {fmt_scientific(desc_map['政府补助']['标准差'])} | {fmt_scientific(desc_map['政府补助']['最小值'])} | {fmt_scientific(desc_map['政府补助']['最大值'])} |",
        f"| 财政补贴强度（lnSubsidy=ln(1+Subsidy)） | {fmt_int(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['N'])} | {fmt_plain(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['均值'])} | {fmt_plain(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['中位数'])} | {fmt_plain(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['标准差'])} | {fmt_plain(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['最小值'])} | {fmt_plain(desc_map['财政补贴强度(lnSubsidy=ln(1+Subsidy))']['最大值'])} |",
        f"| 高管前三名薪酬对数 | {fmt_int(desc_map['高管前三名薪酬对数']['N'])} | {fmt_plain(desc_map['高管前三名薪酬对数']['均值'])} | {fmt_plain(desc_map['高管前三名薪酬对数']['中位数'])} | {fmt_plain(desc_map['高管前三名薪酬对数']['标准差'])} | {fmt_plain(desc_map['高管前三名薪酬对数']['最小值'])} | {fmt_plain(desc_map['高管前三名薪酬对数']['最大值'])} |",
        f"| 企业规模（lnSale） | {fmt_int(desc_map['企业规模(lnSale)']['N'])} | {fmt_plain(desc_map['企业规模(lnSale)']['均值'])} | {fmt_plain(desc_map['企业规模(lnSale)']['中位数'])} | {fmt_plain(desc_map['企业规模(lnSale)']['标准差'])} | {fmt_plain(desc_map['企业规模(lnSale)']['最小值'])} | {fmt_plain(desc_map['企业规模(lnSale)']['最大值'])} |",
        f"| 无形资产占比（IA） | {fmt_int(desc_map['无形资产占比(IA)']['N'])} | {fmt_plain(desc_map['无形资产占比(IA)']['均值'])} | {fmt_plain(desc_map['无形资产占比(IA)']['中位数'])} | {fmt_plain(desc_map['无形资产占比(IA)']['标准差'])} | {fmt_plain(desc_map['无形资产占比(IA)']['最小值'])} | {fmt_plain(desc_map['无形资产占比(IA)']['最大值'])} |",
        f"| 超额薪酬（Overpay） | {fmt_int(desc_map['超额薪酬(Overpay)']['N'])} | {fmt_fixed(desc_map['超额薪酬(Overpay)']['均值'])} | {fmt_fixed(desc_map['超额薪酬(Overpay)']['中位数'])} | {fmt_plain(desc_map['超额薪酬(Overpay)']['标准差'])} | {fmt_fixed(desc_map['超额薪酬(Overpay)']['最小值'])} | {fmt_plain(desc_map['超额薪酬(Overpay)']['最大值'])} |",
        f"| 管理层权力（Power，FA） | {fmt_int(desc_map['管理层权力(Power, FA口径)']['N'])} | {fmt_fixed(desc_map['管理层权力(Power, FA口径)']['均值'])} | {fmt_plain(desc_map['管理层权力(Power, FA口径)']['中位数'])} | {fmt_plain(desc_map['管理层权力(Power, FA口径)']['标准差'])} | {fmt_fixed(desc_map['管理层权力(Power, FA口径)']['最小值'])} | {fmt_plain(desc_map['管理层权力(Power, FA口径)']['最大值'])} |",
    ]
    for idx, row in enumerate(desc_rows, start=1):
        expect(thesis, row, f"描述性统计行{idx}", failures)

    expect(thesis, f"| lnSubsidy | {fmt_plain(vif_map['lnSubsidy']['VIF'])} |", "VIF 行 lnSubsidy", failures)
    expect(thesis, f"| Lever | {fmt_plain(vif_map['Lever']['VIF'])} |", "VIF 行 Lever", failures)
    expect(thesis, f"均值约为{fmt_plain(diagnostics['vif']['mean'], 2)}", "VIF 均值说明", failures)

    for var in ["lnSale", "Roa", "IA", "Zone"]:
        coef = f"{fmt_fixed(stage1_map[var]['coef'])}***"
        expect(thesis, f"| {var} | {coef} |", f"模型1行 {var}", failures)
    expect(thesis, f"| N | {fmt_int(diagnostics['stage1']['sample_size'])} | — |", "模型1样本量", failures)
    expect(thesis, f"| R² | {fmt_plain(diagnostics['stage1']['r2'])} | — |", "模型1R²", failures)
    expect(thesis, f"| 调整后R² | {fmt_plain(diagnostics['stage1']['adj_r2'])} | — |", "模型1调整后R²", failures)
    expect(thesis, f"| F统计量 | {fmt_plain(diagnostics['stage1']['f_stat'], 2)}*** | 整体模型在1%水平上显著 |", "模型1F统计量", failures)
    expect(thesis_raw, "BP 异方差检验显著拒绝同方差原假设（p < 0.001）", "BP检验说明", failures)
    expect(thesis_raw, "Wooldridge 面板序列相关检验也显著（p < 0.001）", "Wooldridge检验说明", failures)

    model2 = causal_map[("双向固定效应", "模型2-主回归")]
    expect(thesis, f"| lnSubsidy_l1 | {fmt_plain(model2['coef'])}（{fmt_t(diagnostics['model2']['t_stat'], 2)}） |", "模型2核心系数行", failures)
    expect(thesis, f"| N | {fmt_int(model2['n'])} |", "模型2样本量", failures)
    expect(thesis, f"| R² | {fmt_plain(model2['r2'])} |", "模型2R²", failures)
    expect(thesis, f"| F统计量 | {fmt_plain(diagnostics['model2']['f_stat'], 2)}*** |", "模型2F统计量", failures)

    iv_map = read_csv_map("iv_comparison_results.csv", "role")
    heckman = read_csv_rows("heckman_results.csv")[0]
    iv_benchmark = iv_map["benchmark"]
    iv_simple = iv_map["simple"]
    iv_refined = iv_map["refined"]
    expect(
        thesis,
        f"| 基准工具变量 | 滞后一期同城同年其他企业平均补助 | Partial F = {fmt_plain(iv_benchmark['partial_f'], 2)} | {fmt_fixed(iv_benchmark['second_stage_coef'])}（t = {fmt_fixed(iv_benchmark['second_stage_t'], 4)}） | {fmt_int(iv_benchmark['sample_size'])} |",
        "IV 基准工具行",
        failures,
    )
    expect(
        thesis,
        f"| 简单替代工具变量 | 滞后一期同行业同年其他企业平均补助 | Partial F = {fmt_plain(iv_simple['partial_f'], 2)} | {fmt_fixed(iv_simple['second_stage_coef'])}（t = {fmt_fixed(iv_simple['second_stage_t'], 4)}） | {fmt_int(iv_simple['sample_size'])} |",
        "IV 简单替代工具行",
        failures,
    )
    expect(
        thesis,
        f"| 精炼双工具变量 | 滞后一期同行业同年排除本省平均补助 + 同省同行业同年排除本市平均补助 | Partial F = {fmt_plain(iv_refined['partial_f'], 2)}；OverID p = {fmt_plain(iv_refined['overid_p'], 3)} | {fmt_fixed(iv_refined['second_stage_coef'])}**（t = {fmt_fixed(iv_refined['second_stage_t'], 4)}） | {fmt_int(iv_refined['sample_size'])} |",
        "IV 精炼双工具行",
        failures,
    )
    expect(
        thesis,
        f"| Heckman 两步法 | 选择方程排除变量：IV_lnSubsidy_l1 | IMR p = {fmt_plain(heckman['imr_p'], 4)} | {fmt_coef(heckman['subsidy_coef'], heckman['subsidy_p'])}（t = {fmt_plain(heckman['subsidy_t'], 4)}） | {fmt_int(heckman['outcome_sample_size'])} |",
        "Heckman 结果行",
        failures,
    )

    med = read_csv_rows("main_mediation_summary.csv")[0]
    med_c = causal_map[("中介效应FA", "模型3-总效应")]
    med_a = causal_map[("中介效应FA", "模型4-路径a")]
    med_b = causal_map[("中介效应FA", "模型5-路径b")]
    med_c_prime = causal_map[("中介效应FA", "模型5-直接效应")]
    expect(thesis, f"| 模型3 总效应 c：lnSubsidy_l1 → Overpay | {fmt_coef(med['coef_c'], med['p_c'])} | {fmt_plain(med_c['se'])} | — |", "中介总效应行", failures)
    expect(thesis, f"| 模型4 路径 a：lnSubsidy_l1 → Power（FA） | {fmt_coef(med['coef_a'], med['p_a'])} | {fmt_plain(med_a['se'])} | — |", "中介路径a行", failures)
    expect(thesis, f"| 模型5 路径 b：Power（FA） → Overpay | {fmt_coef(med['coef_b'], med['p_b'])} | {fmt_plain(med_b['se'])} | — |", "中介路径b行", failures)
    expect(thesis, f"| 模型5 直接效应 c'：lnSubsidy_l1 → Overpay | {fmt_coef(med['coef_c_prime'], med['p_c_prime'])} | {fmt_plain(med_c_prime['se'])} | — |", "中介直接效应行", failures)
    expect(thesis, f"| 间接效应 a×b | {float(med['indirect_effect']):.6f} | — | [{float(med['bootstrap_ci_lower']):.6f}, {float(med['bootstrap_ci_upper']):.6f}] |", "中介间接效应行", failures)
    expect(thesis, f"| Sobel p 值 | {float(med['sobel_p']):.4f} | — | — |", "Sobel p 值行", failures)
    expect(thesis, f"| 中介效应占比（a×b/c） | {float(med['mediation_ratio_pct']):.2f}% | — | — |", "中介占比行", failures)

    expect(thesis, f"| (1) 替换因变量 | 高管前三名薪酬对数 | {fmt_plain(robust_map['替换因变量']['coef'])}*** | {fmt_plain(robust_map['替换因变量']['t_value'], 4)} | {fmt_int(robust_map['替换因变量']['sample_size'])} | {fmt_plain(robust_map['替换因变量']['r_squared'])} |", "稳健性1", failures)
    expect(thesis, f"| (2) 缩小样本期（2010—2020） | Overpay | {fmt_plain(robust_map['缩小样本期(2010-2020)']['coef'])} | {fmt_plain(robust_map['缩小样本期(2010-2020)']['t_value'], 4)} | {fmt_int(robust_map['缩小样本期(2010-2020)']['sample_size'])} | {fmt_plain(robust_map['缩小样本期(2010-2020)']['r_squared'])} |", "稳健性2", failures)
    expect(thesis, f"| (3) 仅制造业 | Overpay | {fmt_fixed(robust_map['仅制造业']['coef'])} | {fmt_fixed(robust_map['仅制造业']['t_value'], 4)} | {fmt_int(robust_map['仅制造业']['sample_size'])} | {fmt_plain(robust_map['仅制造业']['r_squared'])} |", "稳健性3", failures)
    expect(thesis, f"| (4) 替换解释变量ln(补助，仅正值) | Overpay | {fmt_plain(robust_map['替换解释变量ln(补助，仅正值)']['coef'])}*** | {fmt_plain(robust_map['替换解释变量ln(补助，仅正值)']['t_value'], 4)} | {fmt_int(robust_map['替换解释变量ln(补助，仅正值)']['sample_size'])} | {fmt_plain(robust_map['替换解释变量ln(补助，仅正值)']['r_squared'])} |", "稳健性4", failures)

    expect(thesis, "| lnSubsidy_l1 | 0.0000（0.02） | −0.0002（−0.11） |", "产权异质性核心行", failures)
    expect(thesis, f"| N | {fmt_int(own_map['国有企业']['sample_size'])} | {fmt_int(own_map['私营企业']['sample_size'])} |", "产权异质性样本量", failures)
    expect(thesis, f"| R² | {fmt_plain(own_map['国有企业']['r2_model2'])} | {fmt_plain(own_map['私营企业']['r2_model2'])} |", "产权异质性R²", failures)
    expect(thesis, "| lnSubsidy_l1 | −0.0006（−0.36） | 0.0008（0.79） |", "行业异质性核心行", failures)
    expect(thesis, f"| N | {fmt_int(mech_map[('行业管制', '管制行业')]['sample_size'])} | {fmt_int(mech_map[('行业管制', '非管制行业')]['sample_size'])} |", "行业异质性样本量", failures)
    expect(thesis, "| lnSubsidy_l1 | −0.0006（−0.33） | −0.0012（−0.88） |", "央地异质性核心行", failures)
    expect(thesis, "央地可识别比例为74.53%", "央地可识别比例", failures)

    expect(thesis, f"机器学习部分使用{fmt_int(ml_summary['ml_sample_size'])}个样本", "ML 样本量", failures)
    expect(
        thesis,
        f"训练集（{fmt_int(ml_tuning['split']['train_size'])}条）和测试集（{fmt_int(ml_tuning['split']['test_size_n'])}条）",
        "ML 训练/测试样本量",
        failures,
    )
    expect(thesis, f"正类{fmt_int(ml_summary['class_balance']['positive_count'])}个、负类{fmt_int(ml_summary['class_balance']['negative_count'])}个，比例约为{float(ml_summary['class_balance']['positive_ratio']) * 100:.2f}%", "ML 类别分布", failures)
    expect(thesis, f"最优惩罚参数为 $\\lambda^*=0.0001$", "Lasso alpha", failures)
    expect(thesis, "`lnSubsidy`、`Roa` 和 `Mgshder` 位于最重要的几类特征之中", "SHAP 排序描述", failures)
    expect(thesis, "在回归设定中，`Roa`、`Top1`、`Mgshder` 和 `lnSubsidy` 均位于前列", "RF 回归重要性描述", failures)
    expect(thesis, f"| OLS | {fmt_plain(model_map['OLS']['测试集R²'])} | {fmt_plain(model_map['OLS']['RMSE'])} | {fmt_plain(model_map['OLS']['MAE'])} | {fmt_plain(model_map['OLS']['5折CV R²均值'])} ± {fmt_plain(model_map['OLS']['5折CV R²标准差'])} |", "ML 回归表 OLS", failures)
    expect(thesis, f"| Lasso | {fmt_plain(model_map['Lasso']['测试集R²'])} | {fmt_plain(model_map['Lasso']['RMSE'])} | {fmt_plain(model_map['Lasso']['MAE'])} | {fmt_plain(model_map['Lasso']['5折CV R²均值'])} ± {fmt_plain(model_map['Lasso']['5折CV R²标准差'])} |", "ML 回归表 Lasso", failures)
    expect(thesis, f"| 随机森林 | {fmt_plain(model_map['RandomForest']['测试集R²'])} | {fmt_plain(model_map['RandomForest']['RMSE'])} | {fmt_plain(model_map['RandomForest']['MAE'])} | {fmt_plain(model_map['RandomForest']['5折CV R²均值'])} ± {fmt_plain(model_map['RandomForest']['5折CV R²标准差'])} |", "ML 回归表 RF", failures)
    expect(thesis, f"| XGBoost | {fmt_plain(model_map['XGBoost']['测试集R²'])} | {fmt_plain(model_map['XGBoost']['RMSE'])} | {fmt_plain(model_map['XGBoost']['MAE'])} | {fmt_plain(model_map['XGBoost']['5折CV R²均值'])} ± {fmt_plain(model_map['XGBoost']['5折CV R²标准差'])} |", "ML 回归表 XGB", failures)
    expect(thesis, f"| Logit | {fmt_plain(clf_map['Logit']['Accuracy'])} | {fmt_plain(clf_map['Logit']['Precision'])} | {fmt_plain(clf_map['Logit']['Recall'])} | {fmt_plain(clf_map['Logit']['F1'])} | {fmt_plain(clf_map['Logit']['ROC_AUC'])} | {fmt_plain(clf_map['Logit']['5折CV ROC_AUC'])} |", "ML 分类表 Logit", failures)
    expect(thesis, f"| 随机森林分类器 | {fmt_plain(clf_map['RandomForestClassifier']['Accuracy'])} | {fmt_plain(clf_map['RandomForestClassifier']['Precision'])} | {fmt_plain(clf_map['RandomForestClassifier']['Recall'])} | {fmt_plain(clf_map['RandomForestClassifier']['F1'])} | {fmt_plain(clf_map['RandomForestClassifier']['ROC_AUC'])} | {fmt_plain(clf_map['RandomForestClassifier']['5折CV ROC_AUC'])} |", "ML 分类表 RF", failures)
    expect(thesis, f"| XGBoost分类器 | {fmt_plain(clf_map['XGBoostClassifier']['Accuracy'])} | {fmt_plain(clf_map['XGBoostClassifier']['Precision'])} | {fmt_plain(clf_map['XGBoostClassifier']['Recall'])} | {fmt_plain(clf_map['XGBoostClassifier']['F1'])} | {fmt_plain(clf_map['XGBoostClassifier']['ROC_AUC'])} | {fmt_plain(clf_map['XGBoostClassifier']['5折CV ROC_AUC'])} |", "ML 分类表 XGB", failures)
    expect(thesis, f"| 决策树 | {fmt_plain(clf_map['DecisionTree']['Accuracy'])} | {fmt_plain(clf_map['DecisionTree']['Precision'])} | {fmt_plain(clf_map['DecisionTree']['Recall'])} | {fmt_plain(clf_map['DecisionTree']['F1'])} | {fmt_plain(clf_map['DecisionTree']['ROC_AUC'])} | {fmt_plain(clf_map['DecisionTree']['5折CV ROC_AUC'])} |", "ML 分类表 决策树", failures)

    expect(thesis, f"（$\\beta = {fmt_plain(model2['coef'])}$，$p = {fmt_p(model2['p'])}$）", "摘要主回归值", failures)
    expect(thesis, f"精炼双工具后，第二阶段系数转为显著负值（$\\beta = {fmt_fixed(iv_refined['second_stage_coef'], 4, unicode_minus=False)}$，$p = {fmt_p(iv_refined['second_stage_p'])}$）", "摘要精炼双工具", failures)
    expect(thesis, f"针对仅正补助样本的 Heckman 两步校正显示逆米尔斯比率显著（$p = {fmt_p(heckman['imr_p'])}$），而补贴系数仍为正且显著（$\\beta = {fmt_plain(heckman['subsidy_coef'])}$，$p = {fmt_p(heckman['subsidy_p'])}$）", "摘要Heckman", failures)
    expect(thesis, f"路径$a$虽为正，但未达到常用显著性水平（$p = {fmt_p(med_a['p'])}$）", "摘要路径a", failures)
    expect(thesis, f"间接效应约为{float(med['indirect_effect']):.6f}，Bootstrap 95%置信区间为[{float(med['bootstrap_ci_lower']):.6f}, {float(med['bootstrap_ci_upper']):.6f}]", "摘要间接效应", failures)
    expect(thesis, f"XGBoost 回归的测试集 $R^2$ 仅为{fmt_plain(model_map['XGBoost']['测试集R²'])}，分类模型的 ROC-AUC 约为{fmt_plain(clf_map['XGBoostClassifier']['ROC_AUC'], 3)}", "结论机器学习值", failures)

    stale_values = [
        "45,651", "45,659", "42,572", "36,340", "9,311", "22,457", "23,194", "49.19%",
        "74.88%", "0.1037", "0.6228", "0.138", "1.52", "1.22",
        "0.126", "0.1651", "0.000053", "7.06%", "基于FA的五维综合得分",
        "表4-6 工具变量（FE-2SLS）估计结果", "Partial R² 为0.0019，说明工具变量具有统计相关性但解释力度有限。",
        "data:image/png;base64", "�",
    ]
    for value in stale_values:
        expect_absent(thesis_raw, value, "旧值或污染残留", failures)

    check_st_rows(failures)

    if failures:
        print("一致性校验未通过：", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("一致性校验通过：正文中的核心实证数据与当前 results/ 最新结果一一对应。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
