"""
生成论文格式 Word 回归结果表（三线表）
输出：results/回归结果表.docx
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml


def _set_run_font(run, font_name="Times New Roman", east_asia="宋体"):
    """安全设置 run 的中西文字体"""
    run.font.name = font_name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), east_asia)

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


# ============================================================
# 辅助函数
# ============================================================

def format_coef(coef, pval):
    """格式化系数，附加显著性星号"""
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


def fit_model(df, dep_var, core_vars, df_pre=None):
    """回归拟合"""
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


# ============================================================
# Word 三线表样式
# ============================================================

def set_cell_text(cell, text, bold=False, font_size=10.5, alignment=WD_ALIGN_PARAGRAPH.CENTER):
    """设置单元格文本和格式"""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = alignment
    run = p.add_run(str(text))
    run.font.size = Pt(font_size)
    _set_run_font(run, "Times New Roman", "宋体")
    run.bold = bold
    # 段落间距设为0
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = Pt(14)


def set_cell_border(cell, **kwargs):
    """设置单元格边框"""
    from docx.oxml import OxmlElement
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge, val in kwargs.items():
        element = OxmlElement(f'w:{edge}')
        element.set(qn('w:val'), val["val"])
        element.set(qn('w:sz'), val["sz"])
        element.set(qn('w:space'), "0")
        element.set(qn('w:color'), val.get("color", "000000"))
        tcBorders.append(element)
    tcPr.append(tcBorders)


def apply_three_line_style(table, header_rows=2):
    """对表格应用三线表样式：顶部粗线、表头下粗线、底部粗线"""
    # 先清除所有默认边框
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(f'<w:tblPr {nsdecls("w")}>')
    # 清除表格边框
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    n_rows = len(table.rows)
    n_cols = len(table.columns)
    thick = {"val": "single", "sz": "12"}  # 1.5pt
    thin = {"val": "single", "sz": "6"}    # 0.75pt

    # 顶部粗线（第 0 行的上边框）
    for col in range(n_cols):
        set_cell_border(table.cell(0, col), top=thick)

    # 表头下划线（第 header_rows-1 行的下边框）
    for col in range(n_cols):
        set_cell_border(table.cell(header_rows - 1, col), bottom=thin)

    # 底部粗线（最后一行的下边框）
    for col in range(n_cols):
        set_cell_border(table.cell(n_rows - 1, col), bottom=thick)


def add_table_title(doc, title):
    """添加表格标题"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.font.size = Pt(12)
    _set_run_font(run, "Times New Roman", "黑体")
    run.bold = True
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)


def add_table_note(doc, note):
    """添加表格注释"""
    p = doc.add_paragraph()
    run = p.add_run(note)
    run.font.size = Pt(9)
    _set_run_font(run, "Times New Roman", "宋体")
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)


# ============================================================
# 各表格构建
# ============================================================

def build_desc_table_docx(doc, df):
    """表1 描述性统计"""
    add_table_title(doc, "表1  主要变量描述性统计")

    desc_vars_map = {
        "Top3Salary": "高管前三名薪酬总额",
        "SubsidyAmount": "政府补助",
        "Overpay": "超额薪酬(Overpay)",
        "Power": "管理层权力(Power)",
        "Roa": "资产收益率(Roa)",
        "Revenue": "营业收入",
        "Lever": "财务杠杆(Lever)",
        "Top1": "第一大股东持股比例(Top1)",
    }

    headers = ["变量", "N", "均值", "标准差", "最小值", "最大值"]
    rows_data = []
    for var, label in desc_vars_map.items():
        if var in df.columns:
            s = df[var].dropna()
            def fmt(v):
                return f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2e}"
            rows_data.append([label, str(int(s.count())), fmt(s.mean()), fmt(s.std()), fmt(s.min()), fmt(s.max())])

    table = doc.add_table(rows=1 + len(rows_data), cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for i, h in enumerate(headers):
        set_cell_text(table.cell(0, i), h, bold=True)

    # 数据行
    for r, row in enumerate(rows_data):
        for c, val in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(table.cell(r + 1, c), val, alignment=align)

    apply_three_line_style(table, header_rows=1)
    add_table_note(doc, "注：薪酬和政府补助单位为元，连续变量已进行1%/99%缩尾处理。")


def _add_regression_table(doc, title, col_headers, sub_headers, display_vars, models_data, note, extra_rows=None):
    """
    通用回归结果表
    models_data: [(model_object, n_sample), ...]
    extra_rows: [(label, [values...]), ...] 额外行（如 Controls, FE）
    """
    add_table_title(doc, title)

    n_models = len(models_data)
    n_cols = 1 + n_models  # 变量列 + 各模型列

    # 计算行数：表头2行 + 每个变量2行(系数+t值) + 分隔行 + extra + N + R²
    n_extra = len(extra_rows) if extra_rows else 0
    n_data_rows = len(display_vars) * 2
    n_total = 2 + n_data_rows + n_extra + 2  # header(2) + vars + extra + N + R²

    table = doc.add_table(rows=n_total, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头行1
    set_cell_text(table.cell(0, 0), "变量", bold=True)
    for i, h in enumerate(col_headers):
        set_cell_text(table.cell(0, i + 1), h, bold=True)

    # 表头行2（因变量）
    set_cell_text(table.cell(1, 0), "", bold=True)
    for i, sh in enumerate(sub_headers):
        set_cell_text(table.cell(1, i + 1), sh, bold=True)

    # 变量行
    row_idx = 2
    for var in display_vars:
        # 系数行
        set_cell_text(table.cell(row_idx, 0), var, alignment=WD_ALIGN_PARAGRAPH.LEFT)
        for m_idx, (model, _) in enumerate(models_data):
            if model is not None and var in model.params:
                text = format_coef(model.params[var], model.pvalues[var])
            else:
                text = ""
            set_cell_text(table.cell(row_idx, m_idx + 1), text)
        row_idx += 1

        # t值行
        set_cell_text(table.cell(row_idx, 0), "")
        for m_idx, (model, _) in enumerate(models_data):
            if model is not None and var in model.params:
                text = format_tval(model.tvalues[var])
            else:
                text = ""
            set_cell_text(table.cell(row_idx, m_idx + 1), text)
        row_idx += 1

    # 额外行（Controls, FE 等）
    if extra_rows:
        for label, values in extra_rows:
            set_cell_text(table.cell(row_idx, 0), label, alignment=WD_ALIGN_PARAGRAPH.LEFT)
            for i, v in enumerate(values):
                set_cell_text(table.cell(row_idx, i + 1), v)
            row_idx += 1

    # N 行
    set_cell_text(table.cell(row_idx, 0), "N", alignment=WD_ALIGN_PARAGRAPH.LEFT)
    for i, (_, n) in enumerate(models_data):
        set_cell_text(table.cell(row_idx, i + 1), str(n))
    row_idx += 1

    # R² 行
    set_cell_text(table.cell(row_idx, 0), "R²", alignment=WD_ALIGN_PARAGRAPH.LEFT)
    for i, (model, _) in enumerate(models_data):
        r2 = f"{model.rsquared:.4f}" if model else "—"
        set_cell_text(table.cell(row_idx, i + 1), r2)

    apply_three_line_style(table, header_rows=2)
    add_table_note(doc, note)


def build_main_table_docx(doc, df):
    """表2 主回归（统一样本）"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    all_needed = ["lnSubsidy", "Power"] + control_vars + ["Overpay", "IndustrySector"]
    df_unified = df.dropna(subset=all_needed).copy()

    m3, n3 = fit_model(df, "Overpay", ["lnSubsidy"] + control_vars, df_pre=df_unified)
    m4, n4 = fit_model(df, "Overpay", ["lnSubsidy", "Power"] + control_vars, df_pre=df_unified)
    m5, n5 = fit_model(df, "Power", ["lnSubsidy"] + control_vars, df_pre=df_unified)

    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]
    fe = ["控制"] * 3

    _add_regression_table(
        doc,
        "表2  政府补助、管理层权力与高管超额薪酬的回归结果",
        ["模型(3)", "模型(4)", "模型(5)"],
        ["Overpay", "Overpay", "Power"],
        display_vars,
        [(m3, n3), (m4, n4), (m5, n5)],
        "注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。",
        extra_rows=[
            ("Industry FE", fe),
            ("Year FE", fe),
        ],
    )


def build_subsample_table_docx(doc, df, title, group_specs):
    """分组回归表（统一样本）"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    display_vars = ["lnSubsidy", "Power", "Roa", "Lever", "Top1", "Zone"]

    models_data = []
    col_headers = []
    sub_headers = []
    for label, mask in group_specs:
        df_sub = df[mask].copy()
        all_needed = ["lnSubsidy", "Power"] + control_vars + ["Overpay", "IndustrySector"]
        df_sub_unified = df_sub.dropna(subset=all_needed).copy()
        m3, n3 = fit_model(df_sub, "Overpay", ["lnSubsidy"] + control_vars, df_pre=df_sub_unified)
        m4, n4 = fit_model(df_sub, "Overpay", ["lnSubsidy", "Power"] + control_vars, df_pre=df_sub_unified)
        models_data.extend([(m3, n3), (m4, n4)])
        col_headers.extend([f"{label}-模型(3)", f"{label}-模型(4)"])
        sub_headers.extend(["Overpay", "Overpay"])

    fe = ["控制"] * len(models_data)

    _add_regression_table(
        doc, title, col_headers, sub_headers, display_vars, models_data,
        "注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。",
        extra_rows=[("Industry FE", fe), ("Year FE", fe)],
    )


def build_robustness_table_docx(doc, df):
    """表6 稳健性检验"""
    control_vars = ["Roa", "Lever", "Top1", "Zone"]
    core_vars = ["lnSubsidy"] + control_vars

    checks = []

    # (1)
    m1, n1 = fit_model(df, "lnCEOpay", core_vars)
    checks.append(("(1) 替换因变量", "lnCEOpay", m1, n1, "lnSubsidy"))

    # (2)
    df_r2 = df[(df["Year"] >= 2010) & (df["Year"] <= 2020)].copy()
    m2, n2 = fit_model(df_r2, "Overpay", core_vars)
    checks.append(("(2) 缩小样本期", "Overpay", m2, n2, "lnSubsidy"))

    # (3)
    q05 = df["lnSubsidy"].quantile(0.05)
    q95 = df["lnSubsidy"].quantile(0.95)
    df_r3 = df[(df["lnSubsidy"] >= q05) & (df["lnSubsidy"] <= q95)].copy()
    m3, n3 = fit_model(df_r3, "Overpay", core_vars)
    checks.append(("(3) 剔除极端补助", "Overpay", m3, n3, "lnSubsidy"))

    # (4)
    df_r4 = df[df["Industry"] == 1].copy()
    m4, n4 = fit_model(df_r4, "Overpay", core_vars)
    checks.append(("(4) 仅制造业", "Overpay", m4, n4, "lnSubsidy"))

    # (5)
    core_alt = ["lnSubsidy1p"] + control_vars
    m5, n5 = fit_model(df, "Overpay", core_alt)
    checks.append(("(5) 替换解释变量", "Overpay", m5, n5, "lnSubsidy1p"))

    # (6)
    df_lag = df.sort_values(["Symbol", "Year"]).copy()
    df_lag["lnSubsidy_l1"] = df_lag.groupby("Symbol")["lnSubsidy"].shift(1)
    core_lag = ["lnSubsidy_l1"] + control_vars
    m6, n6 = fit_model(df_lag, "Overpay", core_lag)
    checks.append(("(6) 滞后一期", "Overpay", m6, n6, "lnSubsidy_l1"))

    # 构建表格
    add_table_title(doc, "表6  稳健性检验结果")
    n_checks = len(checks)
    # 行：表头2 + 系数1 + t值1 + Controls + Industry FE + Year FE + N + R² = 9
    table = doc.add_table(rows=9, cols=1 + n_checks)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头行1
    set_cell_text(table.cell(0, 0), "变量", bold=True)
    for i, (label, _, _, _, _) in enumerate(checks):
        set_cell_text(table.cell(0, i + 1), label, bold=True, font_size=9)

    # 表头行2 (因变量)
    set_cell_text(table.cell(1, 0), "", bold=True)
    for i, (_, dep, _, _, _) in enumerate(checks):
        set_cell_text(table.cell(1, i + 1), dep, bold=True, font_size=9)

    # 系数行
    set_cell_text(table.cell(2, 0), "补助变量", alignment=WD_ALIGN_PARAGRAPH.LEFT)
    for i, (_, _, model, _, var_name) in enumerate(checks):
        if model and var_name in model.params:
            set_cell_text(table.cell(2, i + 1), format_coef(model.params[var_name], model.pvalues[var_name]))

    # t值行
    set_cell_text(table.cell(3, 0), "")
    for i, (_, _, model, _, var_name) in enumerate(checks):
        if model and var_name in model.params:
            set_cell_text(table.cell(3, i + 1), format_tval(model.tvalues[var_name]))

    # Controls, FE, N, R²
    row_labels = ["Controls", "Industry FE", "Year FE", "N", "R²"]
    for r, label in enumerate(row_labels):
        set_cell_text(table.cell(4 + r, 0), label, alignment=WD_ALIGN_PARAGRAPH.LEFT)
        for i, (_, _, model, n, _) in enumerate(checks):
            if label == "N":
                set_cell_text(table.cell(4 + r, i + 1), str(n))
            elif label == "R²":
                set_cell_text(table.cell(4 + r, i + 1), f"{model.rsquared:.4f}" if model else "—")
            else:
                set_cell_text(table.cell(4 + r, i + 1), "控制")

    apply_three_line_style(table, header_rows=2)
    add_table_note(doc, "注：括号中为 t 值，* p<0.1, ** p<0.05, *** p<0.01。标准误为公司层面聚类稳健标准误。")


# ============================================================
# 主流程
# ============================================================

def main():
    data_dir = os.path.join(os.getcwd(), "processed_data")
    print("加载数据...")
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    df = compute_overpay(df)
    df = compute_power(df)

    print("\n生成 Word 文档...")
    doc = Document()

    # 设置默认字体
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(10.5)
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), '宋体')

    # 文档标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("回归分析结果")
    run.font.size = Pt(16)
    _set_run_font(run, "Times New Roman", "黑体")
    run.bold = True

    # 副标题
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("基于数据挖掘的上市公司财政补贴与高管超额薪酬研究")
    run.font.size = Pt(12)
    _set_run_font(run, "Times New Roman", "宋体")

    # 逐表生成
    print("  生成表1 描述性统计...")
    build_desc_table_docx(doc, df)

    print("  生成表2 主回归...")
    build_main_table_docx(doc, df)

    soe_df = df[df["IsSOE"] == 1].copy()
    identified_ratio = soe_df["IsCentralSOE"].notna().mean() if len(soe_df) > 0 else np.nan

    print("  生成表3 分产权性质...")
    build_subsample_table_docx(
        doc, df,
        "表3  分产权性质回归结果",
        [("国有", df["IsSOE"] == 1), ("非国有", df["IsSOE"] == 0)],
    )

    print("  生成表4 分行业管制...")
    build_subsample_table_docx(
        doc, df,
        "表4  分行业管制回归结果",
        [("管制行业", df["RegulatedIndustry"] == 1), ("非管制行业", df["RegulatedIndustry"] == 0)],
    )

    print("  生成表5 央企vs地方国企...")
    build_subsample_table_docx(
        doc, df,
        f"表5  央企与地方国企回归结果（央地可识别比例={identified_ratio:.2%}）",
        [("央企", df["IsCentralSOE"] == 1), ("地方国企", df["IsCentralSOE"] == 0)],
    )

    print("  生成表6 稳健性检验...")
    build_robustness_table_docx(doc, df)

    # 保存
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "回归结果表.docx")
    doc.save(output_file)
    print(f"\n✅ Word 文件已保存至: {output_file}")


if __name__ == "__main__":
    main()
