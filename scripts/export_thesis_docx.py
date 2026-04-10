"""
使用学校模板骨架 + Pandoc 预处理 + docxcompose 拼装，
稳定导出更符合学院模板要求的论文 Word 文档。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docxcompose.composer import Composer
from lxml import etree, html as lxml_html


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.md"
PREPROCESSED_MD = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2_pandoc.md"
OUTPUT_DOCX = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.docx"
REFERENCE_DOC = ROOT_DIR / "09 【模板 2026届】 论文正文（信息科学与技术学院）.docx"
KAITI_MD = ROOT_DIR / "kaiti.md"
KAITI_REPORT_DOCX = ROOT_DIR / "彭晓俞_开题报告 (1).docx"

COVER_TITLE = "本 科 生 毕 业 论 文 正 文"
DEFAULT_COHORT_TEXT = "（ 2026 届）"
COLLEGE_TEXT = "信息科学与技术学院"
TEMPLATE_DEBUG_FILES = (
    ROOT_DIR / "thesis_final_bundle" / "_pandoc_one_shot.docx",
    ROOT_DIR / "thesis_final_bundle" / "_tmp_template_test.docx",
    ROOT_DIR / "thesis_final_bundle" / "_toc_test.docx",
)
FIGURE_TITLE_CROP_TOP = {
    "fig1_lasso_path.png": 80,
    "fig2_rf_importance.png": 70,
    "fig3_shap_summary.png": 65,
    "fig4_shap_subsidy.png": 60,
    "fig4a_shap_subsidy_issoe.png": 60,
    "fig4b_shap_subsidy_mgshder.png": 60,
    "fig8_model_comparison.png": 50,
}


def extract_cover_info_from_kaiti(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    info: dict[str, str] = {}

    title_match = re.search(r"\*\*论文题目\*\*\s*\n([^\n]+)", text)
    if title_match:
        info["thesis_title"] = title_match.group(1).strip()

    year_match = re.search(r"（\s*(\d{4})\s*届\s*）", text)
    if year_match:
        info["year"] = year_match.group(1)

    field_patterns = {
        "student_name": [r"(?:学生姓名|姓名)\s*[:：]\s*([^\s|]+)"],
        "student_id": [r"(?:学号|学\s*号)\s*[:：]\s*([A-Za-z0-9]+)"],
        "major": [r"专业\s*[:：]\s*([^\n|]+)"],
        "class_name": [r"班级\s*[:：]\s*([^\n|]+)"],
        "advisor_name": [r"(?:指导教师|指导老师|导师)\s*[:：]\s*([^\s|]+)"],
        "advisor_title": [r"职称\s*[:：]\s*([^\n|]+)"],
    }

    for key, patterns in field_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                info[key] = match.group(1).strip()
                break

    return info


def extract_cover_info_from_report_docx(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    doc = Document(path)
    info: dict[str, str] = {}

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        year_match = re.search(r"（\s*(\d{4})\s*届\s*）", text)
        if year_match:
            info["year"] = year_match.group(1)

    for table in doc.tables:
        cells = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if len(cells) >= 3 and len(cells[0]) >= 4:
            if cells[0][0] == "学生姓名" and cells[0][2] == "学号":
                if cells[0][1]:
                    info["student_name"] = cells[0][1]
                if cells[0][3]:
                    info["student_id"] = cells[0][3]
                if cells[1][1]:
                    info["major"] = cells[1][1]
                if cells[1][3]:
                    info["class_name"] = cells[1][3]
                if cells[2][1]:
                    info["advisor_name"] = cells[2][1]
                if cells[2][3]:
                    info["advisor_title"] = cells[2][3]
            if len(cells[0]) >= 2 and cells[0][0] == "论文题目" and cells[0][1]:
                info["thesis_title"] = cells[0][1]
    return {key: value for key, value in info.items() if value}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def extract_title_abstract_keywords_body(text: str) -> tuple[str, list[str], str, str]:
    lines = text.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
    if not title:
        raise ValueError("未找到论文标题（# 一级标题）。")

    abstract_heading_idx = next(
        (i for i, line in enumerate(lines) if re.fullmatch(r"\*\*摘\s*要\*\*", line.strip())),
        None,
    )
    keywords_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"^\*\*关键词\*\*[:：]", line.strip())),
        None,
    )
    body_idx = next((i for i, line in enumerate(lines) if re.match(r"^##\s+", line)), None)
    if abstract_heading_idx is None or keywords_idx is None or body_idx is None:
        raise ValueError("未能从 Markdown 中完整解析出摘要、关键词或正文起点。")

    abstract_lines = lines[abstract_heading_idx + 1 : keywords_idx]
    abstract_blocks: list[str] = []
    current_block: list[str] = []
    for line in abstract_lines:
        if line.strip():
            current_block.append(line.strip())
        elif current_block:
            abstract_blocks.append(" ".join(current_block).strip())
            current_block = []
    if current_block:
        abstract_blocks.append(" ".join(current_block).strip())

    keywords_line = lines[keywords_idx].strip()
    keywords = re.sub(r"^\*\*关键词\*\*[:：]\s*", "", keywords_line).strip()
    body = "\n".join(lines[body_idx:]).strip() + "\n"
    return title, abstract_blocks, keywords, body


def strip_rules(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip() != "---"]
    return "\n".join(lines)


def promote_headings(text: str) -> str:
    promoted: list[str] = []
    for line in text.splitlines():
        if re.match(r"^####\s+", line):
            promoted.append(re.sub(r"^####\s+", "### ", line))
        elif re.match(r"^###\s+", line):
            promoted.append(re.sub(r"^###\s+", "## ", line))
        elif re.match(r"^##\s+", line):
            promoted.append(re.sub(r"^##\s+", "# ", line))
        else:
            promoted.append(line)
    return "\n".join(promoted)


def normalize_caption_lines(text: str) -> str:
    normalized: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\*\*((?:表|图)[^*]+)\*\*$", line.strip())
        if match:
            normalized.append(match.group(1))
        else:
            normalized.append(line)
    return "\n".join(normalized)


def reorder_figure_caption_blocks(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^图\d", line.strip()):
            j = i + 1
            blank_lines: list[str] = []
            while j < len(lines) and not lines[j].strip():
                blank_lines.append(lines[j])
                j += 1
            if j < len(lines) and re.match(r"^!\[[^\]]*\]\([^)]+\)", lines[j].strip()):
                output.append(lines[j])
                output.extend(blank_lines or [""])
                output.append(line)
                i = j + 1
                continue
        output.append(line)
        i += 1
    return "\n".join(output)


def _escape_table_cell(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("|", r"\|")
    return cleaned if cleaned else " "


def html_table_to_pipe(table_html: str) -> str:
    root = lxml_html.fragment_fromstring(table_html, create_parent="div")
    table = root.xpath(".//table")[0]

    header_rows = table.xpath("./thead/tr")
    body_rows = table.xpath("./tbody/tr") or table.xpath("./tr")
    if not header_rows and body_rows:
        header_rows = [body_rows[0]]
        body_rows = body_rows[1:]

    header_cells = [
        _escape_table_cell(" ".join(cell.itertext()))
        for cell in header_rows[0].xpath("./th|./td")
    ]
    if not header_cells:
        raise ValueError("遇到无法解析的空表格。")

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * len(header_cells)) + " |",
    ]

    for row in body_rows:
        cells = [
            _escape_table_cell(" ".join(cell.itertext()))
            for cell in row.xpath("./th|./td")
        ]
        if not cells:
            continue
        if len(cells) < len(header_cells):
            cells.extend([" "] * (len(header_cells) - len(cells)))
        elif len(cells) > len(header_cells):
            cells = cells[: len(header_cells)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def convert_html_tables(text: str) -> str:
    pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
    return pattern.sub(lambda match: html_table_to_pipe(match.group(0)), text)


def build_pandoc_body_markdown(body_text: str) -> str:
    body = strip_rules(body_text)
    body = normalize_caption_lines(body)
    body = reorder_figure_caption_blocks(body)
    body = promote_headings(body)
    body = convert_html_tables(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body + "\n"


def build_pandoc_abstract_markdown(abstract_blocks: list[str], keywords: str) -> str:
    parts = [block for block in abstract_blocks if block]
    if keywords:
        parts.append(f"关键词：{keywords}")
    return "\n\n".join(parts).strip() + "\n"


def _set_run_font(run, western: str = "Times New Roman", east_asia: str = "宋体") -> None:
    run.font.name = western
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:ascii"), western)
    r_fonts.set(qn("w:hAnsi"), western)
    r_fonts.set(qn("w:eastAsia"), east_asia)


def set_paragraph_text(paragraph, text: str) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    if text:
        paragraph.add_run(text)


def apply_font(
    paragraph,
    size_pt: float,
    *,
    bold: bool | None = None,
    east_asia: str = "宋体",
    western: str = "Times New Roman",
) -> None:
    for run in paragraph.runs:
        if bold is not None:
            run.bold = bold
        run.font.size = Pt(size_pt)
        _set_run_font(run, western=western, east_asia=east_asia)


def set_cover_cell_text(cell, text: str, *, size_pt: float = 12, align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.3
    set_paragraph_text(paragraph, text)
    apply_font(paragraph, size_pt, east_asia="楷体", western="楷体")


def split_cover_title(title: str, max_len: int = 18) -> tuple[str, str]:
    compact = re.sub(r"\s+", "", title)
    if len(compact) <= max_len:
        return compact, ""
    for token in ("与", "和", "及", "：", ":", "——"):
        idx = compact.find(token)
        if 8 <= idx <= max_len + 2:
            return compact[:idx], compact[idx:]
    mid = len(compact) // 2
    return compact[:mid], compact[mid:]


def element_tag(element) -> str:
    return element.tag.split("}")[-1]


def element_text(element) -> str:
    return "".join(element.itertext()).strip()


def has_section_break(element) -> bool:
    if element_tag(element) == "sectPr":
        return True
    return bool(element.xpath('.//*[local-name()="sectPr"]'))


def is_empty_paragraph_element(element) -> bool:
    return element_tag(element) == "p" and not element_text(element).strip()


def is_abstract_heading(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized != "" and set(normalized) <= {"摘", "要"} and "摘" in normalized and "要" in normalized


def is_toc_heading(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized != "" and set(normalized) <= {"目", "录"} and "目" in normalized and "录" in normalized


def remove_template_shapes(doc: Document) -> None:
    for tag in ("drawing", "pict", "textbox", "txbxContent"):
        for node in doc.element.xpath(f'.//*[local-name()="{tag}"]'):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def locate_template_layout(doc: Document) -> dict[str, int]:
    blocks = list(doc.element.body.iterchildren())
    section_breaks = [
        idx
        for idx, element in enumerate(blocks)
        if element_tag(element) == "p" and has_section_break(element)
    ]
    if len(section_breaks) < 3:
        raise ValueError("学校模板的分节结构异常，无法定位封面、摘要、目录和正文边界。")

    abstract_heading_idx = next(
        idx
        for idx in range(section_breaks[0] + 1, section_breaks[1])
        if element_tag(blocks[idx]) == "p" and is_abstract_heading(element_text(blocks[idx]))
    )
    toc_title_idx = next(
        idx
        for idx in range(section_breaks[1] + 1, section_breaks[2])
        if element_tag(blocks[idx]) == "p" and is_toc_heading(element_text(blocks[idx]))
    )

    abstract_title_idx = next(
        idx
        for idx in range(abstract_heading_idx - 1, section_breaks[0], -1)
        if element_tag(blocks[idx]) == "p" and normalize_text(element_text(blocks[idx])) != ""
    )

    abstract_insert_idx = abstract_heading_idx + 1
    if abstract_insert_idx < section_breaks[1] and is_empty_paragraph_element(blocks[abstract_insert_idx]):
        abstract_insert_idx += 1

    toc_insert_idx = toc_title_idx + 1
    if toc_insert_idx < section_breaks[2] and is_empty_paragraph_element(blocks[toc_insert_idx]):
        toc_insert_idx += 1

    return {
        "cover_section_break_idx": section_breaks[0],
        "abstract_section_break_idx": section_breaks[1],
        "toc_section_break_idx": section_breaks[2],
        "abstract_title_idx": abstract_title_idx,
        "abstract_heading_idx": abstract_heading_idx,
        "abstract_insert_idx": abstract_insert_idx,
        "toc_title_idx": toc_title_idx,
        "toc_insert_idx": toc_insert_idx,
        "body_insert_idx": section_breaks[2] + 1,
    }


def remove_block_range(doc: Document, start_idx: int, end_idx: int) -> None:
    blocks = list(doc.element.body.iterchildren())
    for element in blocks[start_idx:end_idx]:
        doc.element.body.remove(element)


def set_paragraph_basic_format(paragraph, *, line_spacing: float = 1.3, first_indent: float = 0) -> None:
    fmt = paragraph.paragraph_format
    fmt.line_spacing = line_spacing
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.first_line_indent = Pt(first_indent)


def style_cover_and_titles(doc: Document, thesis_title: str, cover_info: dict[str, str], cohort_text: str) -> None:
    paragraphs = doc.paragraphs

    cover_title_para = next((p for p in paragraphs[:10] if "毕业论文正文" in normalize_text(p.text)), None)
    if cover_title_para is not None:
        set_paragraph_text(cover_title_para, COVER_TITLE)
        cover_title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        apply_font(cover_title_para, 26, bold=True, east_asia="黑体", western="黑体")

    cohort_para = next((p for p in paragraphs[:12] if "届" in p.text), None)
    if cohort_para is not None:
        set_paragraph_text(cohort_para, cohort_text)
        cohort_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        apply_font(cohort_para, 16, east_asia="楷体", western="楷体")

    college_para = next((p for p in paragraphs[:20] if "信息科学与技术学院" in p.text), None)
    if college_para is not None:
        set_paragraph_text(college_para, COLLEGE_TEXT)
        college_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        apply_font(college_para, 16, bold=True, east_asia="仿宋", western="仿宋")

    cover_table = doc.tables[0]
    title_line_1, title_line_2 = split_cover_title(cover_info.get("thesis_title", thesis_title))
    title_align = WD_ALIGN_PARAGRAPH.LEFT if title_line_2 else WD_ALIGN_PARAGRAPH.CENTER
    set_cover_cell_text(cover_table.cell(0, 1), title_line_1, size_pt=14, align=title_align)
    set_cover_cell_text(cover_table.cell(1, 1), title_line_2, size_pt=14, align=title_align)
    set_cover_cell_text(cover_table.cell(2, 0), "", size_pt=14)
    set_cover_cell_text(cover_table.cell(3, 1), cover_info.get("student_name", ""))
    set_cover_cell_text(cover_table.cell(3, 3), cover_info.get("student_id", ""))
    set_cover_cell_text(cover_table.cell(4, 1), cover_info.get("major", ""))
    set_cover_cell_text(cover_table.cell(4, 3), cover_info.get("class_name", ""))
    set_cover_cell_text(cover_table.cell(5, 1), cover_info.get("advisor_name", ""))
    set_cover_cell_text(cover_table.cell(5, 3), cover_info.get("advisor_title", ""))


def style_abstract_and_toc_shell(doc: Document, layout: dict[str, int], thesis_title: str) -> None:
    blocks = list(doc.element.body.iterchildren())
    para_map = {paragraph._p: paragraph for paragraph in doc.paragraphs}

    abstract_title_para = para_map[blocks[layout["abstract_title_idx"]]]
    set_paragraph_text(abstract_title_para, thesis_title)
    abstract_title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_basic_format(abstract_title_para)
    apply_font(abstract_title_para, 16, east_asia="黑体", western="黑体")

    abstract_heading_para = para_map[blocks[layout["abstract_heading_idx"]]]
    set_paragraph_text(abstract_heading_para, "摘  要")
    abstract_heading_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_basic_format(abstract_heading_para)
    apply_font(abstract_heading_para, 14, east_asia="黑体", western="黑体")

    toc_title_para = para_map[blocks[layout["toc_title_idx"]]]
    set_paragraph_text(toc_title_para, "目  录")
    toc_title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_basic_format(toc_title_para)
    apply_font(toc_title_para, 16, east_asia="黑体", western="黑体")


def style_keywords_paragraph(paragraph, text: str) -> None:
    set_paragraph_text(paragraph, "")
    label, _, content = text.partition("：")
    label_run = paragraph.add_run(label + "：")
    content_run = paragraph.add_run(content)
    paragraph.style = paragraph.part.document.styles["Normal"]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_basic_format(paragraph, first_indent=0)
    paragraph.paragraph_format.left_indent = Pt(48)
    paragraph.paragraph_format.first_line_indent = Pt(-48)
    label_run.font.size = Pt(10.5)
    label_run.bold = None
    _set_run_font(label_run, western="黑体", east_asia="黑体")
    content_run.font.size = Pt(10.5)
    _set_run_font(content_run, western="Times New Roman", east_asia="宋体")


def set_cell_borders(cell, *, top: bool = False, bottom: bool = False) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for name in ("top", "bottom", "left", "right", "insideH", "insideV"):
        existing = borders.find(qn(f"w:{name}"))
        if existing is not None:
            borders.remove(existing)
    if top:
        el = OxmlElement("w:top")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    if bottom:
        el = OxmlElement("w:bottom")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def style_table(table) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    rows = table.rows
    if not rows:
        return

    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row.cells):
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                )
                set_paragraph_basic_format(paragraph, line_spacing=1.0, first_indent=0)
                apply_font(paragraph, 9)
                if r_idx == 0:
                    for run in paragraph.runs:
                        run.bold = True
            if r_idx == 0:
                set_cell_borders(cell, top=True, bottom=True)
            elif r_idx == len(rows) - 1:
                set_cell_borders(cell, bottom=True)


def postprocess_abstract_docx(path: Path) -> None:
    doc = Document(path)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if text.startswith("关键词：") or text.startswith("关键词:"):
            style_keywords_paragraph(paragraph, text.replace("关键词:", "关键词："))
            continue
        paragraph.style = doc.styles["Normal"]
        paragraph.alignment = None
        set_paragraph_basic_format(paragraph, first_indent=17.95)
        apply_font(paragraph, 10.5)
    doc.save(path)


def is_heading_2(text: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\s+", text))


def is_heading_3(text: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+\s+", text))


def is_caption_text(text: str) -> bool:
    return bool(re.match(r"^(表|图)\d+(?:-\d+)?[A-Za-z]?\s+[^。！？]+$", text))


def postprocess_body_docx(path: Path) -> None:
    doc = Document(path)
    heading3_style = doc.styles["标题3"] if "标题3" in doc.styles else doc.styles["Heading 3"]
    first_chapter_seen = False
    in_chapter6 = False

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        if re.match(r"^第[一二三四五六七八九十]+章\s+", text) or text in {"附录", "参考文献"}:
            paragraph.style = doc.styles["Heading 1"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_paragraph_basic_format(paragraph)
            paragraph.paragraph_format.page_break_before = first_chapter_seen
            apply_font(paragraph, 16, bold=True, east_asia="黑体")
            first_chapter_seen = True
            in_chapter6 = text.startswith("第六章")
            continue

        if is_heading_2(text):
            paragraph.style = doc.styles["Heading 2"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_paragraph_basic_format(paragraph)
            apply_font(paragraph, 14, bold=True, east_asia="黑体")
            continue

        if is_heading_3(text):
            paragraph.style = heading3_style
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_paragraph_basic_format(paragraph, first_indent=21)
            apply_font(paragraph, 12, east_asia="宋体")
            for run in paragraph.runs:
                run.bold = False
            continue

        if is_caption_text(text):
            paragraph.style = doc.styles["Normal"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_paragraph_basic_format(paragraph)
            apply_font(paragraph, 9, bold=True)
            continue

        if text.startswith("注：") or text.startswith("注:"):
            paragraph.style = doc.styles["Normal"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_paragraph_basic_format(paragraph, first_indent=0)
            apply_font(paragraph, 10.5, bold=False)
            continue

        paragraph.style = doc.styles["Normal"]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_paragraph_basic_format(paragraph, first_indent=21)
        preserve_bold = in_chapter6 and bool(re.match(r"^\d+\.\s+", text))
        apply_font(paragraph, 10.5, bold=None if preserve_bold else False)

    for table in doc.tables:
        style_table(table)

    doc.save(path)


def run_pandoc(
    input_md: Path,
    output_docx: Path,
    *,
    toc: bool = False,
    extra_resource_paths: list[str] | None = None,
) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("未找到 pandoc，请先安装 pandoc。")

    resource_paths = [
        str(SOURCE_MD.parent),
        str(ROOT_DIR / "results"),
        str(ROOT_DIR),
    ]
    if extra_resource_paths:
        resource_paths = extra_resource_paths + resource_paths
    cmd = [
        pandoc,
        "--from",
        "markdown+tex_math_dollars+raw_tex+pipe_tables",
        str(input_md),
        "-o",
        str(output_docx),
        "--reference-doc",
        str(REFERENCE_DOC),
        "--resource-path=" + ":".join(resource_paths),
        "--wrap=none",
    ]
    if toc:
        cmd.extend(["--toc", "--toc-depth=3"])

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Pandoc 导出失败。")


def prepare_docx_export_images(asset_root: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("缺少 Pillow，无法在导出时处理论文图片。") from exc

    image_dir = asset_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_dir = SOURCE_MD.parent / "images"

    for source_path in source_dir.glob("*"):
        target_path = image_dir / source_path.name
        if source_path.name in FIGURE_TITLE_CROP_TOP and source_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            crop_top = FIGURE_TITLE_CROP_TOP[source_path.name]
            with Image.open(source_path) as image:
                cropped = image.crop((0, crop_top, image.size[0], image.size[1]))
                cropped.save(target_path)
        else:
            shutil.copy2(source_path, target_path)


def insert_doc_body_elements(target_doc: Document, insert_idx: int, source_doc: Document) -> None:
    elements = [deepcopy(element) for element in source_doc.element.body if element_tag(element) != "sectPr"]
    for offset, element in enumerate(elements):
        target_doc.element.body.insert(insert_idx + offset, element)


def extract_toc_sdt(path: Path):
    doc = Document(path)
    for element in doc.element.body.iterchildren():
        if element_tag(element) == "sdt":
            return deepcopy(element)
    raise ValueError("未能从 Pandoc 生成的 TOC 文档中提取目录字段。")


def enable_update_fields(doc: Document) -> None:
    settings = doc.settings.element
    existing = settings.find(qn("w:updateFields"))
    if existing is None:
        existing = OxmlElement("w:updateFields")
        settings.append(existing)
    existing.set(qn("w:val"), "true")


def blank_toc_seed_title(sdt) -> None:
    text_nodes = sdt.xpath('.//*[local-name()="sdtContent"]/*[local-name()="p"][1]//*[local-name()="t"]')
    if text_nodes:
        text_nodes[0].text = ""


def rewrite_body_footer_page_fields(docx_path: Path) -> None:
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    w_ns = ns["w"]
    r_ns = ns["r"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(docx_path) as zf:
            zf.extractall(tmp_path)

        document_xml = tmp_path / "word" / "document.xml"
        rels_xml = tmp_path / "word" / "_rels" / "document.xml.rels"

        document_root = etree.parse(str(document_xml))
        rels_root = etree.parse(str(rels_xml))

        rel_map = {
            rel.get("Id"): rel.get("Target")
            for rel in rels_root.xpath("/*[local-name()='Relationships']/*[local-name()='Relationship']")
        }

        sect_prs = document_root.xpath("//*[local-name()='sectPr']")
        if not sect_prs:
            raise ValueError("未找到文档分节信息，无法修正正文页码。")

        body_sect = sect_prs[-1]
        footer_refs = body_sect.xpath("./w:footerReference[@w:type='default']", namespaces=ns)
        if not footer_refs:
            raise ValueError("正文分节未找到默认页脚引用，无法修正正文页码。")

        footer_rid = footer_refs[0].get(f"{{{r_ns}}}id")
        footer_target = rel_map.get(footer_rid)
        if not footer_target:
            raise ValueError("无法解析正文页脚关系目标。")

        footer_xml = tmp_path / "word" / footer_target
        footer_root = etree.parse(str(footer_xml)).getroot()

        for child in list(footer_root):
            footer_root.remove(child)

        p = etree.SubElement(footer_root, f"{{{w_ns}}}p")
        p_pr = etree.SubElement(p, f"{{{w_ns}}}pPr")
        p_style = etree.SubElement(p_pr, f"{{{w_ns}}}pStyle")
        p_style.set(f"{{{w_ns}}}val", "a7")
        jc = etree.SubElement(p_pr, f"{{{w_ns}}}jc")
        jc.set(f"{{{w_ns}}}val", "center")
        p_rpr = etree.SubElement(p_pr, f"{{{w_ns}}}rPr")
        p_lang = etree.SubElement(p_rpr, f"{{{w_ns}}}lang")
        p_lang.set(f"{{{w_ns}}}val", "zh-CN")

        def append_text_run(text: str, *, east_asia: bool = True) -> None:
            r = etree.SubElement(p, f"{{{w_ns}}}r")
            r_pr = etree.SubElement(r, f"{{{w_ns}}}rPr")
            if east_asia:
                r_fonts = etree.SubElement(r_pr, f"{{{w_ns}}}rFonts")
                r_fonts.set(f"{{{w_ns}}}hint", "eastAsia")
            lang = etree.SubElement(r_pr, f"{{{w_ns}}}lang")
            lang.set(f"{{{w_ns}}}val", "zh-CN")
            t = etree.SubElement(r, f"{{{w_ns}}}t")
            if text != text.strip():
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            t.text = text

        def append_field(instr: str, result_text: str) -> None:
            r_begin = etree.SubElement(p, f"{{{w_ns}}}r")
            r_begin_pr = etree.SubElement(r_begin, f"{{{w_ns}}}rPr")
            begin_lang = etree.SubElement(r_begin_pr, f"{{{w_ns}}}lang")
            begin_lang.set(f"{{{w_ns}}}val", "zh-CN")
            fld_begin = etree.SubElement(r_begin, f"{{{w_ns}}}fldChar")
            fld_begin.set(f"{{{w_ns}}}fldCharType", "begin")
            fld_begin.set(f"{{{w_ns}}}dirty", "true")

            r_instr = etree.SubElement(p, f"{{{w_ns}}}r")
            r_instr_pr = etree.SubElement(r_instr, f"{{{w_ns}}}rPr")
            instr_lang = etree.SubElement(r_instr_pr, f"{{{w_ns}}}lang")
            instr_lang.set(f"{{{w_ns}}}val", "zh-CN")
            instr_text = etree.SubElement(r_instr, f"{{{w_ns}}}instrText")
            instr_text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            instr_text.text = f" {instr} "

            r_sep = etree.SubElement(p, f"{{{w_ns}}}r")
            r_sep_pr = etree.SubElement(r_sep, f"{{{w_ns}}}rPr")
            sep_lang = etree.SubElement(r_sep_pr, f"{{{w_ns}}}lang")
            sep_lang.set(f"{{{w_ns}}}val", "zh-CN")
            fld_sep = etree.SubElement(r_sep, f"{{{w_ns}}}fldChar")
            fld_sep.set(f"{{{w_ns}}}fldCharType", "separate")

            r_result = etree.SubElement(p, f"{{{w_ns}}}r")
            r_result_pr = etree.SubElement(r_result, f"{{{w_ns}}}rPr")
            result_lang = etree.SubElement(r_result_pr, f"{{{w_ns}}}lang")
            result_lang.set(f"{{{w_ns}}}val", "zh-CN")
            result_t = etree.SubElement(r_result, f"{{{w_ns}}}t")
            result_t.text = result_text

            r_end = etree.SubElement(p, f"{{{w_ns}}}r")
            r_end_pr = etree.SubElement(r_end, f"{{{w_ns}}}rPr")
            end_lang = etree.SubElement(r_end_pr, f"{{{w_ns}}}lang")
            end_lang.set(f"{{{w_ns}}}val", "zh-CN")
            fld_end = etree.SubElement(r_end, f"{{{w_ns}}}fldChar")
            fld_end.set(f"{{{w_ns}}}fldCharType", "end")

        append_text_run("第")
        append_field("PAGE", "1")
        append_text_run("页")
        append_text_run("  ", east_asia=False)
        append_text_run("共")
        append_field("SECTIONPAGES", "1")
        append_text_run("页")

        etree.ElementTree(footer_root).write(str(footer_xml), encoding="utf-8", xml_declaration=True)

        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as out_zip:
            for path in sorted(tmp_path.rglob("*")):
                if path.is_file():
                    out_zip.write(path, path.relative_to(tmp_path))


def build_scaffold_doc(
    thesis_title: str,
    cohort_text: str,
    cover_info: dict[str, str],
    abstract_docx: Path,
    toc_seed_docx: Path,
) -> Document:
    scaffold = Document(REFERENCE_DOC)
    remove_template_shapes(scaffold)
    style_cover_and_titles(scaffold, thesis_title, cover_info, cohort_text)
    layout = locate_template_layout(scaffold)
    style_abstract_and_toc_shell(scaffold, layout, thesis_title)

    remove_block_range(
        scaffold,
        layout["abstract_insert_idx"],
        layout["abstract_section_break_idx"],
    )

    layout = locate_template_layout(scaffold)
    remove_block_range(
        scaffold,
        layout["toc_insert_idx"],
        layout["toc_section_break_idx"],
    )

    layout = locate_template_layout(scaffold)
    final_body_idx = len(list(scaffold.element.body.iterchildren())) - 1
    remove_block_range(scaffold, layout["body_insert_idx"], final_body_idx)

    layout = locate_template_layout(scaffold)
    insert_doc_body_elements(scaffold, layout["abstract_insert_idx"], Document(abstract_docx))
    layout = locate_template_layout(scaffold)
    toc_sdt = extract_toc_sdt(toc_seed_docx)
    blank_toc_seed_title(toc_sdt)
    scaffold.element.body.insert(layout["toc_insert_idx"], toc_sdt)
    enable_update_fields(scaffold)
    return scaffold


def cleanup_old_docx() -> list[str]:
    removed: list[str] = []
    for path in TEMPLATE_DEBUG_FILES:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main() -> int:
    if not REFERENCE_DOC.exists():
        print(f"未找到学校模板：{REFERENCE_DOC}", file=sys.stderr)
        return 1

    cover_info = extract_cover_info_from_kaiti(KAITI_MD)
    cover_info.update(extract_cover_info_from_report_docx(KAITI_REPORT_DOCX))

    source_text = SOURCE_MD.read_text(encoding="utf-8")
    title_from_md, abstract_blocks, keywords, body_text = extract_title_abstract_keywords_body(source_text)
    thesis_title = cover_info.get("thesis_title", title_from_md)
    cohort_text = f"（ {cover_info['year']} 届）" if cover_info.get("year") else DEFAULT_COHORT_TEXT

    body_md = build_pandoc_body_markdown(body_text)
    PREPROCESSED_MD.write_text(body_md, encoding="utf-8")
    abstract_md = build_pandoc_abstract_markdown(abstract_blocks, keywords)

    tmp_dir = ROOT_DIR / "tmp" / "docs" / "export_school_docx"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    abstract_md_path = tmp_dir / "abstract.md"
    abstract_docx_path = tmp_dir / "abstract.docx"
    body_docx_path = tmp_dir / "body.docx"
    toc_seed_docx_path = tmp_dir / "toc_seed.docx"
    scaffold_docx_path = tmp_dir / "scaffold.docx"
    asset_root = tmp_dir / "assets"

    abstract_md_path.write_text(abstract_md, encoding="utf-8")
    prepare_docx_export_images(asset_root)

    try:
        extra_resource_paths = [str(asset_root)]

        run_pandoc(abstract_md_path, abstract_docx_path, extra_resource_paths=extra_resource_paths)
        postprocess_abstract_docx(abstract_docx_path)

        run_pandoc(PREPROCESSED_MD, body_docx_path, extra_resource_paths=extra_resource_paths)
        postprocess_body_docx(body_docx_path)

        run_pandoc(PREPROCESSED_MD, toc_seed_docx_path, toc=True, extra_resource_paths=extra_resource_paths)

        scaffold = build_scaffold_doc(
            thesis_title=thesis_title,
            cohort_text=cohort_text,
            cover_info=cover_info,
            abstract_docx=abstract_docx_path,
            toc_seed_docx=toc_seed_docx_path,
        )
        scaffold.save(scaffold_docx_path)

        composer = Composer(Document(scaffold_docx_path))
        composer.append(Document(body_docx_path))
        composer.save(str(OUTPUT_DOCX))

        final_doc = Document(OUTPUT_DOCX)
        enable_update_fields(final_doc)
        final_doc.save(OUTPUT_DOCX)
        rewrite_body_footer_page_fields(OUTPUT_DOCX)

        removed = cleanup_old_docx()
        shutil.rmtree(tmp_dir)

        with zipfile.ZipFile(OUTPUT_DOCX) as zf:
            media_count = sum(1 for name in zf.namelist() if name.startswith("word/media/"))

        doc = Document(OUTPUT_DOCX)
        print(f"导出完成：{OUTPUT_DOCX}")
        print(f"Pandoc 预处理稿：{PREPROCESSED_MD}")
        print(f"Section 数量：{len(doc.sections)}")
        print(f"Word 表格数量：{len(doc.tables)}")
        print(f"嵌入图片数量：{media_count}")
        if removed:
            print("已删除旧测试文档：")
            for path in removed:
                print(f"  - {path}")
        if cover_info:
            print("已填入的封面字段：")
            for key in (
                "thesis_title",
                "year",
                "student_name",
                "student_id",
                "major",
                "class_name",
                "advisor_name",
                "advisor_title",
            ):
                if cover_info.get(key):
                    print(f"  - {key}: {cover_info[key]}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
