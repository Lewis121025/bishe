from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


NS = {
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
}

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
W_NS = NS["w"]
R_NS = NS["r"]
W14_NS = NS["w14"]

for prefix, uri in NS.items():
    etree.register_namespace(prefix, uri)


def qn(ns: str, tag: str) -> str:
    return f"{{{NS[ns]}}}{tag}"


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def normalize_spaces(text: str) -> str:
    return "".join(text.split())


def make_para_id(counter: int) -> str:
    return f"{counter:08X}"[-8:]


def set_para_id(paragraph: etree._Element, counter: int) -> int:
    paragraph.set(qn("w14", "paraId"), make_para_id(counter))
    return counter + 1


def replace_text_in_runs(paragraph: etree._Element, new_text: str) -> None:
    texts = paragraph.xpath(".//w:t", namespaces=NS)
    if not texts:
        run = etree.SubElement(paragraph, qn("w", "r"))
        texts = [etree.SubElement(run, qn("w", "t"))]
    texts[0].text = new_text
    for extra in texts[1:]:
        parent = extra.getparent()
        parent.remove(extra)


def build_run(
    text: str,
    *,
    font_ascii: str = "Times New Roman",
    font_east: str = "宋体",
    size: int = 21,
    bold: bool = False,
    preserve: bool = False,
) -> etree._Element:
    run = etree.Element(qn("w", "r"))
    rpr = etree.SubElement(run, qn("w", "rPr"))
    rfonts = etree.SubElement(rpr, qn("w", "rFonts"))
    rfonts.set(qn("w", "ascii"), font_ascii)
    rfonts.set(qn("w", "hAnsi"), font_ascii)
    rfonts.set(qn("w", "eastAsia"), font_east)
    if bold:
        etree.SubElement(rpr, qn("w", "b"))
    sz = etree.SubElement(rpr, qn("w", "sz"))
    sz.set(qn("w", "val"), str(size))
    text_el = etree.SubElement(run, qn("w", "t"))
    if preserve or text.startswith(" ") or text.endswith(" "):
        text_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_el.text = text
    return run


def build_paragraph(
    text: str = "",
    *,
    align: str = "left",
    bold: bool = False,
    font_ascii: str = "Times New Roman",
    font_east: str = "宋体",
    size: int = 21,
    first_line: int | None = 420,
    line: int = 312,
    page_break_before: bool = False,
) -> etree._Element:
    paragraph = etree.Element(qn("w", "p"))
    ppr = etree.SubElement(paragraph, qn("w", "pPr"))
    spacing = etree.SubElement(ppr, qn("w", "spacing"))
    spacing.set(qn("w", "before"), "0")
    spacing.set(qn("w", "after"), "0")
    spacing.set(qn("w", "line"), str(line))
    spacing.set(qn("w", "lineRule"), "auto")
    if first_line is not None:
        ind = etree.SubElement(ppr, qn("w", "ind"))
        ind.set(qn("w", "firstLine"), str(first_line))
    if page_break_before:
        etree.SubElement(ppr, qn("w", "pageBreakBefore"))
    jc = etree.SubElement(ppr, qn("w", "jc"))
    jc.set(qn("w", "val"), align)
    if text:
        paragraph.append(
            build_run(
                text,
                font_ascii=font_ascii,
                font_east=font_east,
                size=size,
                bold=bold,
            )
        )
    return paragraph


def build_code_paragraph(line: str) -> etree._Element:
    paragraph = build_paragraph(
        "",
        align="left",
        bold=False,
        font_ascii="Courier New",
        font_east="等线",
        size=18,
        first_line=0,
        line=240,
    )
    paragraph.append(
        build_run(
            line,
            font_ascii="Courier New",
            font_east="等线",
            size=18,
            preserve=True,
        )
    )
    return paragraph


def build_table(rows: list[list[str]]) -> etree._Element:
    widths = ["1200", "2500", "4800"]
    tbl = etree.Element(qn("w", "tbl"))
    tbl_pr = etree.SubElement(tbl, qn("w", "tblPr"))
    tbl_w = etree.SubElement(tbl_pr, qn("w", "tblW"))
    tbl_w.set(qn("w", "w"), "8500")
    tbl_w.set(qn("w", "type"), "dxa")
    borders = etree.SubElement(tbl_pr, qn("w", "tblBorders"))
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = etree.SubElement(borders, qn("w", border_name))
        border.set(qn("w", "val"), "single")
        border.set(qn("w", "sz"), "4")
        border.set(qn("w", "color"), "000000")
    tbl_grid = etree.SubElement(tbl, qn("w", "tblGrid"))
    for width in widths[: len(rows[0])]:
        grid_col = etree.SubElement(tbl_grid, qn("w", "gridCol"))
        grid_col.set(qn("w", "w"), width)

    for row_idx, row in enumerate(rows):
        tr = etree.SubElement(tbl, qn("w", "tr"))
        for col_idx, cell_text in enumerate(row):
            tc = etree.SubElement(tr, qn("w", "tc"))
            tc_pr = etree.SubElement(tc, qn("w", "tcPr"))
            tc_w = etree.SubElement(tc_pr, qn("w", "tcW"))
            tc_w.set(qn("w", "w"), widths[min(col_idx, len(widths) - 1)])
            tc_w.set(qn("w", "type"), "dxa")
            cell_para = build_paragraph(
                cell_text,
                align="center" if row_idx == 0 else "left",
                bold=row_idx == 0,
                font_ascii="Times New Roman",
                font_east="宋体",
                size=21,
                first_line=0,
                line=312,
            )
            tc.append(cell_para)
        etree.SubElement(tr, qn("w", "trPr"))
    return tbl


def parse_table_block(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        parts = [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"-+", cell) for cell in parts):
            continue
        rows.append(parts)
    return rows


def parse_appendix_markdown(markdown_text: str) -> tuple[str, list[tuple[str, object]]]:
    title = "附录"
    items: list[tuple[str, object]] = []
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            i += 1
            continue
        if stripped.startswith("## "):
            items.append(("section", stripped[3:].strip().replace("`", "")))
            i += 1
            continue
        if stripped.startswith("```"):
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip("\n"))
                i += 1
            items.append(("code", block))
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip("\n"))
                i += 1
            items.append(("table", parse_table_block(table_lines)))
            continue
        items.append(("paragraph", stripped.replace("`", "")))
        i += 1
    return title, items


def clone_header(header_xml: bytes, title: str) -> bytes:
    header_root = etree.fromstring(header_xml)
    texts = header_root.xpath(".//w:t", namespaces=NS)
    texts[-1].text = f"             {title} "
    return etree.tostring(header_root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def add_relationship(rels_root: etree._Element, next_rid: int, target: str) -> tuple[str, int]:
    rid = f"rId{next_rid}"
    rel = etree.SubElement(rels_root, qn("pr", "Relationship"))
    rel.set("Id", rid)
    rel.set(
        "Type",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header",
    )
    rel.set("Target", target)
    return rid, next_rid + 1


def ensure_content_type(content_types: etree._Element, part_name: str) -> None:
    existing = content_types.xpath(
        f'./ct:Override[@PartName="{part_name}"]',
        namespaces=NS,
    )
    if existing:
        return
    override = etree.SubElement(content_types, qn("ct", "Override"))
    override.set("PartName", part_name)
    override.set(
        "ContentType",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml",
    )


def clone_sectpr(
    base_sectpr: etree._Element,
    *,
    header_rid: str,
    continuous: bool,
    restart_page_number: bool,
) -> etree._Element:
    sectpr = deepcopy(base_sectpr)
    header_ref = sectpr.find("w:headerReference[@w:type='default']", NS)
    if header_ref is None:
        header_ref = etree.Element(qn("w", "headerReference"))
        header_ref.set(qn("w", "type"), "default")
        sectpr.insert(0, header_ref)
    header_ref.set(qn("r", "id"), header_rid)

    type_el = sectpr.find("w:type", NS)
    if continuous:
        if type_el is None:
            type_el = etree.Element(qn("w", "type"))
            insert_pos = 0
            header_refs = sectpr.findall("w:headerReference", NS)
            footer_refs = sectpr.findall("w:footerReference", NS)
            insert_pos = len(header_refs) + len(footer_refs)
            sectpr.insert(insert_pos, type_el)
        type_el.set(qn("w", "val"), "continuous")
    elif type_el is not None:
        sectpr.remove(type_el)

    pg_num_type = sectpr.find("w:pgNumType", NS)
    if restart_page_number:
        if pg_num_type is None:
            pg_num_type = etree.SubElement(sectpr, qn("w", "pgNumType"))
        pg_num_type.set(qn("w", "start"), "1")
    elif pg_num_type is not None:
        sectpr.remove(pg_num_type)

    return sectpr


def build_section_break_paragraph(
    template_paragraph: etree._Element,
    sectpr: etree._Element,
) -> etree._Element:
    paragraph = deepcopy(template_paragraph)
    for child in list(paragraph):
        paragraph.remove(child)
    ppr = etree.SubElement(paragraph, qn("w", "pPr"))
    spacing = etree.SubElement(ppr, qn("w", "spacing"))
    spacing.set(qn("w", "line"), "312")
    spacing.set(qn("w", "lineRule"), "auto")
    ind = etree.SubElement(ppr, qn("w", "ind"))
    ind.set(qn("w", "right"), "25")
    ind.set(qn("w", "rightChars"), "12")
    jc = etree.SubElement(ppr, qn("w", "jc"))
    jc.set(qn("w", "val"), "distribute")
    ppr.append(sectpr)
    return paragraph


def update_toc_paragraph(paragraph: etree._Element, bookmark: str, title: str, page: int) -> None:
    for instr in paragraph.xpath(".//w:instrText", namespaces=NS):
        text = instr.text or ""
        text = re.sub(r'_TocAuto\d+', bookmark, text)
        instr.text = text
    display_texts = paragraph.xpath(".//w:r[w:rPr/w:rStyle[@w:val='22']]/w:t", namespaces=NS)
    if display_texts:
        display_texts[0].text = title
    page_texts = paragraph.xpath(
        ".//w:r[preceding-sibling::w:r/w:instrText[contains(., 'PAGEREF')]]/w:t",
        namespaces=NS,
    )
    if page_texts:
        page_texts[-1].text = str(page)


def find_toc_paragraph(body: etree._Element, bookmark: str) -> etree._Element:
    for paragraph in body.findall("w:p", NS):
        instr_texts = paragraph.xpath(".//w:instrText/text()", namespaces=NS)
        if any(bookmark in text for text in instr_texts):
            return paragraph
    raise ValueError(f"TOC paragraph not found for {bookmark}")


def compute_page_numbers(docx_path: Path, anchors: list[str]) -> dict[str, int]:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "qlmanage",
                "-p",
                "-o",
                tmpdir,
                str(docx_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        preview_html = (
            Path(tmpdir)
            / f"{docx_path.name}.qlpreview"
            / "Preview.html"
        ).read_text(encoding="utf-8")
    pages: dict[str, int] = {}
    for anchor in anchors:
        marker = f'name="{anchor}"'
        pos = preview_html.index(marker)
        pages[anchor] = preview_html[:pos].count("\x0c") + 1
    return pages


def update_fields_setting(settings_root: etree._Element) -> None:
    update_fields = settings_root.find("w:updateFields", NS)
    if update_fields is None:
        update_fields = etree.SubElement(settings_root, qn("w", "updateFields"))
    update_fields.set(qn("w", "val"), "true")


def save_docx(files: dict[str, bytes], output_path: Path) -> None:
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def main() -> int:
    import sys

    docx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/lewis/毕业论文/thesis_final_v3.docx")
    appendix_md_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/lewis/毕业论文/APPENDIX_CODE.md")
    tmp_output = docx_path.with_suffix(".tmp.docx")

    with ZipFile(docx_path) as zf:
        files = {name: zf.read(name) for name in zf.namelist()}

    document = etree.fromstring(files["word/document.xml"])
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    content_types = etree.fromstring(files["[Content_Types].xml"])
    settings = etree.fromstring(files["word/settings.xml"])

    body = document.find("w:body", NS)
    assert body is not None, "document body missing"

    direct_children = list(body)
    ref_heading = next(
        child
        for child in direct_children
        if etree.QName(child).localname == "p" and paragraph_text(child) == "参考文献"
    )
    ack_heading = next(
        child
        for child in direct_children
        if etree.QName(child).localname == "p" and paragraph_text(child) == "致 谢"
    )
    section_break_template = next(
        child
        for child in direct_children
        if etree.QName(child).localname == "p"
        and child.find("w:pPr/w:sectPr", NS) is not None
        and not paragraph_text(child)
    )

    final_sectpr = body.find("w:sectPr", NS)
    assert final_sectpr is not None, "final sectPr missing"
    current_body_header_ref = final_sectpr.find("w:headerReference[@w:type='default']", NS)
    assert current_body_header_ref is not None, "current body header missing"
    current_body_header_rid = current_body_header_ref.get(qn("r", "id"))

    relationship_ids = [
        int(match.group(1))
        for rel_id in rels.xpath("./pr:Relationship/@Id", namespaces=NS)
        if (match := re.fullmatch(r"rId(\d+)", rel_id))
    ]
    next_rid = max(relationship_ids) + 1

    existing_headers = sorted(
        int(match.group(1))
        for name in files
        if (match := re.fullmatch(r"word/header(\d+)\.xml", name))
    )
    next_header_num = max(existing_headers) + 1

    header4_xml = files["word/header4.xml"]
    header_specs = [
        ("参考文献", f"header{next_header_num}.xml"),
        ("附录", f"header{next_header_num + 1}.xml"),
        ("致谢", f"header{next_header_num + 2}.xml"),
    ]
    header_rids: dict[str, str] = {}
    for title, filename in header_specs:
        rid, next_rid = add_relationship(rels, next_rid, filename)
        header_rids[title] = rid
        files[f"word/{filename}"] = clone_header(header4_xml, title)
        ensure_content_type(content_types, f"/word/{filename}")

    base_break_sectpr = deepcopy(final_sectpr)
    break_before_ref = build_section_break_paragraph(
        section_break_template,
        clone_sectpr(
            base_break_sectpr,
            header_rid=current_body_header_rid,
            continuous=True,
            restart_page_number=True,
        ),
    )
    break_before_appendix = build_section_break_paragraph(
        section_break_template,
        clone_sectpr(
            base_break_sectpr,
            header_rid=header_rids["参考文献"],
            continuous=True,
            restart_page_number=False,
        ),
    )
    break_before_ack = build_section_break_paragraph(
        section_break_template,
        clone_sectpr(
            base_break_sectpr,
            header_rid=header_rids["附录"],
            continuous=True,
            restart_page_number=False,
        ),
    )

    ack_header_sectpr = clone_sectpr(
        final_sectpr,
        header_rid=header_rids["致谢"],
        continuous=False,
        restart_page_number=False,
    )
    body.replace(final_sectpr, ack_header_sectpr)

    ref_heading.addprevious(break_before_ref)

    appendix_title, appendix_items = parse_appendix_markdown(appendix_md_path.read_text(encoding="utf-8"))
    appendix_heading = deepcopy(ack_heading)
    for bookmark in appendix_heading.xpath(".//w:bookmarkStart | .//w:bookmarkEnd", namespaces=NS):
        bookmark.getparent().remove(bookmark)
    bookmark_id = max(
        int(value)
        for value in document.xpath("//w:bookmarkStart/@w:id", namespaces=NS)
    ) + 1
    p_texts = appendix_heading.xpath(".//w:t", namespaces=NS)
    p_texts[0].text = "附录"
    bookmark_start = etree.Element(qn("w", "bookmarkStart"))
    bookmark_start.set(qn("w", "id"), str(bookmark_id))
    bookmark_start.set(qn("w", "name"), "_TocAuto0031")
    bookmark_end = etree.Element(qn("w", "bookmarkEnd"))
    bookmark_end.set(qn("w", "id"), str(bookmark_id))
    appendix_heading.insert(1, bookmark_start)
    appendix_heading.append(bookmark_end)

    subtitle = appendix_title.replace("附录", "", 1).strip() or "Python 代码实现"
    appendix_elements: list[etree._Element] = [
        break_before_appendix,
        appendix_heading,
        build_paragraph(
            subtitle,
            align="center",
            bold=True,
            font_ascii="Times New Roman",
            font_east="黑体",
            size=28,
            first_line=0,
            line=312,
        ),
        build_paragraph(
            "",
            align="left",
            first_line=0,
            line=312,
        ),
    ]

    for kind, value in appendix_items:
        if kind == "section":
            appendix_elements.append(
                build_paragraph(
                    str(value),
                    align="left",
                    bold=True,
                    font_ascii="Times New Roman",
                    font_east="黑体",
                    size=26,
                    first_line=0,
                    line=312,
                )
            )
        elif kind == "paragraph":
            appendix_elements.append(
                build_paragraph(
                    str(value),
                    align="left",
                    bold=False,
                    font_ascii="Times New Roman",
                    font_east="宋体",
                    size=21,
                    first_line=420,
                    line=312,
                )
            )
        elif kind == "table":
            appendix_elements.append(build_table(value))  # type: ignore[arg-type]
        elif kind == "code":
            for line in value:  # type: ignore[assignment]
                appendix_elements.append(build_code_paragraph(line))
        appendix_elements.append(build_paragraph("", align="left", first_line=0, line=240))

    appendix_elements.append(break_before_ack)

    for elem in appendix_elements:
        ack_heading.addprevious(elem)

    toc_ref = find_toc_paragraph(body, "_TocAuto0029")
    toc_ack = find_toc_paragraph(body, "_TocAuto0030")
    toc_appendix = deepcopy(toc_ref)
    update_toc_paragraph(toc_appendix, "_TocAuto0031", "附录", 0)
    toc_ack.addprevious(toc_appendix)

    update_fields_setting(settings)

    para_counter = max(
        int(value, 16)
        for value in document.xpath("//w:p/@w14:paraId", namespaces=NS)
        if re.fullmatch(r"[0-9A-Fa-f]{8}", value)
    ) + 1
    for paragraph in body.findall("w:p", NS):
        if qn("w14", "paraId") not in paragraph.attrib:
            para_counter = set_para_id(paragraph, para_counter)

    files["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )
    files["word/_rels/document.xml.rels"] = etree.tostring(
        rels,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )
    files["[Content_Types].xml"] = etree.tostring(
        content_types,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )
    files["word/settings.xml"] = etree.tostring(
        settings,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )

    save_docx(files, tmp_output)

    pages = compute_page_numbers(
        tmp_output,
        ["_TocAuto0029", "_TocAuto0030", "_TocAuto0031"],
    )

    document = etree.fromstring(files["word/document.xml"])
    body = document.find("w:body", NS)
    assert body is not None
    toc_ref = find_toc_paragraph(body, "_TocAuto0029")
    toc_ack = find_toc_paragraph(body, "_TocAuto0030")
    toc_appendix = find_toc_paragraph(body, "_TocAuto0031")
    update_toc_paragraph(toc_ref, "_TocAuto0029", "参考文献", pages["_TocAuto0029"])
    update_toc_paragraph(toc_appendix, "_TocAuto0031", "附录", pages["_TocAuto0031"])
    update_toc_paragraph(toc_ack, "_TocAuto0030", "致 谢", pages["_TocAuto0030"])

    files["word/document.xml"] = etree.tostring(
        document,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )
    save_docx(files, tmp_output)
    shutil.move(tmp_output, docx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
