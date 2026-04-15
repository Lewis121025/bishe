from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


NS = {
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def normalize_spaces(text: str) -> str:
    return "".join(text.split())


def load_xml(docx_path: Path, member: str) -> etree._Element:
    with ZipFile(docx_path) as zf:
        return etree.fromstring(zf.read(member))


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 check_thesis_appendix.py /path/to/file.docx")

    docx_path = Path(sys.argv[1])
    if not docx_path.exists():
        raise SystemExit(f"missing file: {docx_path}")

    with ZipFile(docx_path) as zf:
        document = etree.fromstring(zf.read("word/document.xml"))
        rels = etree.fromstring(zf.read("word/_rels/document.xml.rels"))
        body = document.find("w:body", NS)
        assert body is not None, "document body missing"

        paragraphs = body.findall("w:p", NS)
        texts = [paragraph_text(p) for p in paragraphs]

        ref_idx = texts.index("参考文献")
        appendix_idx = texts.index("附录")
        ack_idx = texts.index("致 谢")
        assert ref_idx < appendix_idx < ack_idx, "附录没有插在参考文献和致谢之间"

        rid_to_target = {
            rel.get("Id"): rel.get("Target")
            for rel in rels.findall("pr:Relationship", NS)
        }

        section_headers: list[str] = []
        for sect in body.xpath(".//w:sectPr", namespaces=NS):
            header_ref = sect.find("w:headerReference[@w:type='default']", NS)
            if header_ref is None:
                continue
            target = rid_to_target[header_ref.get(f'{{{NS["r"]}}}id')]
            header_xml = etree.fromstring(zf.read(f"word/{Path(target).name}"))
            header_text = normalize_spaces("".join(header_xml.xpath(".//w:t/text()", namespaces=NS)))
            section_headers.append(header_text)

        assert len(section_headers) >= 4, "没有生成足够的节页眉"
        tail = section_headers[-4:]
        expected = ["论文正文", "参考文献", "附录", "致谢"]
        for actual, want in zip(tail, expected):
            assert want in actual, f"页眉不正确: 期望包含 {want}，实际为 {actual}"

        body_start = texts.index("第一章 绪论")
        toc_texts = [t for t in texts[:body_start] if t]

        def first_index(prefix: str) -> int:
            for idx, text in enumerate(toc_texts):
                if text.startswith(prefix):
                    return idx
            raise ValueError(prefix)

        toc_ref_idx = first_index("参考文献")
        toc_appendix_idx = first_index("附录")
        toc_ack_idx = first_index("致 谢")
        assert toc_ref_idx < toc_appendix_idx < toc_ack_idx, "目录中的附录顺序不正确"

        settings = etree.fromstring(zf.read("word/settings.xml"))
        update_fields = settings.find("w:updateFields", NS)
        assert update_fields is not None and update_fields.get(f'{{{NS["w"]}}}val') == "true", "没有启用字段更新"

    print("structure ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
