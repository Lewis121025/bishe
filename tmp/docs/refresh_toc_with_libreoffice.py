from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def _find_toc_values(docx_path: Path) -> dict[str, str]:
    with ZipFile(docx_path) as zf:
        root = etree.fromstring(zf.read("word/document.xml"))
    paragraphs = root.xpath("//w:p", namespaces=NS)
    values: dict[str, str] = {}
    for paragraph in paragraphs:
        instr_texts = "".join(paragraph.xpath(".//w:instrText/text()", namespaces=NS))
        if "PAGEREF" not in instr_texts or "_TocAuto" not in instr_texts:
            continue
        visible_runs = paragraph.xpath(".//w:r[w:rPr/w:rStyle[@w:val='22']]/w:t/text()", namespaces=NS)
        page_runs = paragraph.xpath(
            ".//w:r[preceding-sibling::w:r/w:instrText[contains(., 'PAGEREF')]]/w:t/text()",
            namespaces=NS,
        )
        if not visible_runs or not page_runs:
            continue
        title = visible_runs[0].strip()
        page = page_runs[-1].strip()
        if title:
            values[title] = page
    return values


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"LibreOffice listener on {host}:{port} did not start")


def _run_lo_update(docx_path: Path, soffice: Path, lo_python: Path) -> None:
    host = "127.0.0.1"
    port = 2002
    with tempfile.TemporaryDirectory(prefix="lo-profile-") as profile_dir:
        profile_uri = Path(profile_dir).as_uri()
        cmd = [
            str(soffice),
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--nologo",
            "--nodefault",
            "--norestore",
            "--nolockcheck",
            f'--accept=socket,host={host},port={port};urp;StarOffice.ComponentContext',
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            _wait_for_port(host, port)
            helper = Path(profile_dir) / "update_indexes.py"
            helper.write_text(
                """
import sys
import uno
from com.sun.star.beans import PropertyValue

host = "127.0.0.1"
port = 2002
path = sys.argv[1]
local_ctx = uno.getComponentContext()
resolver = local_ctx.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local_ctx
)
ctx = resolver.resolve(
    f"uno:socket,host={host},port={port};urp;StarOffice.ComponentContext"
)
desktop = ctx.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.Desktop", ctx
)

def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p

url = uno.systemPathToFileUrl(path)
doc = desktop.loadComponentFromURL(url, "_blank", 0, (prop("Hidden", True),))
indexes = doc.getDocumentIndexes()
for i in range(indexes.getCount()):
    indexes.getByIndex(i).update()
doc.refresh()
doc.store()
doc.close(True)
""".strip()
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [str(lo_python), str(helper), str(docx_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _update_original_toc(original_docx: Path, title_to_page: dict[str, str]) -> None:
    with ZipFile(original_docx) as zf:
        files = {name: zf.read(name) for name in zf.namelist()}

    root = etree.fromstring(files["word/document.xml"])
    paragraphs = root.xpath("//w:p", namespaces=NS)
    for paragraph in paragraphs:
        instr_texts = "".join(paragraph.xpath(".//w:instrText/text()", namespaces=NS))
        if "PAGEREF" not in instr_texts or "_TocAuto" not in instr_texts:
            continue
        visible_runs = paragraph.xpath(".//w:r[w:rPr/w:rStyle[@w:val='22']]/w:t", namespaces=NS)
        page_runs = paragraph.xpath(
            ".//w:r[preceding-sibling::w:r/w:instrText[contains(., 'PAGEREF')]]/w:t",
            namespaces=NS,
        )
        if not visible_runs or not page_runs:
            continue
        title = (visible_runs[0].text or "").strip()
        if title in title_to_page:
            page_runs[-1].text = title_to_page[title]

    files["word/document.xml"] = etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )
    with ZipFile(original_docx, "w", compression=ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python3 refresh_toc_with_libreoffice.py /path/to/docx /path/to/soffice /path/to/libreoffice-python"
        )

    docx_path = Path(sys.argv[1]).resolve()
    soffice = Path(sys.argv[2]).resolve()
    lo_python = Path(sys.argv[3]).resolve()

    temp_docx = docx_path.with_suffix(".lo-refresh.docx")
    temp_docx.write_bytes(docx_path.read_bytes())
    try:
        _run_lo_update(temp_docx, soffice, lo_python)
        pages = _find_toc_values(temp_docx)
        _update_original_toc(docx_path, pages)
    finally:
        temp_docx.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
