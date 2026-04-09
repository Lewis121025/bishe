"""
将论文 Markdown 正文稳定导出为带图片的 Word 文档。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.md"
OUTPUT_DOCX = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.docx"


def main() -> int:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        print("未找到 pandoc，请先安装 pandoc。", file=sys.stderr)
        return 1

    resource_paths = [
        str(SOURCE_MD.parent),
        str(ROOT_DIR / "results"),
        str(ROOT_DIR),
    ]
    cmd = [
        pandoc,
        "--from",
        "gfm+yaml_metadata_block",
        str(SOURCE_MD),
        "-o",
        str(OUTPUT_DOCX),
        "--resource-path=" + ":".join(resource_paths),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return completed.returncode

    media_count = 0
    with zipfile.ZipFile(OUTPUT_DOCX) as zf:
        media_count = sum(1 for name in zf.namelist() if name.startswith("word/media/"))

    print(f"导出完成：{OUTPUT_DOCX}")
    print(f"嵌入图片数量：{media_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
