"""
将论文 Markdown 导出为自包含版本：把本地图片引用替换为 base64 data URI。
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2.md"
OUTPUT_MD = ROOT_DIR / "thesis_final_bundle" / "thesis_final_humanized_v2_embedded.md"
IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def _to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main() -> int:
    if not SOURCE_MD.exists():
        print(f"未找到源文件：{SOURCE_MD}", file=sys.stderr)
        return 1

    source_text = SOURCE_MD.read_text(encoding="utf-8")
    replaced_count = 0

    def replace_image(match: re.Match[str]) -> str:
        nonlocal replaced_count
        raw_path = match.group("path").strip()
        if raw_path.startswith("data:") or raw_path.startswith("http://") or raw_path.startswith("https://"):
            return match.group(0)

        image_path = (SOURCE_MD.parent / raw_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在：{image_path}")

        replaced_count += 1
        alt = match.group("alt")
        return f"![{alt}]({_to_data_uri(image_path)})"

    output_text = IMAGE_PATTERN.sub(replace_image, source_text)
    OUTPUT_MD.write_text(output_text, encoding="utf-8")

    print(f"导出完成：{OUTPUT_MD}")
    print(f"嵌入图片数量：{replaced_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
