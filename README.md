# 论文仓库说明

本仓库用于保存论文《基于数据挖掘的上市公司财政补贴与高管超额薪酬研究》的正文、数据处理脚本、实证分析代码与结果文件。

## 优先查看

- `thesis_final_bundle/thesis_final_humanized_v2.md`：当前采用的论文 Markdown 正文
- `results/回归结果表.docx`：按论文格式导出的三线表
- `results/regression_tables.txt`：论文表格的文本版汇总
- `scripts/regression_analysis.py`：主实证分析脚本
- `scripts/ml_analysis.py`：机器学习补充分析脚本
- `scripts/update_embedded_markdown.sh`：一键更新带图片的自包含 Markdown 版本

## 目录结构

- `data/`：原始 Stata 数据与参考文献 PDF
- `processed_data/`：由原始 `.dta` 转换后的 CSV 数据
- `scripts/`：数据处理、回归分析、机器学习分析和表格导出脚本
- `results/`：回归结果、描述性统计、稳健性检验、机器学习图表与汇总文件
- `thesis_final_bundle/`：论文正文及交付稿目录
- `参考论文笔记_刘剑民2019.md`：过程性参考材料

## 代码运行顺序

建议在仓库根目录执行以下命令：

```bash
python scripts/convert_data.py
python scripts/validate_conversion.py
python scripts/regression_analysis.py
python scripts/generate_tables.py
python scripts/generate_tables_docx.py
python scripts/ml_analysis.py
./scripts/update_embedded_markdown.sh
```

说明：

- 所有脚本已改为基于脚本自身位置解析仓库路径，不再依赖作者本机的绝对路径。
- 运行后，核心结果会写入 `results/` 目录。
- `scripts/panel_regressions.py` 是额外的纯 Python 面板回归检验脚本，依赖 `results/regression_dataset.csv`。
- 若更新了论文主文件并需要同步生成带图片的单文件 Markdown，可直接运行 `./scripts/update_embedded_markdown.sh`，输出为 `thesis_final_bundle/thesis_final_humanized_v2_embedded.md`。

## 运行环境

- Python 3.13
- 依赖清单见 `requirements.txt`
- 当前仓库内已有本地虚拟环境 `venv/`，但对外提交时通常不必附带

## 交付建议

如果是压缩后发送给导师，建议保留以下内容：

- `thesis_final_bundle/`
- `scripts/`
- `data/` 与 `processed_data/`
- `results/`
- `requirements.txt`
- 本说明文件

通常可以不附带以下内容，以减少体积和噪音：

- `.git/`
- `venv/`
- `.claude/`

## 进一步说明

- `scripts/README.md` 说明各脚本用途与输出。
- `results/README.md` 说明结果目录中各文件的对应关系。
- 仓库不再保留旧值草稿、内嵌 base64 的临时正文和 PDF 转 Markdown 中间文件，以避免误用过时口径。
