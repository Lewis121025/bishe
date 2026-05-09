# 论文实验复现包

本仓库用于归档论文《基于数据挖掘的上市公司财政补贴与高管超额薪酬研究》的原始数据、数据处理脚本、实证分析代码与机器学习补充检验代码。

## 优先查看

- `data/`：原始 Stata 数据
- `scripts/regression_analysis.py`：主实证分析脚本
- `scripts/ml_analysis.py`：机器学习补充分析脚本
- `requirements.txt`：Python 依赖清单

## 目录结构

- `data/`：原始 Stata 数据
- `scripts/`：数据处理、回归分析、机器学习分析和表格导出脚本
- `tests/`：轻量级单元测试

## 代码运行顺序

建议在仓库根目录执行以下命令：

```bash
python scripts/convert_data.py
python scripts/validate_conversion.py
python scripts/regression_analysis.py
python scripts/generate_tables.py
python scripts/generate_tables_docx.py
python scripts/ml_analysis.py
```

说明：

- 所有脚本已改为基于脚本自身位置解析仓库路径，不再依赖作者本机的绝对路径。
- 运行后，核心结果会重新写入 `results/` 目录。
- 当前归档包不保留 `processed_data/` 中间 CSV；运行 `python scripts/convert_data.py` 会重新生成。
- `scripts/panel_regressions.py` 是额外的纯 Python 面板回归检验脚本，依赖主回归脚本重新生成的 `results/regression_dataset.csv`，该中间数据不在归档包内保留。
- 若只需要重新生成文本表格，可在完成主回归后运行 `python scripts/generate_tables.py`。
- 若只需要重新生成 Word 三线表，可在完成主回归后运行 `python scripts/generate_tables_docx.py`。

## 运行环境

- Python 3.13
- 依赖清单见 `requirements.txt`

建议在独立环境中安装依赖：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 归档内容

本归档包仅保留复现实验所需内容：

- `data/`
- `scripts/`
- `tests/`
- `requirements.txt`
- `README.md`

## 进一步说明

- `scripts/README.md` 说明各脚本用途与输出。
- 运行脚本生成的 `processed_data/` 与 `results/` 均为可再生内容，不预置在提交归档包中。
- 仓库已清理论文正文草稿、开题/答辩材料、格式检测报告、PDF 渲染缓存、临时脚本、中间数据、结果文件和历史探索输出，以减少归档噪音。
