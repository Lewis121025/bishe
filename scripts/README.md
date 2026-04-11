# scripts 目录说明

本目录保存论文复现所需的主要代码，建议按以下顺序理解和运行。

## 核心脚本

- `convert_data.py`
  - 作用：将 `data/` 中的 `.dta` 文件转换为 `processed_data/` 下的 `.csv`
  - 输出：`processed_data/*.csv`

- `validate_conversion.py`
  - 作用：校验原始 `.dta` 与转换后的 `.csv` 在行数和字段上的对应关系
  - 输出：终端校验信息

- `regression_analysis.py`
  - 作用：构建分析样本，完成期望薪酬模型、主回归、中介效应、异质性、稳健性、相关性和 VIF 等分析
  - 主要输出：`results/regression_dataset.csv` 及多个回归结果汇总文件

- `generate_tables.py`
  - 作用：将回归结果整理成论文文本表格
  - 输出：`results/regression_tables.txt`

- `generate_tables_docx.py`
  - 作用：将主要实证结果导出为 Word 三线表
  - 输出：`results/回归结果表.docx`

- `ml_analysis.py`
  - 作用：围绕 OLS 主回归完成随机森林、Lasso、XGBoost 三种机器学习稳健性检验
  - 主要输出：`results/lasso_coefficients.csv`、`results/rf_reg_importance.csv`、`results/xgb_importance.csv`、`results/xgb_partial_dependence.csv` 及机器学习验证摘要文件

- `verify_thesis_consistency.py`
  - 作用：核对 `thesis_final_bundle/thesis_final_humanized_v2.md` 中的核心实证数字是否与 `results/` 目录中的最新结果逐项一致
  - 输出：终端一致性校验结果，失败时列出不一致项

- `export_thesis_markdown_embedded.py`
  - 作用：将主论文 Markdown 中的本地图片引用替换为 base64 data URI，生成可单文件分发的带图片版本
  - 输出：`thesis_final_bundle/thesis_final_humanized_v2_embedded.md`

- `update_embedded_markdown.sh`
  - 作用：一键调用上面的导出脚本，适合每次更新主文件后直接同步带图片版
  - 输出：`thesis_final_bundle/thesis_final_humanized_v2_embedded.md`

## 辅助脚本

- `panel_regressions.py`
  - 作用：基于 `results/regression_dataset.csv` 做补充性的纯 Python 面板回归检验
  - 说明：现已改为仓库相对路径，可在不同机器上直接运行

## 推荐执行顺序

```bash
python scripts/convert_data.py
python scripts/validate_conversion.py
python scripts/regression_analysis.py
python scripts/generate_tables.py
python scripts/generate_tables_docx.py
venv/bin/python scripts/ml_analysis.py
python scripts/verify_thesis_consistency.py
./scripts/update_embedded_markdown.sh
```

## 使用说明

- 所有脚本都默认以仓库目录为根，不要求必须从某一台特定电脑或绝对路径运行。
- `ml_analysis.py` 依赖仓库内虚拟环境中的科学计算库，推荐使用 `venv/bin/python` 运行。
- 若只需查看论文最终结果，通常只需打开 `results/` 目录中的 `.csv`、`.docx` 和 `.png` 文件，无需重新运行脚本。
- 若只想同步带图片的 Markdown 单文件版本，直接执行 `./scripts/update_embedded_markdown.sh` 即可。
