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
  - 主要输出：多个回归结果汇总文件；运行时会生成 `results/regression_dataset.csv` 中间数据，但归档包默认不保留

- `generate_tables.py`
  - 作用：将回归结果整理成论文文本表格
  - 输出：`results/regression_tables.txt`

- `generate_tables_docx.py`
  - 作用：将主要实证结果导出为 Word 三线表
  - 输出：`results/回归结果表.docx`

- `ml_analysis.py`
  - 作用：围绕 OLS 主回归完成随机森林、Lasso、XGBoost 三种机器学习稳健性检验
  - 主要输出：`results/lasso_coefficients.csv`、`results/rf_reg_importance.csv`、`results/xgb_importance.csv`、`results/xgb_partial_dependence.csv` 及机器学习验证摘要文件

## 辅助脚本

- `panel_regressions.py`
  - 作用：基于主回归脚本生成的 `results/regression_dataset.csv` 做补充性的纯 Python 面板回归检验
  - 说明：现已改为仓库相对路径，可在不同机器上直接运行

## 推荐执行顺序

```bash
python scripts/convert_data.py
python scripts/validate_conversion.py
python scripts/regression_analysis.py
python scripts/generate_tables.py
python scripts/generate_tables_docx.py
python scripts/ml_analysis.py
```

## 使用说明

- 所有脚本都默认以仓库目录为根，不要求必须从某一台特定电脑或绝对路径运行。
- 当前归档包不保留 `processed_data/` 中间 CSV；如需复现，请先运行 `convert_data.py`，再运行后续分析脚本。
- 当前归档包也不预置 `results/` 结果文件；如需核对论文结果，请按推荐顺序重新运行脚本生成。
- `ml_analysis.py` 依赖 `scikit-learn`、`shap` 和 `xgboost`，运行时间通常长于回归脚本。
