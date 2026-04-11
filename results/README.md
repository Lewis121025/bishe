# results 目录说明

本目录保存论文实证分析的核心输出文件，可与正文中的表格和图形相互对应。

## 论文中最常用的结果文件

- `回归结果表.docx`：论文三线表 Word 版
- `regression_tables.txt`：论文格式的文本表格
- `descriptive_statistics.csv`：描述性统计
- `correlation_matrix.csv`：相关性分析
- `vif_results.csv`：多重共线性检验
- `diagnostics_summary.json`：正文会引用的模型诊断值汇总（如模型1调整后R²、模型2 F统计量、VIF 最大值/均值）
- `causal_results.csv`：固定效应、滞后项与工具变量相关结果
- `main_mediation_summary.csv`：FA 口径管理层权力中介效应汇总
- `robustness_results.csv`：稳健性检验结果
- `heterogeneity_ownership_results.csv`：按产权性质分组的主回归结果
- `heterogeneity_mechanism_results.csv`：按行业管制、央地分类分组的主回归结果

## 机器学习稳健性检验

- `lasso_alpha_search.csv`：Lasso 惩罚参数搜索结果
- `lasso_coefficients.csv`：Lasso 系数与保留情况
- `rf_reg_importance.csv`：随机森林回归特征重要性
- `xgb_importance.csv`：XGBoost 特征重要性排序
- `xgb_partial_dependence.csv`：XGBoost 对财政补贴变量的部分依赖曲线
- `ml_validation_summary.csv`、`ml_validation_summary.json`：三种模型对 OLS 主回归的稳健性检验摘要
- `ml_tuning_summary.json`：分组划分、交叉验证与调参记录

## 图形文件

- `fig1_lasso_path.png`：Lasso 系数收缩图
- `fig2_rf_importance.png`：随机森林特征重要性图
- `fig4_shap_subsidy.png`：XGBoost 特征重要性与财政补贴部分依赖图

## 说明

- 本目录默认仅保留与当前正文口径一致的结果文件和图形文件。
- 旧版分类、聚类和模型对比结果文件若仍存在，可视为历史输出，不再对应当前论文正文。
- 旧值整理稿、PDF 转 Markdown 中间文件和其他排版缓存不再保留，以避免误用过时结果。
