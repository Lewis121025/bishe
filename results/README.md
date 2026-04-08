# results 目录说明

本目录保存论文实证分析的核心输出文件，可与正文中的表格和图形相互对应。

## 论文中最常用的结果文件

- `回归结果表.docx`：论文三线表 Word 版
- `regression_tables.txt`：论文格式的文本表格
- `descriptive_statistics.csv`：描述性统计
- `correlation_matrix.csv`：相关性分析
- `vif_results.csv`：多重共线性检验
- `causal_results.csv`：固定效应、滞后项与工具变量相关结果
- `main_mediation_summary.csv`：FA 口径管理层权力中介效应汇总
- `robustness_results.csv`：稳健性检验结果
- `heterogeneity_ownership_results.csv`：按产权性质分组的主回归结果
- `heterogeneity_mechanism_results.csv`：按行业管制、央地分类分组的主回归结果

## 机器学习补充分析

- `model_comparison.csv`：回归模型拟合表现对比
- `classification_comparison.csv`：分类模型辅助识别结果对比
- `lasso_coefficients.csv`：Lasso 系数筛选结果
- `rf_reg_importance.csv`、`rf_clf_importance.csv`：随机森林特征重要性
- `shap_importance.csv`：SHAP 重要性排序
- `ml_summary.json`、`ml_tuning_summary.json`：机器学习摘要与调参记录

## 图形文件

- `fig1_lasso_path.png`：Lasso 路径图
- `fig2_rf_importance.png`：随机森林特征重要性
- `fig3_shap_summary.png`：SHAP 总结图
- `fig4_shap_subsidy.png`：财政补贴 SHAP 依赖图
- `fig4a_shap_subsidy_issoe.png`、`fig4b_shap_subsidy_mgshder.png`：交互结构补充图
- `fig5_decision_tree.png`：决策树可视化
- `fig6_kmeans_clusters.png`、`fig7_cluster_heatmap.png`：聚类分析图
- `fig8_model_comparison.png`：模型对比图

## 辅助性目录

- `mineru_正文修改/`：PDF 转 Markdown 过程文件，属于排版处理辅助产物
- `pdf_page_renders/`：PDF 页面渲染缓存

以上两个目录不影响论文结论复现，主要用于排版和校对过程留存。
