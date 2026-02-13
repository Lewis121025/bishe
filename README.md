# 基于数据挖掘的上市公司财政补贴与高管超额薪酬研究

## 项目简介

本项目是本科毕业论文的实证分析代码，研究政府补助对上市公司高管超额薪酬的影响，并探讨管理层权力的中介效应。研究方法包括计量经济学（行业/年份固定效应、公司层面聚类标准误、中介与机制检验、稳健性检验）和数据挖掘（随机森林、XGBoost、Lasso、K-Means 聚类）。

**参考论文**：刘剑民(2019)《政府补助、管理层权力与国有企业高管超额薪酬》

---

## 目录结构

```
.
├── data/                          # 原始数据（Stata .dta 格式，来自 CSMAR）
│   ├── 营业收入+净利润+总资产+无形资产+行业变量.dta
│   ├── 政府补助（两个文件上下拼接）-1.dta
│   ├── 政府补助（两个文件上下拼接）-2.dta
│   ├── 高管前三名薪酬总合.dta
│   ├── 两职合一+管理层持股比例+董事会规模+独立董事占比.dta
│   ├── 总经理任期年限（两个文件上下拼接）-1.dta
│   ├── 总经理任期年限（两个文件上下拼接）-2.dta
│   ├── 总负债.dta
│   ├── 所在地+公司性质.dta
│   ├── 第一大股东持股比例+实际控制人股权性质.dta
│   └── 刘剑民-2019-...pdf            # 参考论文 PDF
│
├── processed_data/                 # 转换后的 CSV 数据
│
├── scripts/                        # 分析脚本
│   ├── convert_data.py             # Step 1: .dta → .csv 格式转换
│   ├── validate_conversion.py      # Step 2: 转换结果验证
│   ├── regression_analysis.py      # Step 3: 计量经济学回归分析（核心）
│   ├── generate_tables.py          # Step 4: 生成论文格式回归表格
│   └── ml_analysis.py              # Step 5: 数据挖掘分析
│
├── results/                        # 分析结果
│   ├── regression_tables.txt       # 论文格式回归结果表
│   ├── regression_dataset.csv      # 含全部构造变量的最终数据集
│   ├── 论文实证结果摘要.md          # 可直接用于论文撰写的结果摘要
│   ├── model_comparison.csv        # ML 模型性能对比
│   ├── fig1_lasso_path.png         # Lasso 正则化路径图
│   ├── fig2_rf_importance.png      # 随机森林特征重要性
│   ├── fig3_shap_summary.png       # SHAP 特征重要性
│   ├── fig4_shap_subsidy.png       # 政府补助 SHAP 依赖图
│   ├── fig5_decision_tree.png      # 决策树可视化
│   ├── fig6_kmeans_clusters.png    # K-Means 聚类散点图
│   ├── fig7_cluster_heatmap.png    # 聚类中心热力图
│   └── fig8_model_comparison.png   # 模型对比柱状图
│
├── kaiti.md                        # 开题报告
├── 参考论文笔记_刘剑民2019.md       # 参考论文笔记
├── .gitignore
└── README.md                       # 本文件
```

---

## 运行方式

### 环境配置

```bash
# 1) 拉取 LFS 大文件（首次）
git lfs install
git lfs pull

# 2) macOS 上 xgboost 依赖（首次）
brew install libomp

# 3) Python 环境
python3 -m venv venv
source venv/bin/activate
pip install pandas pyreadstat numpy statsmodels scikit-learn xgboost shap matplotlib seaborn scipy
```

### 执行步骤

```bash
# 1. 数据转换（.dta → .csv）
python scripts/convert_data.py

# 2. 验证转换结果
python scripts/validate_conversion.py

# 3. 计量经济学回归分析（主回归 + 中介 + 异质性 + 机制 + 稳健性）
python scripts/regression_analysis.py

# 4. 生成论文格式表格（统一公司层面聚类标准误）
python scripts/generate_tables.py

# 5. 数据挖掘分析（随机森林 + XGBoost + SHAP + 聚类）
python scripts/ml_analysis.py
```

---

## 研究模型

### 计量经济学部分

| 模型 | 公式 | 用途 |
|:---|:---|:---|
| 模型1 | lnCEOpay = α₀ + α₁·lnSale + α₂·Roa + α₃·IA + α₄·Zone + ΣIndustry + ε | 期望薪酬（残差=超额薪酬） |
| 模型3 | Overpay = α₀ + α₁·lnSubsidy + Controls + ΣIndustry + ΣYear + ε | 基准回归（检验 H1） |
| 模型4 | Overpay = α₀ + α₁·lnSubsidy + α₂·Power + Controls + ΣIndustry + ΣYear + ε | 中介效应检验 |
| 模型5 | Power = α₀ + α₁·lnSubsidy + Controls + ΣIndustry + ΣYear + ε | 补助→权力路径 |

- 标准误：公司层面聚类稳健标准误
- 中介效应：Baron & Kenny 逐步法 + Sobel 检验
- 机制检验：管制行业/非管制行业、央企/地方国企
- 稳健性：替换被解释变量、样本期缩减、极值处理、制造业子样本、解释变量替换、滞后一期补助
- 管制行业口径：`B/D/G/I/N` 行业大类；央地口径：实控人代码 `2100=央企`、`2120=地方国企`

### 数据挖掘部分

| 方法 | 用途 |
|:---|:---|
| Lasso 回归 | 变量筛选与正则化 |
| 随机森林 | 超额薪酬预测与特征重要性 |
| XGBoost + SHAP | 预测 + 可解释性分析 |
| 决策树 | 风险路径可视化 |
| K-Means 聚类 | 企业画像与模式识别 |

---

## 数据来源

- **CSMAR（国泰安）数据库**
- 样本：2003-2024 年 A 股上市公司（剔除金融行业）
- 最终样本量：52,402 个公司-年观测值（当前口径下）
