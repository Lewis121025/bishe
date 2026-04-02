# Gemini 审稿意见逐条核验报告

## 总评

本次核验以 [正文.md](/Users/lewis/毕业论文/正文.md)、[scripts/regression_analysis.py](/Users/lewis/毕业论文/scripts/regression_analysis.py)、[scripts/ml_analysis.py](/Users/lewis/毕业论文/scripts/ml_analysis.py)、[results/regression_dataset.csv](/Users/lewis/毕业论文/results/regression_dataset.csv)、[results/main_mediation_summary.csv](/Users/lewis/毕业论文/results/main_mediation_summary.csv)、[results/power_method_comparison.csv](/Users/lewis/毕业论文/results/power_method_comparison.csv) 和 [results/model_comparison.csv](/Users/lewis/毕业论文/results/model_comparison.csv) 为主要证据源。

总体判断如下：

- Gemini 抓中了几项真实且重要的风险：因果语言过强、中介分析被写成因果机制、机器学习结果被越界解释为“机制验证”和“阈值效应”、以及 H2 对权力测度高度敏感却仍被写成“支持假设”。
- 但 Gemini 也有几处明显说过头：尤其是“二阶段 `Roa` 系数数学上必须严格为 0，所以代码必错”“当前切分必然把同一公司 2020 年训练去预测其 2010 年”“这是典型 P-hacking / 学术不端”“建议全部推导不成立”。这些表述都超出了现有证据能支持的范围。
- 更准确的结论不是“论文全部失效”，而是：这篇论文的**识别口径、机制表述和机器学习解释存在严重的学术表述与方法边界问题**；其中有些属于**严重但可修**，有些只是**措辞夸大**，真正能直接判成“致命错误”的部分没有 Gemini 说得那么多。

本文将问题分为三档：

- `致命错误`：足以直接推翻该部分结论，且难以仅靠改写措辞补救。
- `严重但可修`：核心写法或解释存在明显问题，但通过重估模型、降级结论、补做说明仍可修复。
- `措辞夸大`：结论写得过强，但现有证据不足以支持“造假”“学术不端”等更重指控。

## 1. “残差二次回归的数学悖论”

- Gemini 原指控：因为 `Overpay` 是第一阶段残差，所以第二阶段再回归 `Roa` 和 `Zone` 时，其系数数学上必须严格为 0；表 4-3 中 `Roa=-0.3893` 说明代码有根本错误。
- 核验结论：`部分成立`。
- 证据位置：
  - [正文.md:141](/Users/lewis/毕业论文/正文.md#L141) 到 [正文.md:145](/Users/lewis/毕业论文/正文.md#L145) 明确把 `Overpay` 定义为“实际薪酬对数与拟合值之差”。
  - [正文.md:197](/Users/lewis/毕业论文/正文.md#L197) 到 [正文.md:205](/Users/lewis/毕业论文/正文.md#L205) 显示第一阶段包含 `lnSale`、`Roa`、`IA`、`Zone`，第二阶段则换成 `lnSubsidy`、`Roa`、`Lever`、`Top1`、`Zone`。
  - [scripts/regression_analysis.py:603](/Users/lewis/毕业论文/scripts/regression_analysis.py#L603) 到 [scripts/regression_analysis.py:649](/Users/lewis/毕业论文/scripts/regression_analysis.py#L649) 证明 `Overpay` 的确直接取自第一阶段残差。
  - [scripts/regression_analysis.py:1036](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1036) 到 [scripts/regression_analysis.py:1054](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1054) 与 [scripts/regression_analysis.py:1221](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1221) 到 [scripts/regression_analysis.py:1245](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1245) 证明主回归采用了与第一阶段不同的控制集，并且统一到了 `Power` 可用样本。
  - [正文.md:303](/Users/lewis/毕业论文/正文.md#L303) 到 [正文.md:318](/Users/lewis/毕业论文/正文.md#L318) 报告了 `Roa=-0.3893***`。
- 结果数值核验：
  - 对 [results/regression_dataset.csv](/Users/lewis/毕业论文/results/regression_dataset.csv#L1) 直接复核后发现，在第一阶段原样本 `N=52,182` 中，`corr(Overpay, Roa)≈-3.8e-16`、`corr(Overpay, Zone)≈1.6e-15`，说明第一阶段残差与第一阶段解释变量在原样本内确实正交。
  - 但在主回归统一样本 `N=44,831` 中，上述正交性已不再精确成立；同时第二阶段还加入了 `lnSubsidy / Lever / Top1`，并移除了第一阶段的 `lnSale / IA`。因此，Gemini 所说“第二阶段系数数学上必须严格为 0”并不成立。
- 更准确的学术表述：
  - 真实问题不是“代码一看就错”，而是**第二阶段模型不再是对第一阶段残差性质的纯检验**。一旦样本切换且控制集变化，`Roa` 的偏回归系数不必为 0。
  - 但正文把 `Roa` 的负系数解释为“绩效越好，超额攫取越少”的经济机制，这个说法站不住。该系数只是当前第二阶段设定下的条件相关，不是残差定义本身推出的行为解释。
- 对论文的实际影响等级：`严重但可修`。

## 2. “机器学习存在未来穿越与数据泄露”

- Gemini 原指控：作者采用 80/20 group split 和 5 折交叉验证，会把同一公司未来年份用于训练、过去年份用于测试，属于严重 look-ahead bias。
- 核验结论：`部分成立`。
- 证据位置：
  - [正文.md:445](/Users/lewis/毕业论文/正文.md#L445) 到 [正文.md:449](/Users/lewis/毕业论文/正文.md#L449) 说明使用公司分组的 80/20 切分。
  - [scripts/ml_analysis.py:158](/Users/lewis/毕业论文/scripts/ml_analysis.py#L158) 到 [scripts/ml_analysis.py:175](/Users/lewis/毕业论文/scripts/ml_analysis.py#L175) 证明测试集切分使用 `GroupShuffleSplit`，按公司隔离训练集与测试集。
  - [scripts/ml_analysis.py:184](/Users/lewis/毕业论文/scripts/ml_analysis.py#L184) 到 [scripts/ml_analysis.py:207](/Users/lewis/毕业论文/scripts/ml_analysis.py#L207) 显示 Lasso 的最终评估用了 `GroupKFold`，但 `LassoCV` 的内部选参回退成了普通 `KFold`。
  - [scripts/ml_analysis.py:291](/Users/lewis/毕业论文/scripts/ml_analysis.py#L291) 到 [scripts/ml_analysis.py:315](/Users/lewis/毕业论文/scripts/ml_analysis.py#L315)、[scripts/ml_analysis.py:390](/Users/lewis/毕业论文/scripts/ml_analysis.py#L390) 到 [scripts/ml_analysis.py:419](/Users/lewis/毕业论文/scripts/ml_analysis.py#L419)、[scripts/ml_analysis.py:468](/Users/lewis/毕业论文/scripts/ml_analysis.py#L468) 到 [scripts/ml_analysis.py:523](/Users/lewis/毕业论文/scripts/ml_analysis.py#L523) 显示随机森林、XGBoost、分类模型的交叉验证均使用 `GroupKFold`。
- 结果数值核验：
  - [results/model_comparison.csv:2](/Users/lewis/毕业论文/results/model_comparison.csv#L2) 到 [results/model_comparison.csv:5](/Users/lewis/毕业论文/results/model_comparison.csv#L5) 表明机器学习性能确实来自当前这套分组方案。
  - 代码层面没有证据支持 Gemini 那句“同一公司 2020 年数据用于训练、预测其 2010 年数据”，因为 holdout 已按公司隔离。
- 更准确的学术表述：
  - Gemini 指控的**同公司跨期泄露**不成立。
  - 但更合理的批评是两点：
  - 第一，所有机器学习验证都**没有做时间感知切分**。如果要声称对 2003—2024 面板数据具有严格的时序预测意义，那么当前方案仍存在跨年份的前视偏差风险。
  - 第二，Lasso 的内部调参确实退回了普通 `KFold`，这会造成组间信息混入，方法上不够严谨。
- 对论文的实际影响等级：`严重但可修`。

## 3. “在内生性未解决下滥用因果中介分析”

- Gemini 原指控：在没有严格因果识别的前提下使用 Baron & Kenny、Sobel 和 bootstrap 证明“中介效应”，属于方法滥用。
- 核验结论：`成立`。
- 证据位置：
  - [正文.md:69](/Users/lewis/毕业论文/正文.md#L69) 把第三层次写成“机制检验”。
  - [正文.md:211](/Users/lewis/毕业论文/正文.md#L211) 到 [正文.md:229](/Users/lewis/毕业论文/正文.md#L229) 明确采用 Baron & Kenny 三步法、Sobel 和 bootstrap。
  - [正文.md:405](/Users/lewis/毕业论文/正文.md#L405) 到 [正文.md:421](/Users/lewis/毕业论文/正文.md#L421) 明确写出“支持假设 H2”“完整机制”。
  - [正文.md:519](/Users/lewis/毕业论文/正文.md#L519) 与 [正文.md:537](/Users/lewis/毕业论文/正文.md#L537) 又承认“尚未实现严格因果识别”。
  - [results/main_mediation_summary.csv:2](/Users/lewis/毕业论文/results/main_mediation_summary.csv#L2) 仅能证明在 FA 口径下存在统计上的路径关联，不能单独赋予因果含义。
- 更准确的学术表述：
  - 当前结果最多支持“**相关性路径分析**”或“**关联式中介检验**”。
  - 不能把它写成“补贴通过权力扩张间接推高超额薪酬”的因果机制，更不能据此推出政策干预链条。
  - Gemini 在“方法边界”上批得对，但把它直接上升为“学术不端”则过重。
- 对论文的实际影响等级：`严重但可修`。

## 4. “P-hacking / 结果挑选以迎合 H2”

- Gemini 原指控：FA 显著、PCA 不显著、熵值法方向相反，作者却主观选择支持 H2 的 FA 结果，是典型 P-hacking。
- 核验结论：`部分成立`。
- 证据位置：
  - [正文.md:169](/Users/lewis/毕业论文/正文.md#L169) 预先声明主文分析以 FA 为准，PCA 和熵值法作为敏感性对照。
  - [正文.md:425](/Users/lewis/毕业论文/正文.md#L425) 到 [正文.md:441](/Users/lewis/毕业论文/正文.md#L441) 公开展示了三种口径下完全不同的结果。
  - [results/power_method_comparison.csv:2](/Users/lewis/毕业论文/results/power_method_comparison.csv#L2) 到 [results/power_method_comparison.csv:4](/Users/lewis/毕业论文/results/power_method_comparison.csv#L4) 明确显示：FA 支持、PCA 不支持、熵值法为遮掩方向。
  - [正文.md:513](/Users/lewis/毕业论文/正文.md#L513) 仍然把 H2 写成“具有统计意义”，只是附加“高度敏感”的保留语。
- 更准确的学术表述：
  - 这不构成严格意义上的“隐瞒不利结果”，因为相反结果在正文里确实被展示了。
  - 但它确实构成**结论过强**：在三种口径结论相互冲突时，最稳妥的写法应是“**H2 仅在 FA 口径下得到支持，整体上不稳健**”，而不是继续在摘要和结论中把 H2 当作成立命题。
  - “P-hacking/学术不端”这一指控证据不足；“稳健性不足而结论降级不够”才是更准确的批评。
- 对论文的实际影响等级：`严重但可修`。

## 5. “3.27% 的中介占比属于统计噪音，被严重夸大”

- Gemini 原指控：3.27% 纯属噪音，不具备经济意义，却被上升为核心机制。
- 核验结论：`部分成立`。
- 证据位置：
  - [正文.md:417](/Users/lewis/毕业论文/正文.md#L417) 到 [正文.md:421](/Users/lewis/毕业论文/正文.md#L421) 把 3.27% 写成“真实存在的传导路径”。
  - [正文.md:513](/Users/lewis/毕业论文/正文.md#L513) 延续了这一表述。
  - [results/main_mediation_summary.csv:2](/Users/lewis/毕业论文/results/main_mediation_summary.csv#L2) 证实中介占比约为 `3.2699%`。
- 更准确的学术表述：
  - 3.27% 是**较小的部分中介**，说明即便 FA 口径成立，权力路径也不是主导机制。
  - 但不能仅凭“3.27% 很小”就说它是“纯统计噪音”或“没有任何经济意义”；这一步 Gemini 说得过满。
  - 真正成立的批评是：在“效应很小且对测度敏感”的前提下，正文不应把它写成强机制，更不宜据此推出多条政策建议。
- 对论文的实际影响等级：`措辞夸大`。

## 6. “样本量前后矛盾、缺乏数学依据”

- Gemini 原指控：52,358、52,182、44,831、44,650 等样本量变化没有严谨解释。
- 核验结论：`部分成立`。
- 证据位置：
  - [正文.md:135](/Users/lewis/毕业论文/正文.md#L135) 声称期望薪酬模型最终保留 52,358 条，主回归 44,831 条。
  - [正文.md:241](/Users/lewis/毕业论文/正文.md#L241) 又说期望薪酬估计样本为 52,182 条。
  - [正文.md:447](/Users/lewis/毕业论文/正文.md#L447) 说机器学习样本为 44,650 条，并称与主回归相差 181 条系“完整案例筛选的自然缩减”。
  - [scripts/regression_analysis.py:614](/Users/lewis/毕业论文/scripts/regression_analysis.py#L614) 说明第一阶段实际用的是对 `lnSale / Roa / IA / Zone / lnCEOpay / IndustrySector` 完整案例筛选后的样本。
  - [scripts/regression_analysis.py:1042](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1042) 与 [scripts/regression_analysis.py:1243](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1243) 说明主回归统一样本由 `lnSubsidy / Power / Roa / Lever / Top1 / Zone / Overpay / IndustrySector` 的完整案例决定。
  - [scripts/ml_analysis.py:128](/Users/lewis/毕业论文/scripts/ml_analysis.py#L128) 到 [scripts/ml_analysis.py:130](/Users/lewis/毕业论文/scripts/ml_analysis.py#L130) 说明机器学习样本还要求 `IsSOE` 等特征完整。
- 结果数值核验：
  - 对 [results/regression_dataset.csv](/Users/lewis/毕业论文/results/regression_dataset.csv#L1) 直接统计后可解释这些差异：
  - `52,358 -> 52,182` 的 176 条缺口，来自 `lnSale` 缺失 17 条和 `IA` 缺失 159 条。
  - `52,182 -> 44,831` 的 7,351 条缺口，全部来自 `Power` 缺失。
  - `44,831 -> 44,650` 的 181 条缺口，全部来自 `IsSOE` 缺失。
- 更准确的学术表述：
  - 这不是“无端丢失样本”或“数据造假”的证据。
  - 但正文确实写得不自洽：第 3.3 节把 52,358 写成“用于期望薪酬模型估计”的最终样本，而代码和表 4-1 实际使用的是 52,182。对 44,650 与 44,831 的差异，也应明确写出具体缺失变量，而不是只说“自然缩减”。
- 对论文的实际影响等级：`严重但可修`。

## 7. “缩尾标准双标混乱”

- Gemini 原指控：薪酬沿用 CSMAR 口径，其他连续变量再做 1%/99% 缩尾，会破坏联合分布。
- 核验结论：`不成立（但信息披露不充分）`。
- 证据位置：
  - [正文.md:135](/Users/lewis/毕业论文/正文.md#L135) 明确说明原始薪酬沿用 CSMAR 既有口径，其他连续变量再缩尾。
  - [正文.md:255](/Users/lewis/毕业论文/正文.md#L255) 再次说明 `Overpay` 不做二次缩尾。
  - [scripts/regression_analysis.py:553](/Users/lewis/毕业论文/scripts/regression_analysis.py#L553) 到 [scripts/regression_analysis.py:555](/Users/lewis/毕业论文/scripts/regression_analysis.py#L555) 代码层面也明确写了“不再对 lnCEOpay / Overpay 二次缩尾”。
- 更准确的学术表述：
  - 这种处理方式本身不能自动推出“联合分布被破坏”或“结果失效”。很多实证研究确实会保留数据库预处理后的薪酬变量，并对其他研究者自行构造的连续变量再做统一 winsorize。
  - 更成立的批评是：作者应更明确交代 CSMAR 对薪酬变量做过什么处理，以及为何不对其他变量采用完全相同口径，并补做敏感性检验。
- 对论文的实际影响等级：`措辞夸大`。

## 8. “因果术语的学术话术欺骗”

- Gemini 原指控：正文广泛使用“推高效应”“影响机制”“传导链条”等因果词，最后却承认没有严格因果识别。
- 核验结论：`成立`。
- 证据位置：
  - [正文.md:7](/Users/lewis/毕业论文/正文.md#L7) 摘要里连续使用“影响机制”“推高效应”“为中介机制提供旁证”。
  - [正文.md:27](/Users/lewis/毕业论文/正文.md#L27) 把研究方法写成“实现了线性因果识别与非线性特征刻画的优势互补”。
  - [正文.md:81](/Users/lewis/毕业论文/正文.md#L81) 甚至把财政补贴描述成“理想准自然实验场景”。
  - [正文.md:509](/Users/lewis/毕业论文/正文.md#L509) 到 [正文.md:517](/Users/lewis/毕业论文/正文.md#L517) 结论部分继续使用“推高效应”“机制”“验证了假设 H1/H2”等措辞。
  - [正文.md:519](/Users/lewis/毕业论文/正文.md#L519) 与 [正文.md:537](/Users/lewis/毕业论文/正文.md#L537) 又承认尚未实现严格因果识别。
- 更准确的学术表述：
  - 这是本文最需要统一修正的地方之一。全文应系统降级为“正相关”“条件相关”“关联路径”“解释性证据”，避免“效应”“传导链条”“验证机制”。
  - Gemini 对“语言前强后弱”的批评是对的，但“学术话术欺骗”这类定性仍然偏重。更适合的表述是：**识别强度与叙述强度不匹配**。
- 对论文的实际影响等级：`严重但可修`。

## 9. “SHAP 被非科学地解读为现实阈值效应和代理问题”

- Gemini 原指控：作者把 SHAP 依赖图和交互重要性直接解释成现实中的阈值效应、机制印证和代理问题，这越过了模型解释边界。
- 核验结论：`成立`。
- 证据位置：
  - [正文.md:463](/Users/lewis/毕业论文/正文.md#L463) 把 SHAP 依赖图写成“显著的非线性”“类阶梯型阈值效应”并称“为 H1 提供深化佐证”。
  - [正文.md:467](/Users/lewis/毕业论文/正文.md#L467) 到 [正文.md:471](/Users/lewis/毕业论文/正文.md#L471) 把 SHAP 交互重要性写成“为 H2 提供定量补充证据”“严谨印证异质性结论”。
  - [正文.md:501](/Users/lewis/毕业论文/正文.md#L501) 继续把机器学习定位为“机制检验与稳健性验证”。
  - [scripts/ml_analysis.py:441](/Users/lewis/毕业论文/scripts/ml_analysis.py#L441) 到 [scripts/ml_analysis.py:449](/Users/lewis/毕业论文/scripts/ml_analysis.py#L449) 表明这里输出的只是模型 SHAP 依赖图。
  - [results/model_comparison.csv:2](/Users/lewis/毕业论文/results/model_comparison.csv#L2) 到 [results/model_comparison.csv:5](/Users/lewis/毕业论文/results/model_comparison.csv#L5) 说明模型预测力本身也只是中等水平，并不支持对现实机制做强解释。
- 更准确的学术表述：
  - SHAP 只能解释“在该模型中，哪些特征更影响预测值，以及这种影响如何随特征取值变化”，不能单独证明现实世界存在结构性阈值，更不能替代因果识别。
  - 本文在机器学习部分最合适的定位应是“**预测性补充分析**”或“**模型层面的可解释性描述**”，而不是“机制验证”“稳健性确认”。
- 对论文的实际影响等级：`严重但可修`。

## 10. “建议全部推导不成立”

- Gemini 原结论：由于上述问题严重，建议全部推导不成立。
- 核验结论：`不成立`。
- 依据：
  - 主效应部分的正相关结果在代码、数据和多组稳健性检验中是一致的，见 [正文.md:303](/Users/lewis/毕业论文/正文.md#L303) 到 [正文.md:318](/Users/lewis/毕业论文/正文.md#L318)、[scripts/regression_analysis.py:1517](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1517) 到 [scripts/regression_analysis.py:1598](/Users/lewis/毕业论文/scripts/regression_analysis.py#L1598)。
  - 中介效应部分在 FA 口径下确有统计关联证据，见 [results/main_mediation_summary.csv:2](/Users/lewis/毕业论文/results/main_mediation_summary.csv#L2)。
  - 真正失效的是“把这些相关性和模型解释写成稳健机制乃至政策因果”的叙述方式，不是所有数值结果都自动无效。
- 更准确的总体结论：
  - 这篇论文当前版本**不能支撑强因果机制和强政策含义**。
  - 但它仍可支撑一个更弱、也更诚实的结论：在所设控制变量和固定效应条件下，财政补贴与高管超额薪酬之间存在稳定的正相关；至于“管理层权力中介”“阈值效应”“机制印证”，均应显著降级表述。
- 对论文的实际影响等级：`措辞夸大`。

## 修订优先级排序

建议按下面顺序修：

- 第一优先级：把全文因果语言降级，尤其是摘要、研究意义、结论、政策建议和机器学习章节。
- 第二优先级：把 H2 改写为“FA 口径下的关联式路径证据”，删除“支持假设 H2”的强表述。
- 第三优先级：重写机器学习章节定位，明确其是预测性补充分析，不是机制验证；并说明当前没有做时间感知切分。
- 第四优先级：把样本量差异写清楚，明确 `52,358 -> 52,182 -> 44,831 -> 44,650` 的具体缺失来源。
- 第五优先级：弱化对 `Roa` 负系数、SHAP 阈值、交互项的行为学解释。

## 如果要对导师解释，最稳妥的说法

最稳妥的口径是：

“Gemini 的批评里，关于因果语言过强、机器学习结果被写成机制验证、以及 H2 对测度高度敏感却仍被写成成立，这几项是有实质依据的，我需要系统降级表述。  
但它也有几处说过头，比如把二阶段 `Roa` 非零直接判成代码错误、把当前分组切分说成同公司跨期泄露、以及把问题直接上升到 P-hacking 和全部推导不成立，这些都超出了现有代码和结果能支持的范围。  
更准确的结论不是论文完全无效，而是当前版本只能支撑较弱的条件相关性结论，不能支撑强机制和强政策含义。” 

