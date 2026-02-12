"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 机器学习分析脚本
=============================================================================

分析内容：
  1. Lasso 回归：变量筛选与正则化
  2. 随机森林：超额薪酬预测与特征重要性
  3. XGBoost：梯度提升模型与 SHAP 可解释性分析
  4. 决策树可视化：直观展示高超额薪酬的条件路径
  5. K-Means 聚类：企业画像与模式识别
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, classification_report, confusion_matrix,
                              silhouette_score)
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 导入数据加载函数
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from regression_analysis import (load_and_clean_data, construct_variables,
                                  filter_and_describe, compute_overpay,
                                  compute_power)


def prepare_ml_data(df):
    """准备机器学习所需的特征矩阵和标签"""
    print("\n" + "=" * 70)
    print("数据准备：构建特征矩阵")
    print("=" * 70)

    # 特征列
    feature_cols = ["lnSubsidy", "Roa", "Lever", "Top1", "Zone", "Industry",
                    "lnSale", "IA", "Boardsize", "Dual", "Insider",
                    "Mgshder", "Tenure", "IsSOE"]

    # 目标变量
    target_col = "Overpay"

    # 筛选完整数据
    use_cols = feature_cols + [target_col, "Symbol", "Year", "SubsidyAmount", "Top3Salary"]
    df_ml = df.dropna(subset=feature_cols + [target_col]).copy()

    print(f"  可用样本量: {len(df_ml)}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  特征列表: {feature_cols}")

    X = df_ml[feature_cols].values
    y = df_ml[target_col].values
    feature_names = feature_cols

    # 构建分类标签：超额薪酬 > 0 为正 (1)，否则为负 (0)
    y_class = (y > 0).astype(int)
    print(f"  超额薪酬为正的比例: {y_class.mean():.2%}")

    return df_ml, X, y, y_class, feature_names


# ============================================================
# 第一部分：Lasso 回归 — 变量筛选
# ============================================================

def lasso_analysis(X, y, feature_names, output_dir):
    """Lasso 回归用于变量筛选和正则化"""
    print("\n" + "=" * 70)
    print("第一部分：Lasso 回归 — 变量筛选与正则化")
    print("=" * 70)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 交叉验证选择最优 lambda (alpha)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y)

    print(f"\n  最优正则化参数 α = {lasso_cv.alpha_:.6f}")
    print(f"  R² (交叉验证) = {lasso_cv.score(X_scaled, y):.4f}")

    # 系数分析
    coefs = pd.DataFrame({
        "变量": feature_names,
        "Lasso系数": lasso_cv.coef_
    }).sort_values("Lasso系数", key=abs, ascending=False)

    print("\n  Lasso 回归系数（按绝对值排序）:")
    print("  " + "-" * 40)
    for _, row in coefs.iterrows():
        status = "✓ 保留" if abs(row["Lasso系数"]) > 1e-6 else "✗ 剔除"
        print(f"    {row['变量']:15s}: {row['Lasso系数']:>10.4f}  {status}")

    retained = coefs[abs(coefs["Lasso系数"]) > 1e-6]["变量"].tolist()
    removed = coefs[abs(coefs["Lasso系数"]) <= 1e-6]["变量"].tolist()
    print(f"\n  保留变量 ({len(retained)}): {retained}")
    print(f"  剔除变量 ({len(removed)}): {removed}")

    # 画系数路径图
    alphas = np.logspace(-5, 0, 100)
    coef_paths = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        coef_paths.append(lasso.coef_)
    coef_paths = np.array(coef_paths)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coef_paths[:, i], label=name)
    ax.axvline(lasso_cv.alpha_, color='black', linestyle='--', label=f'最优α={lasso_cv.alpha_:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('正则化参数 α (log)')
    ax.set_ylabel('Lasso 回归系数')
    ax.set_title('图1  Lasso 正则化路径图')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_lasso_path.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig1_lasso_path.png")

    return retained


# ============================================================
# 第二部分：随机森林 — 回归预测与特征重要性
# ============================================================

def random_forest_analysis(X, y, y_class, feature_names, output_dir):
    """随机森林回归 + 分类 + 特征重要性"""
    print("\n" + "=" * 70)
    print("第二部分：随机森林 — 预测与特征重要性")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42)

    # --- 2a. 随机森林回归 ---
    print("\n  [2a] 随机森林回归（预测超额薪酬数值）")
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    y_pred_reg = rf_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    mae = mean_absolute_error(y_test, y_pred_reg)

    print(f"    R² = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE = {mae:.4f}")

    # 交叉验证
    cv_scores = cross_val_score(rf_reg, X, y, cv=5, scoring='r2')
    print(f"    5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- 2b. 随机森林分类 ---
    print("\n  [2b] 随机森林分类（预测是否存在超额薪酬）")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, yc_train)
    yc_pred = rf_clf.predict(X_test)

    acc = accuracy_score(yc_test, yc_pred)
    print(f"    准确率 = {acc:.4f}")
    print(f"\n    分类报告:")
    print(f"    {'':4}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    report = classification_report(yc_test, yc_pred, output_dict=True)
    for label in ['0', '1']:
        r = report[label]
        lab = "无超额薪酬" if label == '0' else "有超额薪酬"
        print(f"    {lab:8}{r['precision']:>10.4f}{r['recall']:>10.4f}{r['f1-score']:>10.4f}{r['support']:>10.0f}")
    print(f"    {'总体准确率':8}{'':>10}{'':>10}{report['accuracy']:>10.4f}{len(yc_test):>10}")

    # --- 特征重要性 ---
    importance_reg = pd.DataFrame({
        "变量": feature_names,
        "回归重要性": rf_reg.feature_importances_
    }).sort_values("回归重要性", ascending=False)

    importance_clf = pd.DataFrame({
        "变量": feature_names,
        "分类重要性": rf_clf.feature_importances_
    }).sort_values("分类重要性", ascending=False)

    print("\n  特征重要性排名（随机森林回归）:")
    print("  " + "-" * 35)
    for i, (_, row) in enumerate(importance_reg.iterrows()):
        bar = "█" * int(row["回归重要性"] * 50)
        print(f"    {i+1:2d}. {row['变量']:15s}: {row['回归重要性']:.4f} {bar}")

    # 画特征重要性对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 回归
    imp_reg_sorted = importance_reg.sort_values("回归重要性", ascending=True)
    axes[0].barh(imp_reg_sorted["变量"], imp_reg_sorted["回归重要性"], color='steelblue')
    axes[0].set_title(f'(a) 随机森林回归特征重要性\nR²={r2:.4f}')
    axes[0].set_xlabel('重要性')

    # 分类
    imp_clf_sorted = importance_clf.sort_values("分类重要性", ascending=True)
    axes[1].barh(imp_clf_sorted["变量"], imp_clf_sorted["分类重要性"], color='coral')
    axes[1].set_title(f'(b) 随机森林分类特征重要性\nAccuracy={acc:.4f}')
    axes[1].set_xlabel('重要性')

    plt.suptitle('图2  随机森林特征重要性分析', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_rf_importance.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig2_rf_importance.png")

    return rf_reg, rf_clf, importance_reg


# ============================================================
# 第三部分：XGBoost + SHAP 可解释性
# ============================================================

def xgboost_shap_analysis(X, y, feature_names, output_dir):
    """XGBoost 回归 + SHAP 值分析"""
    print("\n" + "=" * 70)
    print("第三部分：XGBoost + SHAP 可解释性分析")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # XGBoost 回归
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                   verbose=False)

    y_pred = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  XGBoost 回归结果:")
    print(f"    R² = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")

    cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
    print(f"    5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # SHAP 分析
    print("\n  计算 SHAP 值...")
    explainer = shap.TreeExplainer(xgb_model)
    # 用测试集的一个子集来计算 SHAP（加速）
    sample_size = min(2000, len(X_test))
    X_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # SHAP 重要性排名
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "变量": feature_names,
        "SHAP均值": shap_importance
    }).sort_values("SHAP均值", ascending=False)

    print("\n  SHAP 特征重要性排名:")
    print("  " + "-" * 35)
    for i, (_, row) in enumerate(shap_df.iterrows()):
        bar = "█" * int(row["SHAP均值"] / shap_df["SHAP均值"].max() * 30)
        print(f"    {i+1:2d}. {row['变量']:15s}: {row['SHAP均值']:.4f} {bar}")

    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                       show=False, max_display=14)
    plt.title('图3  SHAP 特征重要性分析（XGBoost）', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_shap_summary.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig3_shap_summary.png")

    # SHAP 分析 lnSubsidy 的边际效应
    fig, ax = plt.subplots(figsize=(8, 5))
    subsidy_idx = feature_names.index("lnSubsidy")
    shap.dependence_plot(subsidy_idx, shap_values, X_sample,
                          feature_names=feature_names, show=False, ax=ax)
    ax.set_title('图4  政府补助(lnSubsidy)的SHAP依赖图', fontsize=13)
    ax.set_xlabel('lnSubsidy (政府补助对数)')
    ax.set_ylabel('SHAP值 (对超额薪酬的边际影响)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_shap_subsidy.png"), bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig4_shap_subsidy.png")

    return xgb_model, shap_df


# ============================================================
# 第四部分：决策树可视化
# ============================================================

def decision_tree_analysis(X, y_class, feature_names, output_dir):
    """决策树分类 — 可视化超额薪酬风险路径"""
    print("\n" + "=" * 70)
    print("第四部分：决策树 — 超额薪酬风险路径")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42)

    # 浅层决策树（便于可视化）
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100,
                                 random_state=42)
    dt.fit(X_train, y_train)

    acc = dt.score(X_test, y_test)
    print(f"\n  决策树准确率: {acc:.4f}")

    # 文本规则
    tree_rules = export_text(dt, feature_names=feature_names, max_depth=4)
    print(f"\n  决策树规则:")
    for line in tree_rules.split('\n')[:20]:
        print(f"    {line}")
    if len(tree_rules.split('\n')) > 20:
        print(f"    ... (共 {len(tree_rules.split(chr(10)))} 行)")

    # 可视化
    cn = ['无超额薪酬', '有超额薪酬']
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(dt, feature_names=feature_names, class_names=cn,
              filled=True, rounded=True, fontsize=8, ax=ax,
              proportion=True, impurity=False)
    ax.set_title('图5  决策树：高管超额薪酬风险分类路径', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_decision_tree.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig5_decision_tree.png")

    return dt


# ============================================================
# 第五部分：K-Means 聚类 — 企业画像
# ============================================================

def kmeans_analysis(df_ml, feature_names, output_dir):
    """K-Means 聚类分析 — 企业分群画像"""
    print("\n" + "=" * 70)
    print("第五部分：K-Means 聚类 — 企业画像分析")
    print("=" * 70)

    cluster_features = ["lnSubsidy", "Overpay", "Roa", "Lever", "Top1", "lnSale"]
    df_cluster = df_ml.dropna(subset=cluster_features).copy()

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster[cluster_features])

    # 肘部法则选 K
    print("\n  肘部法则确定最优 K:")
    inertias = []
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_cluster, km.labels_, sample_size=5000)
        sil_scores.append(sil)
        print(f"    K={k}: 轮廓系数={sil:.4f}, 惯性={km.inertia_:.0f}")

    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"\n  最优K（轮廓系数最大）: K={best_k}")

    # 使用最优 K 聚类
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_cluster["Cluster"] = km_final.fit_predict(X_cluster)

    # 各类特征均值
    print(f"\n  各聚类中心特征均值:")
    print("  " + "-" * 70)
    cluster_summary = df_cluster.groupby("Cluster")[cluster_features].mean()
    print(cluster_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # 各类样本量和企业性质分布
    print(f"\n  各聚类样本量及国有占比:")
    for c in range(best_k):
        subset = df_cluster[df_cluster["Cluster"] == c]
        soe_pct = subset["IsSOE"].mean() * 100 if "IsSOE" in subset.columns else 0
        print(f"    聚类{c}: {len(subset)} 家  |  国有占比: {soe_pct:.1f}%  |  "
              f"平均补助: {subset['SubsidyAmount'].mean():.0f}  |  "
              f"平均薪酬: {subset['Top3Salary'].mean():.0f}")

    # 画聚类散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0', '#795548'][:best_k]
    cluster_names = [f'聚类{i}' for i in range(best_k)]

    # 补助-超额薪酬
    for c in range(best_k):
        mask = df_cluster["Cluster"] == c
        axes[0].scatter(df_cluster.loc[mask, "lnSubsidy"],
                        df_cluster.loc[mask, "Overpay"],
                        c=colors[c], label=cluster_names[c], alpha=0.3, s=5)
    axes[0].set_xlabel('lnSubsidy (政府补助对数)')
    axes[0].set_ylabel('Overpay (超额薪酬)')
    axes[0].set_title('(a) 聚类分布：补助 vs 超额薪酬')
    axes[0].legend()

    # 规模-杠杆
    for c in range(best_k):
        mask = df_cluster["Cluster"] == c
        axes[1].scatter(df_cluster.loc[mask, "lnSale"],
                        df_cluster.loc[mask, "Lever"],
                        c=colors[c], label=cluster_names[c], alpha=0.3, s=5)
    axes[1].set_xlabel('lnSale (营业收入对数)')
    axes[1].set_ylabel('Lever (财务杠杆)')
    axes[1].set_title('(b) 聚类分布：规模 vs 杠杆')
    axes[1].legend()

    plt.suptitle('图6  K-Means企业聚类分析', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_kmeans_clusters.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig6_kmeans_clusters.png")

    # 画聚类中心热力图
    fig, ax = plt.subplots(figsize=(10, 5))
    # 标准化后的聚类中心
    centers_std = pd.DataFrame(
        scaler.transform(cluster_summary.values),
        columns=cluster_features,
        index=[f'聚类{i}' for i in range(best_k)]
    )
    sns.heatmap(centers_std, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('图7  聚类中心特征热力图（标准化）', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_cluster_heatmap.png"), bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig7_cluster_heatmap.png")

    return km_final, df_cluster


# ============================================================
# 第六部分：模型对比总结
# ============================================================

def model_comparison(X, y, y_class, feature_names, output_dir):
    """各模型性能对比表"""
    print("\n" + "=" * 70)
    print("第六部分：模型性能对比")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # 1. Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_s, y_train)
    y_pred = lasso.predict(X_test_s)
    results.append({"模型": "Lasso回归", "R²": r2_score(y_test, y_pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                     "MAE": mean_absolute_error(y_test, y_pred)})

    # 2. 随机森林
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append({"模型": "随机森林", "R²": r2_score(y_test, y_pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                     "MAE": mean_absolute_error(y_test, y_pred)})

    # 3. XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=10, random_state=42, n_jobs=-1)
    xgb_m.fit(X_train, y_train, verbose=False)
    y_pred = xgb_m.predict(X_test)
    results.append({"模型": "XGBoost", "R²": r2_score(y_test, y_pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                     "MAE": mean_absolute_error(y_test, y_pred)})

    # 对比表
    results_df = pd.DataFrame(results)
    print("\n  表4  模型性能对比")
    print("  " + "=" * 50)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("  " + "=" * 50)

    # 保存
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # 画对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = results_df["模型"].tolist()
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for i, metric in enumerate(["R²", "RMSE", "MAE"]):
        vals = results_df[metric].tolist()
        bars = axes[i].bar(models, vals, color=colors)
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_ylabel(metric)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('图8  回归模型性能对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_model_comparison.png"), bbox_inches='tight')
    plt.close()
    print(f"\n  已保存: fig8_model_comparison.png")

    return results_df


# ============================================================
# 主流程
# ============================================================

def main():
    data_dir = os.path.join(os.getcwd(), "processed_data")
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)

    # 加载与准备数据（复用 regression_analysis.py 逻辑）
    print("加载数据...")
    df = load_and_clean_data(data_dir)
    df = construct_variables(df)
    df = filter_and_describe(df)
    df = compute_overpay(df)
    df = compute_power(df)

    # 准备 ML 数据
    df_ml, X, y, y_class, feature_names = prepare_ml_data(df)

    print("\n" + "#" * 70)
    print("#  数据挖掘分析开始")
    print("#" * 70)

    # 1. Lasso 变量筛选
    retained_vars = lasso_analysis(X, y, feature_names, output_dir)

    # 2. 随机森林
    rf_reg, rf_clf, rf_importance = random_forest_analysis(X, y, y_class, feature_names, output_dir)

    # 3. XGBoost + SHAP
    xgb_model, shap_importance = xgboost_shap_analysis(X, y, feature_names, output_dir)

    # 4. 决策树
    dt_model = decision_tree_analysis(X, y_class, feature_names, output_dir)

    # 5. K-Means 聚类
    km_model, df_clustered = kmeans_analysis(df_ml, feature_names, output_dir)

    # 6. 模型对比
    comparison = model_comparison(X, y, y_class, feature_names, output_dir)

    # 最终总结
    print("\n" + "=" * 70)
    print("数据挖掘分析完成！")
    print("=" * 70)
    print(f"\n生成的图表文件 (均在 results/ 目录下):")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("fig"):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024
            print(f"  📊 {f}  ({size:.0f} KB)")

    print(f"\n数据文件:")
    print(f"  📄 model_comparison.csv")
    print(f"  📄 regression_dataset.csv")
    print(f"  📄 regression_tables.txt")


if __name__ == "__main__":
    main()
