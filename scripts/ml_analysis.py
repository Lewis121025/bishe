"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 机器学习分析脚本
=============================================================================

分析内容：
  1. 回归预测：OLS / Lasso / RandomForest / XGBoost
  2. 分类识别：Logit / RandomForestClassifier / XGBoostClassifier / DecisionTree
  3. 可解释性与补充：SHAP、决策树可视化、K-Means 聚类
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

plt.rcParams["font.sans-serif"] = ["Heiti SC", "STHeiti", "Arial Unicode MS", "SimHei", "Heiti TC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
_TITLE_FONT = next((font_manager.FontProperties(fname=path, size=16) for path in _CJK_FONT_CANDIDATES if os.path.exists(path)), None)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from regression_analysis import build_analysis_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _format_params(params):
    if not params:
        return "无"
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _regression_metrics(y_true, y_pred):
    return {
        "R²": float(r2_score(y_true, y_pred)),
        "RMSE": _rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, y_score)),
    }


def _classification_cv_roc_auc(model, X_train, y_train, groups_train, cv):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    return float(scores.mean()), float(scores.std())


def _grouped_lasso_alpha_search(X_train, y_train, groups_train, alpha_grid, cv):
    """在公司分组口径下搜索使平均 CV R² 最高的 alpha。"""
    rows = []
    best_alpha = None
    best_score = -np.inf
    best_std = None

    for alpha in alpha_grid:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000)),
        ])
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="r2",
            n_jobs=1,
        )
        score_mean = float(scores.mean())
        score_std = float(scores.std())
        rows.append({
            "alpha": float(alpha),
            "cv_r2_mean": score_mean,
            "cv_r2_std": score_std,
        })

        if score_mean > best_score + 1e-12 or (
            np.isclose(score_mean, best_score, atol=1e-12) and (best_alpha is None or alpha < best_alpha)
        ):
            best_alpha = float(alpha)
            best_score = score_mean
            best_std = score_std

    alpha_df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    return best_alpha, best_score, best_std, alpha_df


def prepare_ml_data(df):
    """准备机器学习所需的数据。"""
    print("\n" + "=" * 70)
    print("数据准备：构建特征矩阵")
    print("=" * 70)

    feature_cols = [
        "lnSubsidy",
        "Roa",
        "Lever",
        "Top1",
        "Zone",
        "Industry",
        "lnSale",
        "IA",
        "Boardsize",
        "Dual",
        "Insider",
        "Mgshder",
        "Tenure",
        "IsSOE",
    ]
    target_col = "Overpay"
    use_cols = feature_cols + [target_col, "Symbol", "Year", "SubsidyAmount", "Top3Salary"]

    df_ml = df.dropna(subset=use_cols).copy()
    y_class = (df_ml[target_col] > 0).astype(int)
    positive_count = int(y_class.sum())
    total_count = int(len(df_ml))
    negative_count = total_count - positive_count
    positive_ratio = positive_count / total_count if total_count else np.nan

    print(f"  可用样本量: {total_count}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  超额薪酬为正样本: {positive_count}")
    print(f"  超额薪酬为负样本: {negative_count}")
    print(f"  正类比例: {positive_ratio:.4%}")
    print("  判断：类别分布基本均衡，不采用 SMOTE；holdout 采用 GroupShuffleSplit 按公司分组随机划分，并在分类模型中比较 class_weight/scale_pos_weight。")

    return {
        "df_ml": df_ml,
        "X": df_ml[feature_cols].copy(),
        "y": df_ml[target_col].copy(),
        "y_class": y_class.copy(),
        "feature_names": feature_cols,
        "class_balance": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": float(positive_ratio),
        },
    }


def create_holdout_splits(X, y, y_class, df_ml):
    """以公司为分组单位的随机划分（GroupShuffleSplit），确保同一公司全部年度观测仅进入训练集或测试集之一，避免公司内序列相关导致性能高估。"""
    groups = df_ml['Symbol']
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    n_train_companies = groups.iloc[train_idx].nunique()
    n_test_companies = groups.iloc[test_idx].nunique()
    print(f"  GroupShuffleSplit: 训练集 {len(train_idx)} 条（{n_train_companies} 家公司），"
          f"测试集 {len(test_idx)} 条（{n_test_companies} 家公司）")

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "yc_train": y_class.iloc[train_idx].copy(),
        "yc_test": y_class.iloc[test_idx].copy(),
        "groups_train": groups.iloc[train_idx].copy(),
        "groups_test": groups.iloc[test_idx].copy(),
    }


def lasso_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    """Lasso 回归：按公司分组搜索 alpha，并用同一分组口径评估泛化表现。"""
    print("\n" + "=" * 70)
    print("第一部分：Lasso 回归 — 变量筛选与正则化")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=CV_FOLDS)
    alpha_grid = np.logspace(-4, 1, 60)

    best_alpha, best_cv_mean, best_cv_std, alpha_df = _grouped_lasso_alpha_search(
        X_train, y_train, groups_train, alpha_grid, cv
    )
    alpha_df.to_csv(os.path.join(output_dir, "lasso_alpha_search.csv"), index=False)

    lasso_fixed = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, max_iter=20000)),
    ])
    lasso_fixed.fit(X_train, y_train)
    y_pred = lasso_fixed.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)
    cv_scores = cross_val_score(lasso_fixed, X_train, y_train, groups=groups_train, cv=cv, scoring="r2", n_jobs=1)

    scaler = lasso_fixed.named_steps["scaler"]
    lasso_model = lasso_fixed.named_steps["model"]
    X_train_scaled = scaler.transform(X_train)

    coefs = pd.DataFrame({
        "变量": feature_names,
        "Lasso系数": lasso_model.coef_,
    }).sort_values("Lasso系数", key=abs, ascending=False)
    coefs.to_csv(os.path.join(output_dir, "lasso_coefficients.csv"), index=False)

    retained = coefs.loc[coefs["Lasso系数"].abs() > 1e-6, "变量"].tolist()
    removed = coefs.loc[coefs["Lasso系数"].abs() <= 1e-6, "变量"].tolist()

    print(f"  最优 alpha = {best_alpha:.6f}")
    print(f"  测试集 R² = {metrics['R²']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  保留变量数 = {len(retained)}")

    coef_paths = []
    for alpha in alpha_grid:
        model = Lasso(alpha=alpha, max_iter=20000)
        model.fit(X_train_scaled, y_train)
        coef_paths.append(model.coef_)
    coef_paths = np.array(coef_paths)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(feature_names):
        ax.plot(alpha_grid, coef_paths[:, idx], label=name)
    ax.axvline(best_alpha, color="black", linestyle="--", label=f"最优α={best_alpha:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("正则化参数 α (log)")
    ax.set_ylabel("Lasso 回归系数")
    ax.set_title("图5-5 Lasso系数收缩图", fontproperties=_TITLE_FONT)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_lasso_path.png"), bbox_inches="tight")
    plt.close()

    return {
        "model": lasso_fixed,
        "alpha": float(best_alpha),
        "retained": retained,
        "removed": removed,
        "coefficients": coefs,
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "param_summary": {
            "alpha": float(best_alpha),
            "alpha_search_cv_mean": float(best_cv_mean),
            "alpha_search_cv_std": float(best_cv_std),
        },
        "tuning_method": "自定义alpha网格搜索(内外层均为GroupKFold)",
    }


def ols_baseline_analysis(X_train, X_test, y_train, y_test, groups_train):
    """OLS 基准模型。"""
    print("\n" + "=" * 70)
    print("OLS 基准模型")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=CV_FOLDS)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=cv, scoring="r2", n_jobs=1)

    print(f"  测试集 R² = {metrics['R²']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model": model,
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "param_summary": {},
        "tuning_method": "无调参（线性基准）",
    }


def random_forest_analysis(X_train, X_test, y_train, y_test, yc_train, yc_test, groups_train, feature_names, output_dir):
    """随机森林回归与分类。"""
    print("\n" + "=" * 70)
    print("第二部分：随机森林 — 回归预测与分类识别")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    cv_reg = GroupKFold(n_splits=CV_FOLDS)
    cv_clf = GroupKFold(n_splits=CV_FOLDS)

    rf_reg_space = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [10, 20, 50],
        "max_features": ["sqrt", 0.6, 1.0],
    }
    rf_reg_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_reg_space,
        n_iter=8,
        scoring="r2",
        cv=cv_reg,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    rf_reg_search.fit(X_train, y_train, groups=groups_train)
    rf_reg = rf_reg_search.best_estimator_
    rf_reg_pred = rf_reg.predict(X_test)
    rf_reg_metrics = _regression_metrics(y_test, rf_reg_pred)
    rf_reg_cv = cross_val_score(rf_reg, X_train, y_train, groups=groups_train, cv=cv_reg, scoring="r2", n_jobs=1)

    rf_clf_space = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [10, 20, 50],
        "max_features": ["sqrt", 0.6, 1.0],
        "class_weight": [None, "balanced"],
    }
    rf_clf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_clf_space,
        n_iter=8,
        scoring="f1",
        cv=cv_clf,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    rf_clf_search.fit(X_train, yc_train, groups=groups_train)
    rf_clf = rf_clf_search.best_estimator_
    rf_clf_metrics = _classification_metrics(rf_clf, X_test, yc_test)
    rf_clf_cv_mean, rf_clf_cv_std = _classification_cv_roc_auc(rf_clf, X_train, yc_train, groups_train, cv_clf)

    importance_reg = pd.DataFrame({
        "变量": feature_names,
        "回归重要性": rf_reg.feature_importances_,
    }).sort_values("回归重要性", ascending=False)
    importance_clf = pd.DataFrame({
        "变量": feature_names,
        "分类重要性": rf_clf.feature_importances_,
    }).sort_values("分类重要性", ascending=False)
    importance_reg.to_csv(os.path.join(output_dir, "rf_reg_importance.csv"), index=False)
    importance_clf.to_csv(os.path.join(output_dir, "rf_clf_importance.csv"), index=False)

    print(f"  随机森林回归测试集 R² = {rf_reg_metrics['R²']:.4f}")
    print(f"  随机森林回归 5-fold CV R² = {rf_reg_cv.mean():.4f} ± {rf_reg_cv.std():.4f}")
    print(f"  随机森林分类 Accuracy = {rf_clf_metrics['Accuracy']:.4f}")
    print(f"  随机森林分类 F1 = {rf_clf_metrics['F1']:.4f}")
    print(f"  随机森林分类 5-fold CV ROC_AUC = {rf_clf_cv_mean:.4f} ± {rf_clf_cv_std:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    reg_sorted = importance_reg.sort_values("回归重要性", ascending=True)
    clf_sorted = importance_clf.sort_values("分类重要性", ascending=True)
    axes[0].barh(reg_sorted["变量"], reg_sorted["回归重要性"], color="steelblue")
    axes[0].set_title(f"(a) 随机森林回归特征重要性\nR²={rf_reg_metrics['R²']:.4f}")
    axes[0].set_xlabel("重要性")
    axes[1].barh(clf_sorted["变量"], clf_sorted["分类重要性"], color="coral")
    axes[1].set_title(f"(b) 随机森林分类特征重要性\nF1={rf_clf_metrics['F1']:.4f}")
    axes[1].set_xlabel("重要性")
    plt.suptitle("图5-2  随机森林特征重要性剖析", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_rf_importance.png"), bbox_inches="tight")
    plt.close()

    return {
        "rf_reg": rf_reg,
        "rf_reg_metrics": rf_reg_metrics,
        "rf_reg_cv_mean": float(rf_reg_cv.mean()),
        "rf_reg_cv_std": float(rf_reg_cv.std()),
        "rf_reg_params": rf_reg_search.best_params_,
        "rf_reg_tuning": "RandomizedSearchCV(n_iter=8, cv=5)",
        "rf_clf": rf_clf,
        "rf_clf_metrics": rf_clf_metrics,
        "rf_clf_cv_mean": rf_clf_cv_mean,
        "rf_clf_cv_std": rf_clf_cv_std,
        "rf_clf_params": rf_clf_search.best_params_,
        "rf_clf_tuning": "RandomizedSearchCV(n_iter=8, cv=5, scoring=F1)",
        "importance_reg": importance_reg,
        "importance_clf": importance_clf,
    }


def xgboost_regression_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    """XGBoost 回归与 SHAP 分析。"""
    print("\n" + "=" * 70)
    print("第三部分：XGBoost 回归 + SHAP 可解释性")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=CV_FOLDS)
    param_space = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [5, 10, 20],
    }
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_space,
        n_iter=10,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    xgb_model = search.best_estimator_
    y_pred = xgb_model.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, groups=groups_train, cv=cv, scoring="r2", n_jobs=1)

    print(f"  XGBoost 测试集 R² = {metrics['R²']:.4f}")
    print(f"  XGBoost 5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    sample_size = min(2000, len(X_test))
    X_sample = X_test.iloc[:sample_size].copy()
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "变量": feature_names,
        "SHAP均值": shap_importance,
    }).sort_values("SHAP均值", ascending=False)
    shap_df.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=14)
    plt.title("图5-1  SHAP 特征重要性汇总图", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_shap_summary.png"), bbox_inches="tight")
    plt.close()

    subsidy_idx = feature_names.index("lnSubsidy")
    fig, ax = plt.subplots(figsize=(8, 5))
    dependence_kwargs = {
        "feature_names": feature_names,
        "ax": ax,
        "show": False,
        "interaction_index": None,
    }
    shap.dependence_plot(subsidy_idx, shap_values, X_sample, **dependence_kwargs)
    ax.set_title("图5-3  财政补贴的SHAP依赖图", fontsize=13)
    ax.set_xlabel("lnSubsidy (政府补助对数)")
    ax.set_ylabel("SHAP值 (对超额薪酬的边际影响)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_shap_subsidy.png"), bbox_inches="tight")
    plt.close()

    # SHAP交互图：lnSubsidy × IsSOE（补贴×产权性质）
    if "IsSOE" in feature_names:
        issoe_idx = feature_names.index("IsSOE")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(subsidy_idx, shap_values, X_sample, interaction_index=issoe_idx,
                             feature_names=feature_names, ax=ax, show=False)
        ax.set_title("图5-4a  SHAP交互图：lnSubsidy × IsSOE", fontsize=13)
        ax.set_xlabel("lnSubsidy (政府补助对数)")
        ax.set_ylabel("SHAP值 (对超额薪酬的边际影响)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4a_shap_subsidy_issoe.png"), bbox_inches="tight")
        plt.close()

    # SHAP交互图：lnSubsidy × Mgshder（补贴×管理层持股）
    if "Mgshder" in feature_names:
        mgshder_idx = feature_names.index("Mgshder")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(subsidy_idx, shap_values, X_sample, interaction_index=mgshder_idx,
                             feature_names=feature_names, ax=ax, show=False)
        ax.set_title("图5-4b  SHAP交互图：lnSubsidy × Mgshder", fontsize=13)
        ax.set_xlabel("lnSubsidy (政府补助对数)")
        ax.set_ylabel("SHAP值 (对超额薪酬的边际影响)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4b_shap_subsidy_mgshder.png"), bbox_inches="tight")
        plt.close()

    return {
        "model": xgb_model,
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "params": search.best_params_,
        "tuning_method": "RandomizedSearchCV(n_iter=10, cv=5)",
        "shap_df": shap_df,
    }


def classification_comparison(X_train, X_test, yc_train, yc_test, groups_train):
    """分类模型统一对比：Logit / RF / XGB / DT。"""
    print("\n" + "=" * 70)
    print("第四部分：分类模型对比")
    print("=" * 70)

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=CV_FOLDS)
    pos_weight = (len(yc_train) - int(yc_train.sum())) / max(int(yc_train.sum()), 1)

    logit_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    logit_grid = {
        "model__C": [0.1, 1.0, 3.0],
        "model__class_weight": [None, "balanced"],
    }
    logit_search = GridSearchCV(logit_pipe, logit_grid, scoring="f1", cv=cv, n_jobs=1)
    logit_search.fit(X_train, yc_train, groups=groups_train)
    logit_model = logit_search.best_estimator_

    xgb_clf_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_clf_space = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "min_child_weight": [1, 5, 10],
        "scale_pos_weight": [1.0, round(pos_weight, 4)],
    }
    xgb_clf_search = RandomizedSearchCV(
        xgb_clf_base,
        param_distributions=xgb_clf_space,
        n_iter=10,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_clf_search.fit(X_train, yc_train, groups=groups_train)
    xgb_clf = xgb_clf_search.best_estimator_

    dt_grid = {
        "max_depth": [3, 4, 5],
        "min_samples_leaf": [50, 100, 200],
        "class_weight": [None, "balanced"],
    }
    dt_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )
    dt_search.fit(X_train, yc_train, groups=groups_train)
    dt_model = dt_search.best_estimator_

    models = [
        {
            "模型": "Logit",
            "estimator": logit_model,
            "tuning_method": "GridSearchCV(cv=5, scoring=F1)",
            "params": logit_search.best_params_,
            "balance": f"class_weight={logit_search.best_params_.get('model__class_weight')}",
        },
        {
            "模型": "XGBoostClassifier",
            "estimator": xgb_clf,
            "tuning_method": "RandomizedSearchCV(n_iter=10, cv=5, scoring=F1)",
            "params": xgb_clf_search.best_params_,
            "balance": f"scale_pos_weight={xgb_clf_search.best_params_.get('scale_pos_weight')}",
        },
        {
            "模型": "DecisionTree",
            "estimator": dt_model,
            "tuning_method": "GridSearchCV(cv=5, scoring=F1)",
            "params": dt_search.best_params_,
            "balance": f"class_weight={dt_search.best_params_.get('class_weight')}",
        },
    ]

    rows = []
    for spec in models:
        metrics = _classification_metrics(spec["estimator"], X_test, yc_test)
        cv_mean, cv_std = _classification_cv_roc_auc(spec["estimator"], X_train, yc_train, groups_train, cv)
        rows.append({
            "模型": spec["模型"],
            **metrics,
            "5折CV ROC_AUC": cv_mean,
            "调参方式": spec["tuning_method"],
            "最优参数摘要": _format_params(spec["params"]),
            "平衡处理": spec["balance"],
        })
        print(
            f"  {spec['模型']} -> Accuracy={metrics['Accuracy']:.4f}, "
            f"F1={metrics['F1']:.4f}, ROC_AUC={metrics['ROC_AUC']:.4f}, "
            f"5-fold CV ROC_AUC={cv_mean:.4f} ± {cv_std:.4f}"
        )

    return {
        "comparison": pd.DataFrame(rows),
        "logit_model": logit_model,
        "logit_params": logit_search.best_params_,
        "xgb_clf_model": xgb_clf,
        "xgb_clf_params": xgb_clf_search.best_params_,
        "decision_tree_model": dt_model,
        "decision_tree_params": dt_search.best_params_,
        "search_spaces": {
            "logit": logit_grid,
            "xgb_classifier": xgb_clf_space,
            "decision_tree": dt_grid,
        },
    }


def decision_tree_analysis(model, feature_names, output_dir):
    """决策树可视化。"""
    print("\n" + "=" * 70)
    print("第五部分：决策树可视化")
    print("=" * 70)

    tree_rules = export_text(model, feature_names=feature_names, max_depth=model.get_params().get("max_depth", 4))
    print("  决策树规则（前 20 行）:")
    for line in tree_rules.split("\n")[:20]:
        print(f"    {line}")

    cn = ["无超额薪酬", "有超额薪酬"]
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=cn,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        proportion=True,
        impurity=False,
    )
    ax.set_title("图5  决策树：高管超额薪酬风险分类路径", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_decision_tree.png"), bbox_inches="tight")
    plt.close()


def kmeans_analysis(df_ml, output_dir):
    """K-Means 聚类分析。"""
    print("\n" + "=" * 70)
    print("第六部分：K-Means 聚类 — 企业画像分析")
    print("=" * 70)

    cluster_features = ["lnSubsidy", "Overpay", "Roa", "Lever", "Top1", "lnSale"]
    df_cluster = df_ml.dropna(subset=cluster_features).copy()
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster[cluster_features])

    inertias = []
    sil_scores = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_cluster, km.labels_, sample_size=min(5000, len(df_cluster)))
        sil_scores.append(sil)
        print(f"  K={k}: 轮廓系数={sil:.4f}, 惯性={km.inertia_:.0f}")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"  最优 K = {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df_cluster["Cluster"] = km_final.fit_predict(X_cluster)
    cluster_summary = df_cluster.groupby("Cluster")[cluster_features].mean()
    cluster_summary.to_csv(os.path.join(output_dir, "kmeans_cluster_summary.csv"))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#795548"][:best_k]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c in range(best_k):
        mask = df_cluster["Cluster"] == c
        axes[0].scatter(df_cluster.loc[mask, "lnSubsidy"], df_cluster.loc[mask, "Overpay"], c=colors[c], alpha=0.3, s=5, label=f"聚类{c}")
        axes[1].scatter(df_cluster.loc[mask, "lnSale"], df_cluster.loc[mask, "Lever"], c=colors[c], alpha=0.3, s=5, label=f"聚类{c}")
    axes[0].set_xlabel("lnSubsidy")
    axes[0].set_ylabel("Overpay")
    axes[0].set_title("(a) 聚类分布：补助 vs 超额薪酬")
    axes[1].set_xlabel("lnSale")
    axes[1].set_ylabel("Lever")
    axes[1].set_title("(b) 聚类分布：规模 vs 杠杆")
    axes[0].legend()
    axes[1].legend()
    plt.suptitle("图6  K-Means 企业聚类分析", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_kmeans_clusters.png"), bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    centers_std = pd.DataFrame(
        scaler.transform(cluster_summary.values),
        columns=cluster_features,
        index=[f"聚类{i}" for i in range(best_k)],
    )
    sns.heatmap(centers_std, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax, linewidths=0.5)
    ax.set_title("图7  聚类中心特征热力图（标准化）", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_cluster_heatmap.png"), bbox_inches="tight")
    plt.close()

    return {
        "best_k": int(best_k),
        "cluster_summary": cluster_summary,
        "silhouette_scores": {int(k): float(v) for k, v in zip(k_range, sil_scores)},
    }


def build_regression_comparison(ols_result, lasso_result, rf_result, xgb_result, output_dir):
    """回归模型性能总表。"""
    print("\n" + "=" * 70)
    print("第七部分：回归模型性能对比")
    print("=" * 70)

    rows = [
        {
            "模型": "OLS",
            "任务": "回归预测",
            "测试集R²": ols_result["test_metrics"]["R²"],
            "RMSE": ols_result["test_metrics"]["RMSE"],
            "MAE": ols_result["test_metrics"]["MAE"],
            "5折CV R²均值": ols_result["cv_mean"],
            "5折CV R²标准差": ols_result["cv_std"],
            "调参方式": ols_result["tuning_method"],
            "最优参数摘要": _format_params(ols_result["param_summary"]),
        },
        {
            "模型": "Lasso",
            "任务": "回归预测",
            "测试集R²": lasso_result["test_metrics"]["R²"],
            "RMSE": lasso_result["test_metrics"]["RMSE"],
            "MAE": lasso_result["test_metrics"]["MAE"],
            "5折CV R²均值": lasso_result["cv_mean"],
            "5折CV R²标准差": lasso_result["cv_std"],
            "调参方式": lasso_result["tuning_method"],
            "最优参数摘要": _format_params(lasso_result["param_summary"]),
        },
        {
            "模型": "RandomForest",
            "任务": "回归预测",
            "测试集R²": rf_result["rf_reg_metrics"]["R²"],
            "RMSE": rf_result["rf_reg_metrics"]["RMSE"],
            "MAE": rf_result["rf_reg_metrics"]["MAE"],
            "5折CV R²均值": rf_result["rf_reg_cv_mean"],
            "5折CV R²标准差": rf_result["rf_reg_cv_std"],
            "调参方式": rf_result["rf_reg_tuning"],
            "最优参数摘要": _format_params(rf_result["rf_reg_params"]),
        },
        {
            "模型": "XGBoost",
            "任务": "回归预测",
            "测试集R²": xgb_result["test_metrics"]["R²"],
            "RMSE": xgb_result["test_metrics"]["RMSE"],
            "MAE": xgb_result["test_metrics"]["MAE"],
            "5折CV R²均值": xgb_result["cv_mean"],
            "5折CV R²标准差": xgb_result["cv_std"],
            "调参方式": xgb_result["tuning_method"],
            "最优参数摘要": _format_params(xgb_result["params"]),
        },
    ]
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#7E57C2", "#2196F3", "#4CAF50", "#FF5722"]
    for idx, metric in enumerate(["测试集R²", "RMSE", "MAE"]):
        vals = comparison_df[metric].tolist()
        bars = axes[idx].bar(comparison_df["模型"], vals, color=colors)
        axes[idx].set_title(metric)
        for bar, val in zip(bars, vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("图5-6  回归模型拟合表现对比（含 OLS 基准）", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_model_comparison.png"), bbox_inches="tight")
    plt.close()

    return comparison_df


def save_ml_outputs(output_dir, prepared, splits, ols_result, lasso_result, rf_result, xgb_result, classification_result, kmeans_result, comparison_df):
    """保存 JSON/CSV 摘要。"""
    class_balance = prepared["class_balance"]
    classification_df = classification_result["comparison"].copy()
    rf_row = pd.DataFrame([{
        "模型": "RandomForestClassifier",
        **rf_result["rf_clf_metrics"],
        "5折CV ROC_AUC": rf_result["rf_clf_cv_mean"],
        "调参方式": rf_result["rf_clf_tuning"],
        "最优参数摘要": _format_params(rf_result["rf_clf_params"]),
        "平衡处理": f"class_weight={rf_result['rf_clf_params'].get('class_weight')}",
    }])
    classification_df = pd.concat([classification_df.iloc[:1], rf_row, classification_df.iloc[1:]], ignore_index=True)
    classification_df.to_csv(os.path.join(output_dir, "classification_comparison.csv"), index=False)

    tuning_summary = {
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "prepared_sample_size": int(len(prepared["df_ml"])),
            "prepared_company_count": int(prepared["df_ml"]["Symbol"].nunique()),
            "train_size": int(len(splits["train_idx"])),
            "test_size_n": int(len(splits["test_idx"])),
            "holdout_method": "GroupShuffleSplit (by company Symbol)",
            "group_isolation": True,
            "classification_stratified": False,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "lasso_internal": "GroupKFold(n_splits=5)",
            "regression": "GroupKFold(n_splits=5)",
            "classification": "GroupKFold(n_splits=5)",
        },
        "class_balance": {
            **class_balance,
            "imbalance_judgment": "不严重失衡",
            "resampling": "未使用SMOTE",
            "handling": (
                "holdout采用GroupShuffleSplit按公司分组随机划分（8:2），"
                "确保同一公司全部年度仅进入训练或测试集之一；"
                "分类模型调参比较 class_weight / scale_pos_weight"
            ),
        },
        "search_spaces": {
            "lasso_alpha_grid": list(np.logspace(-4, 1, 60)),
            "rf_regression": {
                "n_estimators": [200, 300, 400],
                "max_depth": [6, 10, None],
                "min_samples_leaf": [10, 20, 50],
                "max_features": ["sqrt", 0.6, 1.0],
            },
            "rf_classification": {
                "n_estimators": [200, 300, 400],
                "max_depth": [6, 10, None],
                "min_samples_leaf": [10, 20, 50],
                "max_features": ["sqrt", 0.6, 1.0],
                "class_weight": [None, "balanced"],
            },
            "xgb_regression": {
                "n_estimators": [200, 300, 400],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "min_child_weight": [5, 10, 20],
            },
            **classification_result["search_spaces"],
        },
        "best_params": {
            "ols": {},
            "lasso": lasso_result["param_summary"],
            "rf_regression": rf_result["rf_reg_params"],
            "rf_classification": rf_result["rf_clf_params"],
            "xgb_regression": xgb_result["params"],
            "logit": classification_result["logit_params"],
            "xgb_classification": classification_result["xgb_clf_params"],
            "decision_tree": classification_result["decision_tree_params"],
        },
    }
    with open(os.path.join(output_dir, "ml_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(tuning_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "class_balance": class_balance,
        "ml_sample_size": int(len(prepared["df_ml"])),
        "ml_company_count": int(prepared["df_ml"]["Symbol"].nunique()),
        "train_size": int(len(splits["train_idx"])),
        "test_size": int(len(splits["test_idx"])),
        "train_company_count": int(splits["groups_train"].nunique()),
        "test_company_count": int(splits["groups_test"].nunique()),
        "ols_r2": ols_result["test_metrics"]["R²"],
        "lasso_alpha": lasso_result["alpha"],
        "lasso_retained": lasso_result["retained"],
        "rf_regression_r2": rf_result["rf_reg_metrics"]["R²"],
        "rf_classification_f1": rf_result["rf_clf_metrics"]["F1"],
        "xgb_regression_r2": xgb_result["test_metrics"]["R²"],
        "xgb_top_shap_feature": xgb_result["shap_df"].iloc[0]["变量"],
        "best_kmeans_k": kmeans_result["best_k"],
        "regression_comparison": comparison_df.to_dict(orient="records"),
        "classification_comparison": classification_df.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "ml_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    data_dir = os.path.join(ROOT_DIR, "processed_data")
    output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据...")
    df, _ = build_analysis_dataset(data_dir)

    prepared = prepare_ml_data(df)
    splits = create_holdout_splits(prepared["X"], prepared["y"], prepared["y_class"], prepared["df_ml"])

    print("\n" + "#" * 70)
    print("#  机器学习补充分析开始")
    print("#" * 70)

    ols_result = ols_baseline_analysis(splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"], splits["groups_train"])
    lasso_result = lasso_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )
    rf_result = random_forest_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["yc_train"],
        splits["yc_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )
    xgb_result = xgboost_regression_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )
    classification_result = classification_comparison(
        splits["X_train"],
        splits["X_test"],
        splits["yc_train"],
        splits["yc_test"],
        splits["groups_train"],
    )
    decision_tree_analysis(classification_result["decision_tree_model"], prepared["feature_names"], output_dir)
    kmeans_result = kmeans_analysis(prepared["df_ml"], output_dir)
    comparison_df = build_regression_comparison(ols_result, lasso_result, rf_result, xgb_result, output_dir)
    save_ml_outputs(
        output_dir,
        prepared,
        splits,
        ols_result,
        lasso_result,
        rf_result,
        xgb_result,
        classification_result,
        kmeans_result,
        comparison_df,
    )

    print("\n" + "=" * 70)
    print("机器学习补充分析完成")
    print("=" * 70)
    print("已生成：model_comparison.csv、classification_comparison.csv、lasso_alpha_search.csv、ml_tuning_summary.json、ml_summary.json 及 fig1-fig8 图表。")


if __name__ == "__main__":
    main()
