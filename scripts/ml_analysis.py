"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 机器学习补充验证脚本
=============================================================================

本脚本围绕当前 OLS 主回归开展三类机器学习补充验证：
  1. 随机森林：检验财政补贴强度及控制变量的特征重要性
  2. SHAP：补充解释随机森林中的边际贡献大小
  3. Lasso：检验财政补贴变量是否被保留，以及方向是否与 OLS 一致
  4. XGBoost：检验财政补贴在更灵活树模型中的重要性，并绘制部分依赖图
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
BUNDLE_IMAGES_DIR = os.path.join(ROOT_DIR, "thesis_final_bundle", "images")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Heiti SC", "STHeiti", "Arial Unicode MS", "SimHei", "Heiti TC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
_TITLE_FONT = next(
    (font_manager.FontProperties(fname=path, size=16) for path in _CJK_FONT_CANDIDATES if os.path.exists(path)),
    None,
)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from regression_analysis import ALT_SUBSIDY_LAG_COL, build_analysis_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SUBSIDY_FEATURE = ALT_SUBSIDY_LAG_COL


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": _rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _format_params(params):
    if not params:
        return "无"
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _grouped_lasso_alpha_search(X_train, y_train, groups_train, alpha_grid, cv):
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
    """准备机器学习补充分析数据。"""
    print("\n" + "=" * 70)
    print("数据准备：围绕 OLS 主回归核心变量构建机器学习样本")
    print("=" * 70)

    feature_cols = [
        ALT_SUBSIDY_LAG_COL,
        "Roa",
        "Lever",
        "Top1",
    ]
    target_col = "Overpay"
    use_cols = feature_cols + [target_col, "Symbol", "Year"]

    # 与新的主回归保持一致：只比较上一年已获得财政补贴企业内部的补贴强度差异。
    df_ml = df.dropna(subset=use_cols).copy()
    print(f"  可用样本量: {len(df_ml)}")
    print(f"  公司数量: {df_ml['Symbol'].nunique()}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  核心解释变量: {SUBSIDY_FEATURE}")
    print(f"  控制变量: {[x for x in feature_cols if x != SUBSIDY_FEATURE]}")

    return {
        "df_ml": df_ml,
        "X": df_ml[feature_cols].copy(),
        "y": df_ml[target_col].copy(),
        "feature_names": feature_cols,
    }


def create_holdout_splits(X, y, df_ml):
    groups = df_ml["Symbol"]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    print(
        "  GroupShuffleSplit: "
        f"训练集 {len(train_idx)} 条（{groups.iloc[train_idx].nunique()} 家公司），"
        f"测试集 {len(test_idx)} 条（{groups.iloc[test_idx].nunique()} 家公司）"
    )

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X.iloc[train_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_test": y.iloc[test_idx].copy(),
        "groups_train": groups.iloc[train_idx].copy(),
        "groups_test": groups.iloc[test_idx].copy(),
    }


def lasso_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第三部分：Lasso 回归 — 变量保留与方向一致性检验")
    print("=" * 70)

    cv = GroupKFold(n_splits=CV_FOLDS)
    alpha_grid = np.logspace(-4, 1, 60)

    best_alpha, best_cv_mean, best_cv_std, alpha_df = _grouped_lasso_alpha_search(
        X_train, y_train, groups_train, alpha_grid, cv
    )
    alpha_df.to_csv(os.path.join(output_dir, "lasso_alpha_search.csv"), index=False)

    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, max_iter=20000)),
    ])
    lasso_pipe.fit(X_train, y_train)
    y_pred = lasso_pipe.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)
    cv_scores = cross_val_score(
        lasso_pipe,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    scaler = lasso_pipe.named_steps["scaler"]
    lasso_model = lasso_pipe.named_steps["model"]
    X_train_scaled = scaler.transform(X_train)

    coef_df = pd.DataFrame({
        "变量": feature_names,
        "Lasso系数": lasso_model.coef_,
        "绝对值": np.abs(lasso_model.coef_),
    }).sort_values("绝对值", ascending=False).reset_index(drop=True)
    coef_df.to_csv(os.path.join(output_dir, "lasso_coefficients.csv"), index=False)

    retained = coef_df.loc[coef_df["绝对值"] > 1e-6, "变量"].tolist()
    removed = coef_df.loc[coef_df["绝对值"] <= 1e-6, "变量"].tolist()
    coef_map = {
        row["变量"]: float(row["Lasso系数"])
        for _, row in coef_df.iterrows()
    }
    subsidy_coef = coef_map[SUBSIDY_FEATURE]

    print(f"  最优 alpha = {best_alpha:.6f}")
    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 系数 = {subsidy_coef:.6f}")
    print(f"  {SUBSIDY_FEATURE} {'被保留' if SUBSIDY_FEATURE in retained else '被压缩为0'}")
    for feature in feature_names:
        if feature == SUBSIDY_FEATURE:
            continue
        print(f"  {feature} 系数 = {coef_map[feature]:.6f}")

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
    ax.set_title("图4-2 Lasso系数收缩图", fontproperties=_TITLE_FONT)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_lasso_path.png"), bbox_inches="tight")
    ax.set_title("")
    plt.savefig(os.path.join(BUNDLE_IMAGES_DIR, "fig1_lasso_path_notitle.png"), bbox_inches="tight")
    plt.close()

    return {
        "alpha": float(best_alpha),
        "alpha_search_cv_mean": float(best_cv_mean),
        "alpha_search_cv_std": float(best_cv_std),
        "retained": retained,
        "removed": removed,
        "subsidy_coef": subsidy_coef,
        "subsidy_retained": SUBSIDY_FEATURE in retained,
        "subsidy_sign": "正" if subsidy_coef > 0 else ("负" if subsidy_coef < 0 else "零"),
        "coef_map": coef_map,
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": {"alpha": float(best_alpha)},
    }


def random_forest_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第一部分：随机森林 — 特征重要性检验")
    print("=" * 70)

    cv = GroupKFold(n_splits=CV_FOLDS)
    param_space = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [10, 20, 50],
        "max_features": ["sqrt", 0.6, 1.0],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_space,
        n_iter=8,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    rf_model = search.best_estimator_

    y_pred = rf_model.predict(X_test)
    metrics = _regression_metrics(y_test, y_pred)
    cv_scores = cross_val_score(
        rf_model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "变量": feature_names,
        "随机森林重要性": rf_model.feature_importances_,
    }).sort_values("随机森林重要性", ascending=False).reset_index(drop=True)
    importance_df["排名"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_csv(os.path.join(output_dir, "rf_reg_importance.csv"), index=False)

    subsidy_row = importance_df.loc[importance_df["变量"] == SUBSIDY_FEATURE].iloc[0]
    subsidy_rank = int(subsidy_row["排名"])
    subsidy_importance = float(subsidy_row["随机森林重要性"])
    top_features = importance_df.head(5)["变量"].tolist()

    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 排名 = {subsidy_rank}")
    print(f"  {SUBSIDY_FEATURE} 重要性 = {subsidy_importance:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = importance_df.sort_values("随机森林重要性", ascending=True)
    ax.barh(plot_df["变量"], plot_df["随机森林重要性"], color="steelblue")
    ax.set_xlabel("重要性")
    ax.set_title("图4-1 随机森林特征重要性图", fontproperties=_TITLE_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_rf_importance.png"), bbox_inches="tight")
    ax.set_title("")
    plt.savefig(os.path.join(BUNDLE_IMAGES_DIR, "fig2_rf_importance_notitle.png"), bbox_inches="tight")
    plt.close()

    return {
        "model": rf_model,
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": search.best_params_,
        "top_features": top_features,
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": subsidy_importance,
        "importance_table": importance_df.to_dict(orient="records"),
    }


def rf_shap_analysis(rf_model, X_reference, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第二部分：SHAP 值解释分析")
    print("=" * 70)

    shap_sample = X_reference.sample(n=min(2000, len(X_reference)), random_state=RANDOM_STATE).copy()
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "变量": feature_names,
        "平均绝对SHAP值": mean_abs,
    }).sort_values("平均绝对SHAP值", ascending=False).reset_index(drop=True)
    importance_df["排名"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)

    subsidy_row = importance_df.loc[importance_df["变量"] == SUBSIDY_FEATURE].iloc[0]
    subsidy_rank = int(subsidy_row["排名"])
    subsidy_importance = float(subsidy_row["平均绝对SHAP值"])

    print(f"  SHAP 计算样本量 = {len(shap_sample)}")
    print(f"  {SUBSIDY_FEATURE} 平均绝对SHAP值排名 = {subsidy_rank}")
    print(f"  {SUBSIDY_FEATURE} 平均绝对SHAP值 = {subsidy_importance:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = importance_df.sort_values("平均绝对SHAP值", ascending=True)
    ax.barh(plot_df["变量"], plot_df["平均绝对SHAP值"], color="#9C27B0")
    ax.set_xlabel("平均绝对SHAP值")
    ax.set_title("图4-2 SHAP值重要性图", fontproperties=_TITLE_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_shap_summary.png"), bbox_inches="tight")
    ax.set_title("")
    plt.savefig(os.path.join(BUNDLE_IMAGES_DIR, "fig3_shap_summary_notitle.png"), bbox_inches="tight")
    plt.close()

    return {
        "sample_size": int(len(shap_sample)),
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": subsidy_importance,
        "importance_table": importance_df.to_dict(orient="records"),
    }


def xgboost_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第四部分：XGBoost — 特征重要性与部分依赖检验")
    print("=" * 70)

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
    cv_scores = cross_val_score(
        xgb_model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    importance_df = pd.DataFrame({
        "变量": feature_names,
        "XGBoost重要性": xgb_model.feature_importances_,
    }).sort_values("XGBoost重要性", ascending=False).reset_index(drop=True)
    importance_df["排名"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_csv(os.path.join(output_dir, "xgb_importance.csv"), index=False)

    subsidy_row = importance_df.loc[importance_df["变量"] == SUBSIDY_FEATURE].iloc[0]
    subsidy_rank = int(subsidy_row["排名"])
    subsidy_importance = float(subsidy_row["XGBoost重要性"])
    top_features = importance_df.head(5)["变量"].tolist()

    pd_result = partial_dependence(
        xgb_model,
        X_train,
        features=[SUBSIDY_FEATURE],
        grid_resolution=40,
    )
    grid_values = pd_result["grid_values"][0]
    avg_values = pd_result["average"][0]
    pdp_df = pd.DataFrame({
        SUBSIDY_FEATURE: grid_values,
        "partial_dependence": avg_values,
    })
    pdp_df.to_csv(os.path.join(output_dir, "xgb_partial_dependence.csv"), index=False)

    pdp_direction = "上升" if avg_values[-1] > avg_values[0] else ("下降" if avg_values[-1] < avg_values[0] else "平缓")
    pdp_turn_points = int(np.sum(((avg_values[1:-1] - avg_values[:-2]) * (avg_values[2:] - avg_values[1:-1])) < 0))

    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 排名 = {subsidy_rank}")
    print(f"  {SUBSIDY_FEATURE} 重要性 = {subsidy_importance:.6f}")
    print(f"  {SUBSIDY_FEATURE} 部分依赖整体趋势 = {pdp_direction}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = importance_df.head(10).sort_values("XGBoost重要性", ascending=True)
    axes[0].barh(plot_df["变量"], plot_df["XGBoost重要性"], color="#4CAF50")
    axes[0].set_title("(a) XGBoost特征重要性")
    axes[0].set_xlabel("重要性")

    PartialDependenceDisplay.from_estimator(
        xgb_model,
        X_train,
        [SUBSIDY_FEATURE],
        ax=axes[1],
        line_kw={"color": "#FF5722", "linewidth": 2},
    )
    axes[1].set_title("(b) 财政补贴部分依赖图")
    axes[1].set_xlabel(SUBSIDY_FEATURE)
    axes[1].set_ylabel("预测的 Overpay")

    plt.suptitle("图4-3 XGBoost特征重要性与财政补贴部分依赖图", fontproperties=_TITLE_FONT, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_shap_subsidy.png"), bbox_inches="tight")
    axes[0].set_title("")
    axes[1].set_title("")
    plt.suptitle("")
    plt.savefig(os.path.join(BUNDLE_IMAGES_DIR, "fig4_shap_subsidy_notitle.png"), bbox_inches="tight")
    plt.close()

    return {
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": search.best_params_,
        "top_features": top_features,
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": subsidy_importance,
        "importance_table": importance_df.to_dict(orient="records"),
        "pdp_direction": pdp_direction,
        "pdp_start": float(avg_values[0]),
        "pdp_end": float(avg_values[-1]),
        "pdp_min": float(avg_values.min()),
        "pdp_max": float(avg_values.max()),
        "pdp_turn_points": pdp_turn_points,
        "pdp_values_head": [float(x) for x in avg_values[:5]],
        "pdp_values_tail": [float(x) for x in avg_values[-5:]],
    }


def save_ml_outputs(output_dir, prepared, splits, lasso_result, rf_result, shap_result, xgb_result):
    tuning_summary = {
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "prepared_sample_size": int(len(prepared["df_ml"])),
            "prepared_company_count": int(prepared["df_ml"]["Symbol"].nunique()),
            "train_size": int(len(splits["train_idx"])),
            "test_size_n": int(len(splits["test_idx"])),
            "train_company_count": int(splits["groups_train"].nunique()),
            "test_company_count": int(splits["groups_test"].nunique()),
            "holdout_method": "GroupShuffleSplit (by company Symbol)",
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "lasso": "GroupKFold(n_splits=5)",
            "random_forest": "GroupKFold(n_splits=5)",
            "xgboost": "GroupKFold(n_splits=5)",
        },
        "best_params": {
            "lasso": lasso_result["best_params"],
            "random_forest": rf_result["best_params"],
            "xgboost": xgb_result["best_params"],
        },
    }
    with open(os.path.join(output_dir, "ml_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(tuning_summary, f, ensure_ascii=False, indent=2)

    validation_summary = {
        "ml_sample_size": int(len(prepared["df_ml"])),
        "ml_company_count": int(prepared["df_ml"]["Symbol"].nunique()),
        "feature_names": prepared["feature_names"],
        "core_feature": SUBSIDY_FEATURE,
        "lasso": {
            "alpha": lasso_result["alpha"],
            "subsidy_retained": lasso_result["subsidy_retained"],
            "subsidy_coef": lasso_result["subsidy_coef"],
            "subsidy_sign": lasso_result["subsidy_sign"],
            "retained_count": len(lasso_result["retained"]),
            "retained": lasso_result["retained"],
            "removed": lasso_result["removed"],
            "coef_map": lasso_result["coef_map"],
        },
        "random_forest": {
            "subsidy_rank": rf_result["subsidy_rank"],
            "subsidy_importance": rf_result["subsidy_importance"],
            "top_features": rf_result["top_features"],
            "importance_table": rf_result["importance_table"],
        },
        "rf_shap": {
            "sample_size": shap_result["sample_size"],
            "subsidy_rank": shap_result["subsidy_rank"],
            "subsidy_importance": shap_result["subsidy_importance"],
            "importance_table": shap_result["importance_table"],
        },
        "xgboost": {
            "subsidy_rank": xgb_result["subsidy_rank"],
            "subsidy_importance": xgb_result["subsidy_importance"],
            "top_features": xgb_result["top_features"],
            "importance_table": xgb_result["importance_table"],
            "pdp_direction": xgb_result["pdp_direction"],
            "pdp_start": xgb_result["pdp_start"],
            "pdp_end": xgb_result["pdp_end"],
            "pdp_min": xgb_result["pdp_min"],
            "pdp_max": xgb_result["pdp_max"],
            "pdp_turn_points": xgb_result["pdp_turn_points"],
            "pdp_values_head": xgb_result["pdp_values_head"],
            "pdp_values_tail": xgb_result["pdp_values_tail"],
        },
    }
    with open(os.path.join(output_dir, "ml_validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(
        [
            {"模型": "随机森林", "变量": row["变量"], "重要性": row["随机森林重要性"], "排名": row["排名"]}
            for row in rf_result["importance_table"]
        ]
    ).to_csv(os.path.join(output_dir, "ml_validation_summary.csv"), index=False)


def main():
    data_dir = os.path.join(ROOT_DIR, "processed_data")
    output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(BUNDLE_IMAGES_DIR, exist_ok=True)

    print("加载数据...")
    df, _ = build_analysis_dataset(data_dir)

    prepared = prepare_ml_data(df)
    splits = create_holdout_splits(prepared["X"], prepared["y"], prepared["df_ml"])

    print("\n" + "#" * 70)
    print("#  机器学习稳健性检验开始")
    print("#" * 70)

    rf_result = random_forest_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )
    rf_shap_result = rf_shap_analysis(
        rf_result["model"],
        splits["X_test"],
        prepared["feature_names"],
        output_dir,
    )
    lasso_result = lasso_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )
    xgb_result = xgboost_analysis(
        splits["X_train"],
        splits["X_test"],
        splits["y_train"],
        splits["y_test"],
        splits["groups_train"],
        prepared["feature_names"],
        output_dir,
    )

    save_ml_outputs(
        output_dir,
        prepared,
        splits,
        lasso_result,
        rf_result,
        rf_shap_result,
        xgb_result,
    )

    print("\n" + "=" * 70)
    print("机器学习稳健性检验完成")
    print("=" * 70)
    print(
        "已生成：lasso_alpha_search.csv、lasso_coefficients.csv、rf_reg_importance.csv、"
        "shap_importance.csv、xgb_importance.csv、xgb_partial_dependence.csv、"
        "ml_validation_summary.csv、ml_validation_summary.json 及图4-1到图4-4。"
    )


if __name__ == "__main__":
    main()
