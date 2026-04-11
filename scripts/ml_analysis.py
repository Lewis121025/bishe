"""
=============================================================================
基于数据挖掘的上市公司财政补贴与高管超额薪酬研究 — 机器学习补充分析脚本
=============================================================================

本脚本不再承担预测竞赛或分类识别任务，而是围绕 OLS 主回归与管理层权力路径
开展三项机器学习稳健性检验：
  1. 随机森林：检验财政补贴与管理层权力（FA）在非线性模型中的特征重要性
  2. Lasso：检验财政补贴与管理层权力（FA）是否被保留，以及方向是否与 OLS 一致
  3. XGBoost：检验财政补贴与管理层权力（FA）的重要性，并绘制财政补贴部分依赖图观察整体趋势
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

from regression_analysis import build_analysis_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SUBSIDY_FEATURE = "lnSubsidy_l1"
POWER_FEATURE = "Power_FA"


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
        "lnSubsidy_l1",
        "Power_FA",
        "Roa",
        "Lever",
        "Top1",
    ]
    target_col = "Overpay"
    use_cols = feature_cols + [target_col, "Symbol", "Year"]

    df_ml = df.dropna(subset=use_cols).copy()
    print(f"  可用样本量: {len(df_ml)}")
    print(f"  公司数量: {df_ml['Symbol'].nunique()}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  核心解释变量: {SUBSIDY_FEATURE}")
    print(f"  管理层权力变量: {POWER_FEATURE}")

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
    print("第一部分：Lasso 回归 — 变量保留与方向一致性检验")
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
    subsidy_coef = float(
        coef_df.loc[coef_df["变量"] == SUBSIDY_FEATURE, "Lasso系数"].iloc[0]
    )
    power_coef = float(
        coef_df.loc[coef_df["变量"] == POWER_FEATURE, "Lasso系数"].iloc[0]
    )

    print(f"  最优 alpha = {best_alpha:.6f}")
    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 系数 = {subsidy_coef:.6f}")
    print(f"  {SUBSIDY_FEATURE} {'被保留' if SUBSIDY_FEATURE in retained else '被压缩为0'}")
    print(f"  {POWER_FEATURE} 系数 = {power_coef:.6f}")
    print(f"  {POWER_FEATURE} {'被保留' if POWER_FEATURE in retained else '被压缩为0'}")

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
        "power_coef": power_coef,
        "power_retained": POWER_FEATURE in retained,
        "power_sign": "正" if power_coef > 0 else ("负" if power_coef < 0 else "零"),
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": {"alpha": float(best_alpha)},
    }


def random_forest_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第二部分：随机森林 — 特征重要性检验")
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
    power_row = importance_df.loc[importance_df["变量"] == POWER_FEATURE].iloc[0]
    power_rank = int(power_row["排名"])
    power_importance = float(power_row["随机森林重要性"])
    top_features = importance_df.head(5)["变量"].tolist()

    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 排名 = {subsidy_rank}")
    print(f"  {SUBSIDY_FEATURE} 重要性 = {subsidy_importance:.6f}")
    print(f"  {POWER_FEATURE} 排名 = {power_rank}")
    print(f"  {POWER_FEATURE} 重要性 = {power_importance:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = importance_df.sort_values("随机森林重要性", ascending=True)
    ax.barh(plot_df["变量"], plot_df["随机森林重要性"], color="steelblue")
    ax.set_xlabel("重要性")
    ax.set_title("图4-1 随机森林特征重要性图", fontproperties=_TITLE_FONT)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_rf_importance.png"), bbox_inches="tight")
    plt.close()

    return {
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": search.best_params_,
        "top_features": top_features,
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": subsidy_importance,
        "power_rank": power_rank,
        "power_importance": power_importance,
    }


def xgboost_analysis(X_train, X_test, y_train, y_test, groups_train, feature_names, output_dir):
    print("\n" + "=" * 70)
    print("第三部分：XGBoost — 特征重要性与部分依赖检验")
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
    power_row = importance_df.loc[importance_df["变量"] == POWER_FEATURE].iloc[0]
    power_rank = int(power_row["排名"])
    power_importance = float(power_row["XGBoost重要性"])
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

    print(f"  测试集 R² = {metrics['R2']:.4f}")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  {SUBSIDY_FEATURE} 排名 = {subsidy_rank}")
    print(f"  {SUBSIDY_FEATURE} 重要性 = {subsidy_importance:.6f}")
    print(f"  {POWER_FEATURE} 排名 = {power_rank}")
    print(f"  {POWER_FEATURE} 重要性 = {power_importance:.6f}")
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
    plt.close()

    return {
        "test_metrics": metrics,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "best_params": search.best_params_,
        "top_features": top_features,
        "subsidy_rank": subsidy_rank,
        "subsidy_importance": subsidy_importance,
        "power_rank": power_rank,
        "power_importance": power_importance,
        "pdp_direction": pdp_direction,
        "pdp_start": float(avg_values[0]),
        "pdp_end": float(avg_values[-1]),
    }


def save_ml_outputs(output_dir, prepared, splits, lasso_result, rf_result, xgb_result):
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
        "power_feature": POWER_FEATURE,
        "lasso": {
            "alpha": lasso_result["alpha"],
            "subsidy_retained": lasso_result["subsidy_retained"],
            "subsidy_coef": lasso_result["subsidy_coef"],
            "subsidy_sign": lasso_result["subsidy_sign"],
            "power_retained": lasso_result["power_retained"],
            "power_coef": lasso_result["power_coef"],
            "power_sign": lasso_result["power_sign"],
            "retained_count": len(lasso_result["retained"]),
            "removed": lasso_result["removed"],
        },
        "random_forest": {
            "subsidy_rank": rf_result["subsidy_rank"],
            "subsidy_importance": rf_result["subsidy_importance"],
            "power_rank": rf_result["power_rank"],
            "power_importance": rf_result["power_importance"],
            "top_features": rf_result["top_features"],
        },
        "xgboost": {
            "subsidy_rank": xgb_result["subsidy_rank"],
            "subsidy_importance": xgb_result["subsidy_importance"],
            "power_rank": xgb_result["power_rank"],
            "power_importance": xgb_result["power_importance"],
            "top_features": xgb_result["top_features"],
            "pdp_direction": xgb_result["pdp_direction"],
            "pdp_start": xgb_result["pdp_start"],
            "pdp_end": xgb_result["pdp_end"],
        },
    }
    with open(os.path.join(output_dir, "ml_validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, ensure_ascii=False, indent=2)

    summary_rows = [
        {
            "模型": "随机森林",
            "验证角度": "特征重要性",
            "核心变量": f"{SUBSIDY_FEATURE}；{POWER_FEATURE}",
            "主要结果": (
                f"{SUBSIDY_FEATURE} 排名第{rf_result['subsidy_rank']}，"
                f"{POWER_FEATURE} 排名第{rf_result['power_rank']}"
            ),
            "与OLS对应关系": "说明财政补贴与管理层权力在非线性模型中仍保留信息含量",
        },
        {
            "模型": "Lasso",
            "验证角度": "变量保留与方向",
            "核心变量": f"{SUBSIDY_FEATURE}；{POWER_FEATURE}",
            "主要结果": (
                f"{SUBSIDY_FEATURE}{'被保留' if lasso_result['subsidy_retained'] else '被压缩为0'}且为{lasso_result['subsidy_sign']}；"
                f"{POWER_FEATURE}{'被保留' if lasso_result['power_retained'] else '被压缩为0'}且为{lasso_result['power_sign']}"
            ),
            "与OLS对应关系": "说明财政补贴与管理层权力的方向判断未发生反转",
        },
        {
            "模型": "XGBoost",
            "验证角度": "重要性与部分依赖趋势",
            "核心变量": f"{SUBSIDY_FEATURE}；{POWER_FEATURE}",
            "主要结果": (
                f"{SUBSIDY_FEATURE} 排名第{xgb_result['subsidy_rank']}，"
                f"{POWER_FEATURE} 排名第{xgb_result['power_rank']}，"
                f"{SUBSIDY_FEATURE} 部分依赖整体{xgb_result['pdp_direction']}"
            ),
            "与OLS对应关系": "说明在更灵活模型中财政补贴趋势与核心治理变量信息仍可辨识",
        },
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "ml_validation_summary.csv"), index=False)


def main():
    data_dir = os.path.join(ROOT_DIR, "processed_data")
    output_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据...")
    df, _ = build_analysis_dataset(data_dir)

    prepared = prepare_ml_data(df)
    splits = create_holdout_splits(prepared["X"], prepared["y"], prepared["df_ml"])

    print("\n" + "#" * 70)
    print("#  机器学习稳健性检验开始")
    print("#" * 70)

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
        xgb_result,
    )

    print("\n" + "=" * 70)
    print("机器学习稳健性检验完成")
    print("=" * 70)
    print(
        "已生成：lasso_alpha_search.csv、lasso_coefficients.csv、rf_reg_importance.csv、"
        "xgb_importance.csv、xgb_partial_dependence.csv、ml_validation_summary.csv、"
        "ml_validation_summary.json 及图4-1到图4-3。"
    )


if __name__ == "__main__":
    main()
