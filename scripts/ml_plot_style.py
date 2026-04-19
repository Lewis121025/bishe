from __future__ import annotations

import matplotlib.pyplot as plt

SINGLE_FIGSIZE = (10.6, 6.1)
WIDE_FIGSIZE = (11.8, 5.2)
BAR_HEIGHT = 0.46
BASE_BAR_COLOR = "#4F81BD"
HIGHLIGHT_COLOR = "#D97732"
GRID_COLOR = "#D9E2F0"
LINE_COLORS = {
    "lnSubsidy_pos_l1": "#D97732",
    "Roa": "#4F81BD",
    "Lever": "#A9B8CC",
    "Top1": "#7E95B8",
}

FEATURE_LABELS = {
    "Roa": "盈利能力\nRoa",
    "Top1": "股权集中度\nTop1",
    "lnSubsidy_pos_l1": "补贴强度\nlnSubsidy_pos_l1",
    "Lever": "财务杠杆\nLever",
}


def get_feature_labels(feature_names: list[str]) -> list[str]:
    return [FEATURE_LABELS.get(name, name) for name in feature_names]


def get_rank_bar_colors(
    feature_names: list[str], highlight_feature: str = "lnSubsidy_pos_l1"
) -> list[str]:
    return [
        HIGHLIGHT_COLOR if name == highlight_feature else BASE_BAR_COLOR
        for name in feature_names
    ]


def highlight_rgb() -> tuple[float, float, float]:
    color = HIGHLIGHT_COLOR.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def build_rank_bar_chart(plot_df, value_col: str, xlabel: str):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    feature_names = plot_df["变量"].tolist()
    labels = get_feature_labels(feature_names)
    colors = get_rank_bar_colors(feature_names)

    ax.barh(labels, plot_df[value_col].tolist(), color=colors, height=BAR_HEIGHT)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def build_lasso_path_chart(alpha_grid, coef_paths, feature_names, best_alpha: float):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    display_names = get_feature_labels(feature_names)

    for idx, feature_name in enumerate(feature_names):
        ax.plot(
            alpha_grid,
            [row[idx] for row in coef_paths],
            label=display_names[idx],
            color=LINE_COLORS.get(feature_name, BASE_BAR_COLOR),
            linewidth=2,
        )

    ax.axvline(
        best_alpha,
        color="#333333",
        linestyle="--",
        linewidth=1.6,
        label=f"最优α={best_alpha:.4g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("正则化参数 α（log）")
    ax.set_ylabel("Lasso 回归系数")
    ax.grid(True, color=GRID_COLOR, alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    return fig, ax


def build_importance_pdp_chart(
    importance_df,
    importance_col: str,
    pdp_df,
    subsidy_feature: str,
):
    fig, axes = plt.subplots(1, 2, figsize=WIDE_FIGSIZE)

    ordered_df = importance_df.sort_values(importance_col, ascending=True)
    colors = get_rank_bar_colors(ordered_df["变量"].tolist(), highlight_feature=subsidy_feature)
    axes[0].barh(
        get_feature_labels(ordered_df["变量"].tolist()),
        ordered_df[importance_col].tolist(),
        color=colors,
        height=BAR_HEIGHT,
    )
    axes[0].set_xlabel("重要性")
    axes[0].grid(axis="x", color=GRID_COLOR, alpha=0.7, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(
        pdp_df[subsidy_feature].tolist(),
        pdp_df["partial_dependence"].tolist(),
        color=HIGHLIGHT_COLOR,
        linewidth=2.4,
    )
    axes[1].set_xlabel(f"补贴强度（{subsidy_feature}）")
    axes[1].set_ylabel("部分依赖值")
    axes[1].grid(True, color=GRID_COLOR, alpha=0.7, linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    return fig, axes
