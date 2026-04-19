import unittest

import pandas as pd


class MlPlotStyleTest(unittest.TestCase):
    def test_rank_bar_chart_localizes_labels_and_highlights_subsidy(self):
        from scripts import ml_plot_style

        labels = ml_plot_style.get_feature_labels(
            ["Roa", "Top1", "lnSubsidy_pos_l1", "Lever"]
        )
        self.assertEqual(
            labels,
            [
                "盈利能力\nRoa",
                "股权集中度\nTop1",
                "补贴强度\nlnSubsidy_pos_l1",
                "财务杠杆\nLever",
            ],
        )

        colors = ml_plot_style.get_rank_bar_colors(
            ["Roa", "Top1", "lnSubsidy_pos_l1", "Lever"]
        )
        self.assertEqual(colors.count(ml_plot_style.HIGHLIGHT_COLOR), 1)
        self.assertEqual(colors[2], ml_plot_style.HIGHLIGHT_COLOR)
        self.assertEqual(
            tuple(round(v, 1) for v in ml_plot_style.SINGLE_FIGSIZE),
            (10.6, 6.1),
        )

    def test_rank_bar_chart_uses_shared_layout_and_display_labels(self):
        from scripts import ml_plot_style

        plot_df = pd.DataFrame(
            {
                "变量": ["Lever", "lnSubsidy_pos_l1", "Top1", "Roa"],
                "随机森林重要性": [0.16, 0.23, 0.26, 0.34],
            }
        )

        fig, ax = ml_plot_style.build_rank_bar_chart(
            plot_df,
            value_col="随机森林重要性",
            xlabel="重要性",
        )

        self.assertEqual(
            tuple(round(v, 1) for v in fig.get_size_inches()),
            (10.6, 6.1),
        )
        self.assertEqual(ax.get_xlabel(), "重要性")
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            [
                "财务杠杆\nLever",
                "补贴强度\nlnSubsidy_pos_l1",
                "股权集中度\nTop1",
                "盈利能力\nRoa",
            ],
        )
        self.assertEqual(
            ax.patches[1].get_facecolor()[:3],
            ml_plot_style.highlight_rgb(),
        )
        self.assertTrue(all(round(p.get_height(), 2) == 0.46 for p in ax.patches))
        fig.clf()

    def test_lasso_chart_marks_best_alpha_with_shared_layout(self):
        from scripts import ml_plot_style

        alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1]
        coef_paths = [
            [0.07, -0.02, -0.06, -0.06],
            [0.065, -0.018, -0.058, -0.06],
            [0.05, -0.005, -0.04, -0.05],
            [0.0, 0.0, 0.0, 0.0],
        ]

        fig, ax = ml_plot_style.build_lasso_path_chart(
            alpha_grid=alpha_grid,
            coef_paths=coef_paths,
            feature_names=["lnSubsidy_pos_l1", "Roa", "Lever", "Top1"],
            best_alpha=1e-3,
        )

        self.assertEqual(
            tuple(round(v, 1) for v in fig.get_size_inches()),
            (10.6, 6.1),
        )
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), "正则化参数 α（log）")
        self.assertEqual(len(ax.lines), 5)
        self.assertEqual(ax.lines[0].get_color(), ml_plot_style.HIGHLIGHT_COLOR)
        fig.clf()

    def test_xgb_chart_uses_wide_layout_and_localized_axis_label(self):
        from scripts import ml_plot_style

        importance_df = pd.DataFrame(
            {
                "变量": ["Roa", "lnSubsidy_pos_l1", "Top1", "Lever"],
                "XGBoost重要性": [0.31, 0.27, 0.23, 0.18],
            }
        )
        pdp_df = pd.DataFrame(
            {
                "lnSubsidy_pos_l1": [14.0, 15.0, 16.0, 17.0],
                "partial_dependence": [-0.08, -0.05, -0.01, 0.04],
            }
        )

        fig, axes = ml_plot_style.build_importance_pdp_chart(
            importance_df=importance_df,
            importance_col="XGBoost重要性",
            pdp_df=pdp_df,
            subsidy_feature="lnSubsidy_pos_l1",
        )

        self.assertEqual(
            tuple(round(v, 1) for v in fig.get_size_inches()),
            (11.8, 5.2),
        )
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_xlabel(), "重要性")
        self.assertEqual(axes[1].get_xlabel(), "补贴强度（lnSubsidy_pos_l1）")
        self.assertTrue(all(round(p.get_height(), 2) == 0.46 for p in axes[0].patches))
        fig.clf()


if __name__ == "__main__":
    unittest.main()
