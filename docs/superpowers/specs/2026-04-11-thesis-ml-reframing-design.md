# Thesis Machine-Learning Reframing Design

**Date:** 2026-04-11

## Goal

Keep the existing OLS main-regression route unchanged, and rewrite the machine-learning material so it functions as a supplementary validation section for the OLS findings rather than a standalone prediction chapter.

## Approved Direction

The thesis will keep the current fourth-chapter empirical main line:

- The baseline full-sample OLS result remains positive but statistically insignificant.
- The machine-learning section will no longer claim predictive superiority or replace econometric inference.
- Random forest, Lasso, and XGBoost will each be rewritten as serving one validation role relative to the OLS main regression.

## Section-Level Changes

### Chapter structure

- Remove the standalone Chapter 5 machine-learning chapter.
- Insert a new final subsection under Chapter 4, positioned after heterogeneity analysis.
- Renumber the conclusion chapter from Chapter 6 to Chapter 5.

### Machine-learning narrative

- Random forest: validate whether fiscal subsidy remains near the top of the feature-importance ranking.
- Lasso: validate whether the fiscal subsidy variable is retained and whether its coefficient direction stays positive.
- XGBoost: validate whether the partial-dependence relationship for fiscal subsidy is upward overall, while fiscal subsidy remains important in the model.

### Material to remove or downgrade

- Prediction-centered framing
- Classification comparison
- Decision tree analysis
- K-means clustering
- Model horse-race wording such as "which model performs best"
- SHAP interaction storytelling that turns the section into a separate research chapter

## Implementation Targets

- Rewrite `thesis_final_bundle/thesis_final_humanized_v2.md`
- Refocus `scripts/ml_analysis.py` on the three approved models and outputs
- Update `scripts/verify_thesis_consistency.py`
- Update `results/README.md` and `scripts/README.md`
- Regenerate ML result files and embedded thesis markdown if the updated script runs successfully

## Writing Constraints

- Do not change the main regression route or rewrite the baseline OLS result as significant.
- Do not overclaim that machine learning proves causality or significance.
- Keep the final wording close to the advisor's framing: machine learning adds support through importance ranking, coefficient retention/direction, and trend consistency.
