# Thesis ML Reframing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the thesis machine-learning content as an OLS supplementary validation section without changing the main OLS regression route.

**Architecture:** The work has four parts: document the approved narrative, refactor the ML script to produce only thesis-relevant outputs, rewrite the thesis markdown to merge ML into Chapter 4, and update consistency checks/docs before rerunning verification outputs.

**Tech Stack:** Markdown, Python, pandas, scikit-learn, xgboost, matplotlib

---

### Task 1: Align The Thesis Narrative

**Files:**
- Modify: `thesis_final_bundle/thesis_final_humanized_v2.md`

- [ ] **Step 1: Rewrite the ML-method section in Chapter 3**

Replace the Chapter 3 ML method block so it describes only the approved three-model validation roles, removes classification/prediction language, and explains that ML supplements OLS through importance, coefficient retention/direction, and partial-dependence trend checks.

- [ ] **Step 2: Move ML analysis into Chapter 4**

Replace the standalone Chapter 5 block with a new Chapter 4 final subsection that:
- states the baseline OLS coefficient is positive but insignificant,
- presents random forest as an importance-ranking cross-check,
- presents Lasso as a retention/sign cross-check,
- presents XGBoost as an importance + partial-dependence cross-check,
- avoids model horse-race wording.

- [ ] **Step 3: Renumber the conclusion chapter**

Rename the existing conclusion chapter from Chapter 6 to Chapter 5 and update any nearby transition wording that still references the old Chapter 5 structure.

- [ ] **Step 4: Rewrite summary language**

Update the Chinese and English abstracts, Chapter 1 technical-route wording, conclusion bullets, policy wording, and limitation wording so they match the new ML positioning.

### Task 2: Refactor The ML Reproduction Script

**Files:**
- Modify: `scripts/ml_analysis.py`

- [ ] **Step 1: Remove non-approved analysis branches**

Delete or stop invoking classification comparison, decision tree analysis, K-means analysis, and regression model horse-race outputs.

- [ ] **Step 2: Keep the three approved model workflows**

Retain Lasso, random forest, and XGBoost analysis, but rewrite function docstrings, console text, saved summaries, and plot titles so they match the thesis framing.

- [ ] **Step 3: Add XGBoost partial dependence output**

Generate a partial-dependence plot for `lnSubsidy` from the XGBoost regression model and save it as a thesis-facing output.

- [ ] **Step 4: Save only the needed summary artifacts**

Write focused CSV/JSON outputs that support the thesis text, including retained Lasso coefficients, random-forest importance, XGBoost importance, partial-dependence support, and an ML validation summary.

### Task 3: Update Documentation And Consistency Checks

**Files:**
- Modify: `scripts/verify_thesis_consistency.py`
- Modify: `results/README.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Update README descriptions**

Rewrite the ML-related sections in the README files so they describe the new supplementary-validation outputs rather than prediction/classification/comparison outputs.

- [ ] **Step 2: Update thesis-consistency checks**

Replace the old ML expectations in `verify_thesis_consistency.py` with checks for the new thesis wording and new ML result files.

### Task 4: Regenerate And Verify Outputs

**Files:**
- Modify: `results/*` generated artifacts as needed
- Modify: `thesis_final_bundle/thesis_final_humanized_v2_embedded.md`

- [ ] **Step 1: Run the refactored ML script**

Run: `python scripts/ml_analysis.py`
Expected: script completes and writes the new ML validation outputs without classification or clustering summaries.

- [ ] **Step 2: Refresh the embedded thesis markdown**

Run: `./scripts/update_embedded_markdown.sh`
Expected: `thesis_final_bundle/thesis_final_humanized_v2_embedded.md` updates successfully.

- [ ] **Step 3: Run consistency verification**

Run: `python scripts/verify_thesis_consistency.py`
Expected: consistency check passes against the updated thesis wording and current results.
