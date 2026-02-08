# Student Success Analysis

## Project Documentation

### Project Description
This project analyzes a dataset of ~20,000 student records to understand which factors influence final exam scores and to build an interpretable predictive model using linear regression (OLS). Beyond the empirical results, a central goal of the project is learning-driven: using explicit model design, comparison, and diagnostics to deepen our understanding of core regression concepts. The emphasis is therefore on explanation + validation (coefficients, uncertainty, diagnostics) rather than black-box machine learning.

**Main objectives**
- Clean and standardize raw CSV data into a fully numeric modeling table.
- Build three regression specifications (Baseline / Behavior / Full) to tell a clear story about what adds explanatory power.
- Estimate effects with confidence intervals and p-values (with optional robust standard errors).
- Run core diagnostics (multicollinearity via VIF; residual sanity checks).
- Produce plots that support the written conclusions.
- Use the structure of the pipeline as a learning tool to deepen our understanding of key regression concepts (model specification, inference, diagnostics, and robustness), while keeping model-comparison decisions explicit and reproducible.

**Expected outcomes / analytical expectations**
- Study- and lifestyle-related variables should explain a substantial portion of exam score variance.
- Some demographic/background predictors may be statistically significant but have smaller effect sizes.
- Heteroskedasticity is plausible in real student-performance data, so robust (HC) standard errors may be appropriate.

**Hypotheses**
- Behavioral variables (study effort, sleep duration, sleep quality) explain more variance in exam scores than structural or demographic variables alone.
- Adding behavioral predictors to a baseline model will lead to a meaningful increase in R² and adjusted R².
- Sleep quality has a stronger association with exam performance than raw sleep duration.
- Very low and very high sleep duration are associated with lower exam scores, suggesting a non-linear or diminishing-returns relationship.
- Study effort has a positive association with exam scores, but its marginal benefit decreases at higher levels.
- Exam difficulty acts as a confounder and should be controlled for when estimating behavioral effects.
- Some demographic/background variables may remain statistically significant, but their practical effect sizes are expected to be smaller than behavioral predictors.

---

## Folder / Module Structure

```
Student-Success-Analysis/
│
├── data/
│   └── Exam_Score_Prediction.csv   # Dataset (default path in conf.py)
│
├── main.py                          # Entry point: run pipeline end-to-end
├── conf.py                          # Configuration (paths, target, options)
├── data.py                          # Cleaning + feature preparation
├── design.py                        # Build X/y for each model specification
├── ols_engine.py                    # OLS wrapper (fit + predict)
├── diagnostics.py                   # VIF + residual summary (+ optional EDA scan)
├── reporting.py                     # Centralized logging & model summaries
├── results.py                       # Plotting utilities and result visualizations
│
├── tests/                           # Automated tests
│   ├── test_data.py                 # Tests for cleaning & role inference
│   ├── test_design.py               # Tests for model design matrices
│   ├── test_ols_engine.py            # Tests for OLS fitting & outputs
│   └── test_diagnostics.py           # Tests for VIF & residual summaries
│
├── requirements.txt                 # Dependencies
├── README.md                        # Documentation
└── LICENSE                          # MIT license
```

---

## Key Stages of the Analysis Pipeline

### 1) Data import (`data.load_data`)
- Reads the CSV (default: `data/Exam_Score_Prediction.csv`).

### 2) Data cleaning & feature preparation (`data.clean_data`)
Cleaning is intentionally strict to keep the modeling table numeric and consistent.

What happens:
- **Column standardization**: column names converted to a consistent snake_case style.
- **Missing handling**: common “missing tokens” in object columns (`""`, `na`, `null`, etc.) are normalized to NaN, then rows with missing values are dropped.
- **Categorical inference** (object → numeric):
  - Binary sets like `{yes/no, true/false}` become `<col>_num` (0/1).
  - Ordinal sets like `{poor/average/good}`, `{easy/moderate/hard}`, `{low/medium/high}` become `<col>_num` (1..3).
  - When a `_num` column is created, the original text column is dropped.
- **Numeric coercion for numeric-like strings**: object columns that are ≥90% numeric and have enough unique values are converted to numeric (identifier-like columns are skipped).
- **Automatic log features for skewed continuous predictors**:
  - For eligible continuous numeric columns (non-negative, sufficiently continuous, right-skew > threshold), a `log1p` feature `<col>_log` is created and the original column is dropped.
- **IQR outlier removal (global)**:
  - Applied only to continuous numeric columns (skips low-cardinality/binary/ordinal columns).
  - A row is removed if it is an outlier in **any** included continuous column.

### 3) Column role inference (`data.infer_column_roles`)
After cleaning, predictors are categorized automatically based on numeric cardinality:
- **binary**: exactly 2 unique values
- **ordinal**: 3..`max_ordinal_levels` unique values (default 7)
- **continuous**: more than `max_ordinal_levels`

Identifier-like columns (e.g., `student_id` or `*_id`) are excluded from predictors in `main.py`.

### 4) Build three model specifications (`design.py`)
- **Baseline model**: binary + ordinal predictors (structural/background factors).
- **Behavior model**: continuous predictors + default controls (adds `exam_difficulty_num` if present).
- **Full model**: binary + ordinal + continuous predictors.

### 5) Fit OLS models (`ols_engine.run_ols`)
- Uses `statsmodels` OLS.
- Optionally applies heteroskedasticity-robust covariance (`HC0/HC1/HC2/HC3`).
- Returns a standardized output object with:
  - coefficient table (coef, SE, t, p, CI)
  - model metrics (R², adj R², n, df_model)

### 6) Diagnostics (`diagnostics.py`)
- **VIF** for multicollinearity (on the Full model X).
- **Residual summary**: mean/std/min/max (quick sanity check).

### 7) Model comparison + plots (saved to `outputs/`)
`main.py` saves:
- `r2_comparison.png` (Baseline vs Behavior vs Full)
- `coef_forest_full.png` (top 15 coefficients by p-value with 95% CI)
- `actual_vs_pred_full.png` (Full model: predicted vs actual)
- `residuals_vs_fitted_full.png` (Full model residuals vs fitted)

---

## Important Definitions
- **OLS (Ordinary Least Squares)**: fits a linear model by minimizing squared residuals.
- **Robust HC standard errors**: adjusts inference when error variance is not constant (heteroskedasticity).
- **VIF (Variance Inflation Factor)**: measures multicollinearity.
- **IQR outlier rule**: flags points outside `[Q1 − k·IQR, Q3 + k·IQR]`.
- **log1p transform**: `log(1 + x)`, safe for zeros.

---

## Data Description
- **File**: `Exam_Score_Prediction.csv`
- **Target**: `exam_score`
- **Predictors**: demographic/background, study habits, sleep/lifestyle variables, exam difficulty.

---

## Instructions for Running the Project

```bash
pip install -r requirements.txt
python main.py
```
    
Outputs are saved to `outputs/`.

---

## License
MIT License. See `LICENSE`.