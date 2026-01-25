'''
Aging sleeping time connection
Aging  sleeping quality connection
'''
import pandas as pd
import numpy as np

from scipy import stats
import statsmodels.api as sm


def sleeping_time_age_correlation(df):
    '''the following function will calculate the correlation between age and sleeping time - by crating temp df with log values
    for desired cells values, creating more stable calculations via pandas commands'''
    tmp_df = df.copy()
    corr = tmp_df['sleep_time_log'].corr(tmp_df['age_log']) #calculating correlation between values
    '''
    close to 1 - sleeping more as aging, close to 0 - no connection, close to (-1) - sleeping less as aging
    '''
    return corr


def age_sleep_quality_correlation(df):
    '''the following function will calculate the correlation between age and sleeping quality - by crating temp df with log values
        for desired cells values, creating more stable calculations via pandas commands'''
    tmp_df = df.copy()
    corr = tmp_df['age_log'].corr(tmp_df['sleep_quality_log']) #calculating correlation between values
    '''
    close to 1 - sleeping better as aging, close to 0 - no connection, close to (-1) - sleeping worse as aging
    '''
    return corr


#
# ----------------------------
# Regression-based inference
# ----------------------------

def _zscore(series: pd.Series) -> pd.Series:
    """Standardize a numeric series (mean=0, sd=1)."""
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * np.nan
    return (s - s.mean()) / sd


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for a joint regression model of exam_score.
    Encodes:
    - Binary: internet_access (yes/no -> 1/0)
    - Ordinal: exam_difficulty, sleep_quality (expects *_num already present for sleep_quality)
    - Nominal: gender, study_method, course (one-hot)
    """
    df = df.copy()

    # Target
    y = pd.to_numeric(df["exam_score"], errors="coerce")

    # Numeric predictors (adjust if your dataset differs)
    numeric_cols = [
        "age",
        "study_hours",
        "class_attendance",
        "sleep_hours",
        "facility_rating",
    ]
    present_numeric = [c for c in numeric_cols if c in df.columns]
    X_num = df[present_numeric].apply(pd.to_numeric, errors="coerce")

    # Ordinal / mapped predictors
    # sleep_quality_num is created in data.py via sleep_quality_map
    ord_cols = []
    if "sleep_quality_num" in df.columns:
        ord_cols.append("sleep_quality_num")

    # Map exam_difficulty if it is a string column
    if "exam_difficulty" in df.columns:
        if df["exam_difficulty"].dtype == object:
            diff_map = {"easy": 0, "moderate": 1, "hard": 2}
            df["exam_difficulty_num"] = (
                df["exam_difficulty"].astype(str).str.lower().str.strip().map(diff_map)
            )
            ord_cols.append("exam_difficulty_num")
        else:
            ord_cols.append("exam_difficulty")

    X_ord = df[ord_cols] if ord_cols else pd.DataFrame(index=df.index)

    # Binary: internet_access
    X_bin = pd.DataFrame(index=df.index)
    if "internet_access" in df.columns:
        if df["internet_access"].dtype == object:
            X_bin["internet_access"] = (
                df["internet_access"].astype(str).str.lower().str.strip().isin(["yes", "y", "true", "1"]).astype(int)
            )
        else:
            X_bin["internet_access"] = pd.to_numeric(df["internet_access"], errors="coerce")

    # Nominal one-hot
    nominal_cols = [c for c in ["gender", "study_method", "course"] if c in df.columns]
    X_nom = pd.get_dummies(df[nominal_cols], drop_first=True) if nominal_cols else pd.DataFrame(index=df.index)

    # Combine
    X = pd.concat([X_num, X_ord, X_bin, X_nom], axis=1)

    # Drop rows with missing in any model column
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    return X, y


def fit_main_regression(df: pd.DataFrame):
    """Fit OLS regression for exam_score using the joint design matrix."""
    X, y = build_design_matrix(df)
    model = sm.OLS(y, X).fit()
    return model


def summarize_effects(model) -> pd.DataFrame:
    """
    Return a tidy table of coefficients, p-values, and 95% CI.
    Also includes standardized coefficients for numeric predictors.
    """
    params = model.params
    conf = model.conf_int(alpha=0.05)
    pvals = model.pvalues

    out = pd.DataFrame({
        "coef": params,
        "p_value": pvals,
        "ci95_lo": conf[0],
        "ci95_hi": conf[1],
    })

    # Standardized betas (approx) for non-dummy predictors: beta_std = beta * sd(x)/sd(y)
    y_sd = model.model.endog.std(ddof=0)
    std_betas = {}
    for name in model.model.exog_names:
        if name == "const":
            continue
        x = model.model.exog[:, model.model.exog_names.index(name)]
        x_sd = np.std(x, ddof=0)
        if x_sd == 0 or y_sd == 0:
            std_betas[name] = np.nan
        else:
            std_betas[name] = params[name] * (x_sd / y_sd)

    out["beta_std"] = pd.Series(std_betas)

    # Rank by absolute standardized beta (ignoring NaNs)
    out["abs_beta_std"] = out["beta_std"].abs()
    out = out.sort_values(by="abs_beta_std", ascending=False)

    return out


def best_study_method(df):
    """
    The following function will sort the most successful learning method by returning a list where 0 index is the best
    scoring method, any method that wont be included in the list have a small data pool (under 5) that cant give accurate info
    """
    tmp_df = df.copy()
    
    method_size = tmp_df.groupby('study_method')['exam_score'].agg(['mean','count'])
    #grouping, counting and mean calculating
    reliable_method_size = method_size[(method_size['count'] > 5)] #not be affected by few unordinary cases
    sorted_method = reliable_method_size.sort_values(by=['mean'], ascending=False) #sorting method where best performing at [0]
    return sorted_method['mean'].to_dict()









def chi_square_table(df: pd.DataFrame, a: str, b: str) -> dict:
    """Run chi-square test for association between two categorical columns."""
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "p": np.nan, "dof": np.nan, "table": ct}

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = int(ct.to_numpy().sum())
    v = np.sqrt(chi2 / (n * (min(ct.shape[0] - 1, ct.shape[1] - 1)))) if n else np.nan

    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(v), "table": ct}