"""OLS regression engine (statsmodels wrapper).

Library-only:
- no CSV loading
- no printing
- no running analysis on import
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

import statsmodels.api as sm  # type: ignore


@dataclass(frozen=True)
class ols_outputs:
    """Container for standardized OLS outputs."""
    model: Any  # statsmodels regression results
    coefficients: pd.DataFrame
    metrics: pd.DataFrame


def _require_statsmodels() -> None:
    if sm is None:
        raise ImportError(
            "statsmodels is required for OLS regression outputs (p-values, CI). "
            "Add 'statsmodels' to requirements and install it."
        )


def run_ols(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str | None = None,
    add_constant: bool = True,
    robust: Literal[None, "HC0", "HC1", "HC2", "HC3"] = None,
    alpha: float = 0.05,
) -> ols_outputs:
    """Fit OLS regression and return standardized outputs.

    Parameters:
    - add_constant: adds intercept term
    - robust: if provided, uses heteroscedasticity-robust covariance (HC0/1/2/3)
    - alpha: CI level (0.05 -> 95% CI)
    - model_name: optional label to attach to output DataFrames (useful when concatenating across specs)
    """
    _require_statsmodels()

    X_mat = X.copy()
    if add_constant:
        X_mat = sm.add_constant(X_mat, has_constant="add")

    model = sm.OLS(y, X_mat).fit()

    if robust is not None:
        model = model.get_robustcov_results(cov_type=robust)

    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues

    # Depending on statsmodels results class (e.g., robustcov),
    # params/bse/tvalues/pvalues may be numpy arrays without an index.
    # Use the design-matrix column names as the canonical predictor index.
    predictor_index = pd.Index(getattr(params, "index", X_mat.columns), name="predictor")

    ci_raw = model.conf_int(alpha=alpha)

    # statsmodels may return either a DataFrame or a numpy array depending on the results class (e.g., robustcov) and statsmodels version.
    if isinstance(ci_raw, pd.DataFrame):
        ci = ci_raw.copy()
        ci.columns = ["ci_low", "ci_high"]
    else:
        # numpy array with shape (k, 2)
        ci = pd.DataFrame(ci_raw, index=predictor_index, columns=["ci_low", "ci_high"])

    coef_df = pd.DataFrame(
        {
            "coef": np.asarray(params),
            "std_err": np.asarray(bse),
            "t": np.asarray(tvals),
            "p": np.asarray(pvals),
        },
        index=predictor_index,
    ).join(ci)
    coef_df = coef_df.reset_index()

    metrics_dict = {
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n_obs": float(model.nobs),
        "df_model": float(model.df_model),
    }

    metrics_df = pd.DataFrame([metrics_dict])

    if model_name is not None:
        # Add model label for easier concatenation/comparison downstream.
        coef_df.insert(0, "model", model_name)
        metrics_df.insert(0, "model", model_name)

    return ols_outputs(model=model, coefficients=coef_df, metrics=metrics_df)


def predict_ols(
    ols: ols_outputs,
    X: pd.DataFrame,
    *,
    add_constant: bool = True,
) -> np.ndarray:
    """Predict using a fitted ols_outputs model."""
    _require_statsmodels()

    X_mat = X.copy()
    if add_constant:
        X_mat = sm.add_constant(X_mat, has_constant="add")

    return np.asarray(ols.model.predict(X_mat))


def predict_ols_df(
    ols: ols_outputs,
    X: pd.DataFrame,
    y_true: pd.Series | None = None,
    *,
    add_constant: bool = True,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Predict and return a tidy DataFrame (optionally including ground truth).

    Columns:
    - pred
    - actual (optional)
    - residual (optional)
    - model (optional)
    """
    preds = predict_ols(ols, X, add_constant=add_constant)
    out = pd.DataFrame({"pred": preds}, index=X.index)

    if y_true is not None:
        out["actual"] = y_true.values
        out["residual"] = out["actual"] - out["pred"]

    if model_name is not None:
        out.insert(0, "model", model_name)

    return out.reset_index(drop=False)
