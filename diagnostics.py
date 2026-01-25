"""Diagnostics + target-centric scan helpers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from design import _get_target

# Optional deps: wrap them so the project can fail loudly with a clear message.
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
except Exception:  # pragma: no cover
    sm = None
    variance_inflation_factor = None

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


def _require_statsmodels() -> None:
    if sm is None:
        raise ImportError(
            "statsmodels is required for diagnostics (VIF). "
            "Add 'statsmodels' to requirements and install it."
        )


def compute_vif(
    X: pd.DataFrame,
    *,
    add_constant: bool = True,
    drop_constant_row: bool = True,
) -> pd.DataFrame:
    """Compute VIF for each predictor.

    Notes:
    - VIF is undefined for a constant column; if `add_constant` is True we add
      a constant but optionally remove it from the returned table.
    """
    _require_statsmodels()
    if variance_inflation_factor is None:
        raise ImportError("statsmodels variance_inflation_factor is unavailable")

    X_mat = X.copy()
    if add_constant:
        X_mat = sm.add_constant(X_mat, has_constant="add")

    values = X_mat.to_numpy(dtype=float)
    cols = list(X_mat.columns)

    vifs: list[float] = []
    for i in range(values.shape[1]):
        vifs.append(float(variance_inflation_factor(values, i)))

    out = pd.DataFrame({"predictor": cols, "vif": vifs})

    if drop_constant_row:
        out = out[out["predictor"] != "const"].reset_index(drop=True)

    return out


def residual_summary(ols: Any) -> dict[str, float]:
    """Return a lightweight residual summary for quick sanity checks.

    Accepts `ols_outputs` (from ols_engine) or any object with `.model.resid`.
    """
    resid = np.asarray(ols.model.resid)
    return {
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid, ddof=1)) if resid.size > 1 else float("nan"),
        "resid_min": float(np.min(resid)),
        "resid_max": float(np.max(resid)),
    }


def scan_predictors_vs_target(
    df: pd.DataFrame,
    roles: dict[str, Any],
    *,
    target: str | None = None,
    corr: Literal["pearson", "spearman"] = "pearson",
    include_p_values: bool = True,
) -> pd.DataFrame:
    """Create a tidy, one-row-per-predictor scan table (EDA support).

    Binary predictors:
    - effect = mean(y|x=1) - mean(y|x=0)
    - p_value = Welch t-test p-value (if scipy available)

    Ordinal/continuous predictors:
    - effect = correlation coefficient (Pearson or Spearman)
    - p_value = correlation p-value (if scipy available)
    """
    y_col = _get_target(roles, target)
    y = df[y_col]

    rows: list[dict[str, Any]] = []

    def _direction(val: float) -> str:
        if np.isnan(val):
            return "unknown"
        return "positive" if val > 0 else ("negative" if val < 0 else "zero")

    # Binary: mean difference + Welch t-test
    for col in list(roles.get("binary", [])):
        x = df[col]
        y0 = y[x == 0]
        y1 = y[x == 1]

        mean0 = float(y0.mean())
        mean1 = float(y1.mean())
        diff = mean1 - mean0

        p_val = float("nan")
        if include_p_values and stats is not None:
            t_res = stats.ttest_ind(y1, y0, equal_var=False)
            p_val = float(t_res.pvalue)

        rows.append(
            {
                "predictor": col,
                "type": "binary",
                "n": int(len(y)),
                "method": "t_test",
                "effect": float(diff),
                "p_value": p_val,
                "direction": _direction(float(diff)),
                "mean_0": mean0,
                "mean_1": mean1,
            }
        )

    # Ordinal/continuous: correlation
    ordinal_set = set(roles.get("ordinal", []))
    for col in list(roles.get("ordinal", [])) + list(roles.get("continuous", [])):
        x = df[col]
        r_val = float("nan")
        p_val = float("nan")

        if stats is not None:
            if corr == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)
            r_val = float(r)
            p_val = float(p) if include_p_values else float("nan")
        else:
            r_val = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
            p_val = float("nan")

        rows.append(
            {
                "predictor": col,
                "type": "ordinal" if col in ordinal_set else "continuous",
                "n": int(len(y)),
                "method": f"{corr}_r",
                "effect": r_val,
                "p_value": p_val,
                "direction": _direction(r_val),
                "mean_0": float("nan"),
                "mean_1": float("nan"),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out[
            [
                "predictor",
                "type",
                "n",
                "method",
                "effect",
                "p_value",
                "direction",
                "mean_0",
                "mean_1",
            ]
        ]

    return out