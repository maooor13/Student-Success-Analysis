from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from diagnostics import compute_vif, residual_summary

logger = logging.getLogger(__name__)


def extract_model_metrics(ols_out: Any) -> dict[str, float]:
    """Return core scalar metrics from an OLS output in a flat dict."""
    m = ols_out.metrics.iloc[0]
    return {
        "r2": float(m.get("r2", float("nan"))),
        "adj_r2": float(m.get("adj_r2", float("nan"))),
        "n_obs": int(m.get("n_obs", 0)),
        "df_model": int(m.get("df_model", 0)),
    }


def extract_coefficients(ols_out: Any) -> pd.DataFrame:
    """Return a cleaned coefficients DataFrame with numeric columns enforced."""
    coef = ols_out.coefficients.copy()

    for col in ["coef", "std_err", "t", "p", "ci_low", "ci_high"]:
        if col in coef.columns:
            coef[col] = pd.to_numeric(coef[col], errors="coerce")

    return coef


def log_model(label: str, ols_out: Any, *, top_n: int = 15) -> None:
    logger.info("%s", "\n" + "=" * 80)
    logger.info("%s", label)
    logger.info("%s", "=" * 80)

    m = extract_model_metrics(ols_out)
    logger.info(
        "R²=%.4f | adj R²=%.4f | n=%d | df_model=%d",
        m["r2"], m["adj_r2"], m["n_obs"], m["df_model"],
    )

    # Show top coefficients only in DEBUG 
    if not logger.isEnabledFor(logging.DEBUG):
        return

    coef = extract_coefficients(ols_out)

    if "t" in coef.columns:
        coef["abs_t"] = pd.to_numeric(coef["t"], errors="coerce").abs()
    else:
        coef["abs_t"] = float("nan")

    if "p" in coef.columns:
        coef["p"] = pd.to_numeric(coef["p"], errors="coerce")

    coef = coef.sort_values(["p", "abs_t"], ascending=[True, False])

    if "predictor" in coef.columns:
        view = coef[coef["predictor"] != "const"].head(int(top_n))
    else:
        view = coef.head(int(top_n))

    cols = [c for c in ["predictor", "coef", "std_err", "t", "p", "ci_low", "ci_high"] if c in view.columns]
    logger.debug("Top coefficients (lowest p-values):\n%s", view[cols].to_string(index=False))


def log_full_model_diagnostics(X_full: pd.DataFrame, ols_full: Any, *, top_vif_n: int = 15) -> None:
    logger.info("%s", "\n" + "-" * 80)
    logger.info("Diagnostics (FULL model)")
    logger.info("%s", "-" * 80)

    try:
        vif = compute_vif(X_full)
        logger.info(
            "Top VIFs (largest first):\n%s",
            vif.sort_values("vif", ascending=False).head(int(top_vif_n)).to_string(index=False),
        )
    except Exception as e:
        logger.warning("VIF skipped (reason: %s)", e)

    try:
        resid = residual_summary(ols_full)
        logger.info("Residual summary:")
        for k, v in resid.items():
            logger.info("%s: %.4f", k, float(v))
    except Exception as e:
        logger.warning("Residual summary skipped (reason: %s)", e)


def log_model_comparison(ols_base: Any, ols_beh: Any, ols_full: Any) -> None:
    logger.info("%s", "\n" + "-" * 80)
    logger.info("Model comparison")
    logger.info("%s", "-" * 80)

    base = extract_model_metrics(ols_base)
    beh = extract_model_metrics(ols_beh)
    full = extract_model_metrics(ols_full)

    logger.info("Baseline R²: %.4f", base["r2"])
    logger.info("Behavior  R²: %.4f", beh["r2"])
    logger.info("Full      R²: %.4f", full["r2"])
    logger.info("ΔR² (Full - Baseline): %.4f", full["r2"] - base["r2"])
    logger.info("ΔR² (Full - Behavior): %.4f", full["r2"] - beh["r2"])