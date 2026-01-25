from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conf import Conf
from data import load_data, clean_data
from design import build_baseline_xy, build_behavior_xy, build_full_xy
from ols_engine import run_ols
from diagnostics import compute_vif, residual_summary
from data import infer_column_roles

logger = logging.getLogger(__name__)


def log_model(label: str, ols_out) -> None:
    logger.info("%s", "\n" + "=" * 80)
    logger.info("%s", label)
    logger.info("%s", "=" * 80)

    m = ols_out.metrics
    logger.info(
        "R²=%.4f | adj R²=%.4f | n=%d | df_model=%d",
        m["r2"],
        m["adj_r2"],
        int(m["n_obs"]),
        int(m["df_model"]),
    )

    # sort by p-value, then by |t|
    coef = ols_out.coefficients.copy()
    coef["abs_t"] = coef["t"].abs()
    coef = coef.sort_values(["p", "abs_t"], ascending=[True, False])

    # show top 15 (excluding intercept)
    view = coef[coef["predictor"] != "const"].head(15)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Top coefficients (lowest p-values):\n%s",
            view[["predictor", "coef", "std_err", "t", "p", "ci_low", "ci_high"]].to_string(index=False),
        )


def main():
    CONF = Conf()

    logging.basicConfig(
        level=getattr(logging, CONF.LOG_LEVEL),
        format="%(levelname)s: %(message)s",
    )

    try:
        raw_df = load_data(CONF.CSV)
    except Exception as e:
        logger.error(f"Error loading data file {CONF.CSV}.")
        logger.debug(e)
        exit()

    df = clean_data(raw_df)

    roles = infer_column_roles(df, target=CONF.TARGET, max_ordinal_levels=7)

    # Analysis policy: exclude identifier-like columns from predictors
    exclude_cols = {"student_id"}
    for k in ("binary", "ordinal", "continuous"):
        roles[k] = [
            c
            for c in roles[k]  # type: ignore[index]
            if c not in exclude_cols and not str(c).lower().endswith("_id")
        ]

    logger.info("Data summary:")
    logger.info("Rows after cleaning: %d", len(df))
    logger.info(
        "Role counts: binary=%d, ordinal=%d, continuous=%d",
        len(roles["binary"]),
        len(roles["ordinal"]),
        len(roles["continuous"]),
    )
    logger.info("Target: %s", roles["target"])

    robust = None if CONF.OSL_ROBUST == "none" else CONF.OSL_ROBUST

    # 3) Build X/y for each model
    X_base, y = build_baseline_xy(df, roles)
    X_behav, y2 = build_behavior_xy(df, roles)   # default includes exam_difficulty_num if present
    X_full, y3 = build_full_xy(df, roles)

    # sanity: same y index
    assert y.index.equals(y2.index) and y.index.equals(y3.index)

    # 4) Fit OLS models
    ols_base = run_ols(X_base, y, robust=robust)
    ols_beh = run_ols(X_behav, y, robust=robust)
    ols_full = run_ols(X_full, y, robust=robust)

    # 5) Print results
    if logger.isEnabledFor(logging.DEBUG):
        log_model("Baseline model (binary + ordinal)", ols_base)
        log_model("Behavior model (continuous + default controls)", ols_beh)

    log_model("Full model (binary + ordinal + continuous)", ols_full)

# Diagnostic part
    logger.info("%s", "\n" + "-" * 80)
    logger.info("Diagnostics (FULL model)")
    logger.info("%s", "-" * 80)

    try:
        vif = compute_vif(X_full)
        logger.info("Top VIFs (largest first):\n%s", vif.sort_values("vif", ascending=False).head(15).to_string(index=False))
    except Exception as e:
        logger.warning("VIF skipped (reason: %s)", e)

    try:
        resid = residual_summary(ols_full)
        logger.info("Residual summary:")
        for k, v in resid.items():
            logger.info("%s: %.4f", k, v)
    except Exception as e:
        logger.warning("Residual summary skipped (reason: %s)", e)

    # 7) Model comparison (key story)
    logger.info("%s", "\n" + "-" * 80)
    logger.info("Model comparison")
    logger.info("%s", "-" * 80)
    logger.info("Baseline R²: %.4f", ols_base.metrics["r2"])
    logger.info("Behavior  R²: %.4f", ols_beh.metrics["r2"])
    logger.info("Full      R²: %.4f", ols_full.metrics["r2"])
    logger.info("ΔR² (Full - Baseline): %.4f", (ols_full.metrics["r2"] - ols_base.metrics["r2"]))
    logger.info("ΔR² (Full - Behavior): %.4f", (ols_full.metrics["r2"] - ols_beh.metrics["r2"]))


if __name__ == "__main__":
    main()