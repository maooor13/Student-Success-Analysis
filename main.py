from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exam score regression pipeline.")
    parser.add_argument(
        "--csv",
        type=str,
        default="Exam_Score_Prediction.csv",
        help="Path to CSV dataset (default: Exam_Score_Prediction.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="exam_score",
        help="Target column name (default: exam_score)",
    )
    parser.add_argument(
        "--robust",
        type=str,
        default="HC3",
        choices=["HC0", "HC1", "HC2", "HC3", "none"],
        help="Robust standard errors (default: HC3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run extra diagnostics (VIF, residual summary). Off by default.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    # 1) Load + clean
    raw_df = load_data(str(csv_path))
    df = clean_data(raw_df)

    # 2) Infer roles (binary/ordinal/continuous)
    roles = infer_column_roles(df, target=args.target, max_ordinal_levels=7)

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

    robust = None if args.robust == "none" else args.robust

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

    # 6) Diagnostics on FULL model (most relevant)
    if args.diagnostics:
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