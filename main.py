from __future__ import annotations

import logging

from conf import Conf
from data import load_data, clean_data, infer_column_roles
from design import build_baseline_xy, build_behavior_xy, build_full_xy
from pathlib import Path
from ols_engine import run_ols, predict_ols
from reporting import log_model, log_full_model_diagnostics, log_model_comparison

from results import (
    ensure_out_dir,
    plot_coef_forest,
    plot_actual_vs_predicted,
    plot_residuals_vs_fitted,
)

logger = logging.getLogger(__name__)


def main():
    CONF = Conf()

    logging.basicConfig(
        level=getattr(logging, CONF.LOG_LEVEL),
        format="%(levelname)s: %(message)s",
    )

    try:
        raw_df = load_data(CONF.CSV)
    except Exception as e:
        logger.error(f"Error loading data file {CONF.CSV}")
        logger.debug("Load error:", exc_info=e)
        return

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
    logger.debug(f"Rows after cleaning: {len(df)}")
    logger.info(
        f"Role counts: binary={len(roles["binary"])}, ordinal={len(roles["ordinal"])}, continuous={len(roles["continuous"])}",
    )
    logger.info(f"Target: {roles["target"]}")

    # 3) Build X/y for each model
    X_base, y = build_baseline_xy(df, roles)
    X_behav, _ = build_behavior_xy(df, roles)   # default includes exam_difficulty_num if present
    X_full, _ = build_full_xy(df, roles)

    # 4) Fit OLS models
    ols_base = run_ols(X_base, y, robust=CONF.OSL_ROBUST)
    ols_beh = run_ols(X_behav, y, robust=CONF.OSL_ROBUST)
    ols_full = run_ols(X_full, y, robust=CONF.OSL_ROBUST)

    # 5) Print results
    if logger.isEnabledFor(logging.DEBUG):
        log_model("Baseline model (binary + ordinal)", ols_base)
        log_model("Behavior model (continuous + default controls)", ols_beh)

    log_model("Full model (binary + ordinal + continuous)", ols_full)

    # Diagnostics + model comparison
    log_full_model_diagnostics(X_full, ols_full)
    log_model_comparison(ols_base, ols_beh, ols_full)


    # Visual outputs (FULL model)
    out_dir = Path(getattr(CONF, "OUT_DIR", "outputs"))
    ensure_out_dir(out_dir)

    y_pred_full = predict_ols(ols_full, X_full)
    robust_label = CONF.OSL_ROBUST or "non-robust"

    plot_coef_forest(
        ols_full,
        title=f"Top 15 coefficients (FULL model) — {robust_label}",
        out_path=out_dir / "coef_forest_full.png",
        top_k=15,
    )

    plot_actual_vs_predicted(
        y_true=y,
        y_pred=y_pred_full,
        title="Actual vs Predicted (FULL model)",
        out_path=out_dir / "actual_vs_pred_full.png",
    )

    plot_residuals_vs_fitted(
        y_true=y,
        y_pred=y_pred_full,
        title="Residuals vs Fitted (FULL model)",
        out_path=out_dir / "residuals_vs_fitted_full.png",
    )

    logger.info(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
