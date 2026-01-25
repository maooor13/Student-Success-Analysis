from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_data, clean_data
from design import build_baseline_xy, build_behavior_xy, build_full_xy
from ols_engine import run_ols
from diagnostics import compute_vif, residual_summary


def infer_roles(df: pd.DataFrame, *, target: str = "exam_score") -> dict[str, object]:
    """
    Infer roles dict expected by design.py:
    roles = {
        "target": "exam_score",
        "binary": [...],
        "ordinal": [...],
        "continuous": [...],
    }

    Heuristics:
    - binary: numeric and unique values subset of {0,1}
    - ordinal: numeric with low cardinality OR name endswith "_num"
    - continuous: remaining numeric predictors
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in df columns")

    roles: dict[str, object] = {"target": target, "binary": [], "ordinal": [], "continuous": []}

    # only numeric predictors; design expects already-encoded columns
    # Exclude identifier-like columns (IDs are not meaningful predictors)
    exclude_cols = {"student_id"}
    num_cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if c != target and c not in exclude_cols and not c.lower().endswith("_id")
    ]

    binary: list[str] = []
    ordinal: list[str] = []
    continuous: list[str] = []

    for c in num_cols:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        uniq = set(s.dropna().unique().tolist())

        # binary
        if uniq.issubset({0, 1}) and nunique <= 2:
            binary.append(c)
            continue

        # ordinal (explicit *_num OR low-cardinality numeric)
        if c.endswith("_num") or nunique <= 10:
            ordinal.append(c)
            continue

        # continuous
        continuous.append(c)

    roles["binary"] = binary
    roles["ordinal"] = ordinal
    roles["continuous"] = continuous
    return roles


def print_model(label: str, ols_out) -> None:
    print("\n" + "=" * 80)
    print(f"{label}")
    print("=" * 80)

    m = ols_out.metrics
    print(f"R²={m['r2']:.4f} | adj R²={m['adj_r2']:.4f} | n={int(m['n_obs'])} | df_model={int(m['df_model'])}")

    # sort by p-value, then by |t|
    coef = ols_out.coefficients.copy()
    coef["abs_t"] = coef["t"].abs()
    coef = coef.sort_values(["p", "abs_t"], ascending=[True, False])

    # show top 15 (excluding intercept)
    view = coef[coef["predictor"] != "const"].head(15)
    print("\nTop coefficients (lowest p-values):")
    print(view[["predictor", "coef", "std_err", "t", "p", "ci_low", "ci_high"]].to_string(index=False))


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
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    # 1) Load + clean
    raw_df = load_data(str(csv_path))
    df = clean_data(raw_df)

    # 2) Infer roles (binary/ordinal/continuous)
    roles = infer_roles(df, target=args.target)

    print("\nData summary:")
    print(f"Rows after cleaning: {len(df)}")
    print("Role counts:",
          f"binary={len(roles['binary'])}, ordinal={len(roles['ordinal'])}, continuous={len(roles['continuous'])}")
    print("Target:", roles["target"])

    robust = None if args.robust == "none" else args.robust

    # 3) Build X/y for each model
    X_base, y = build_baseline_xy(df, roles)
    X_beh, y2 = build_behavior_xy(df, roles)   # default includes exam_difficulty_num if present
    X_full, y3 = build_full_xy(df, roles)

    # sanity: same y index
    assert y.index.equals(y2.index) and y.index.equals(y3.index)

    # 4) Fit OLS models
    ols_base = run_ols(X_base, y, robust=robust)
    ols_beh = run_ols(X_beh, y, robust=robust)
    ols_full = run_ols(X_full, y, robust=robust)

    # 5) Print results
    print_model("Baseline model (binary + ordinal)", ols_base)
    print_model("Behavior model (continuous + default controls)", ols_beh)
    print_model("Full model (binary + ordinal + continuous)", ols_full)

    # 6) Diagnostics on FULL model (most relevant)
    print("\n" + "-" * 80)
    print("Diagnostics (FULL model)")
    print("-" * 80)

    try:
        vif = compute_vif(X_full)
        print("\nTop VIFs (largest first):")
        print(vif.sort_values("vif", ascending=False).head(15).to_string(index=False))
    except Exception as e:
        print(f"\nVIF skipped (reason: {e})")

    try:
        resid = residual_summary(ols_full)
        print("\nResidual summary:")
        for k, v in resid.items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"\nResidual summary skipped (reason: {e})")

    # 7) Model comparison (key story)
    print("\n" + "-" * 80)
    print("Model comparison")
    print("-" * 80)
    print(f"Baseline R²: {ols_base.metrics['r2']:.4f}")
    print(f"Behavior  R²: {ols_beh.metrics['r2']:.4f}")
    print(f"Full      R²: {ols_full.metrics['r2']:.4f}")
    print("\nΔR² (Full - Baseline):", f"{(ols_full.metrics['r2'] - ols_base.metrics['r2']):.4f}")
    print("ΔR² (Full - Behavior):", f"{(ols_full.metrics['r2'] - ols_beh.metrics['r2']):.4f}")


if __name__ == "__main__":
    main()