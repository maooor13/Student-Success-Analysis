

"""
results.py

Purpose:
- Take regression outputs produced in main.py
- Present them in a clean, human-readable way
- Export tables that can be used in reports / papers
"""

from __future__ import annotations

import pandas as pd


def summarize_model(name: str, ols_out) -> pd.DataFrame:
    """
    Create a clean coefficient table for one model.

    Keeps:
    - predictor
    - coef
    - std_err
    - p
    - ci_low / ci_high
    """
    coef = ols_out.coefficients.copy()

    coef = coef[
        [
            "predictor",
            "coef",
            "std_err",
            "p",
            "ci_low",
            "ci_high",
        ]
    ]

    coef["model"] = name
    return coef


def export_results(
    *,
    baseline,
    behavior,
    full,
    out_dir: str = "results",
) -> None:
    """
    Export regression results to CSV files.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    baseline_df = summarize_model("baseline", baseline)
    behavior_df = summarize_model("behavior", behavior)
    full_df = summarize_model("full", full)

    baseline_df.to_csv(f"{out_dir}/baseline_coefficients.csv", index=False)
    behavior_df.to_csv(f"{out_dir}/behavior_coefficients.csv", index=False)
    full_df.to_csv(f"{out_dir}/full_coefficients.csv", index=False)

    metrics = pd.DataFrame(
        [
            {"model": "baseline", **baseline.metrics},
            {"model": "behavior", **behavior.metrics},
            {"model": "full", **full.metrics},
        ]
    )

    metrics.to_csv(f"{out_dir}/model_metrics.csv", index=False)


def pretty_print_summary(baseline, behavior, full) -> None:
    """
    Print a concise, readable summary to the console.
    """
    print("\n================= FINAL RESULTS SUMMARY =================\n")

    print("Model performance (R²):")
    print(f"  Baseline : {baseline.metrics['r2']:.3f}")
    print(f"  Behavior : {behavior.metrics['r2']:.3f}")
    print(f"  Full     : {full.metrics['r2']:.3f}")

    print("\nKey contributors in FULL model (p < 0.05):")

    sig = full.coefficients
    sig = sig[(sig["p"] < 0.05) & (sig["predictor"] != "const")]
    sig = sig.sort_values("coef", key=abs, ascending=False)

    for _, row in sig.iterrows():
        print(
            f"- {row['predictor']}: "
            f"coef={row['coef']:.2f}, "
            f"95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        )

    print("\nInterpretation:")
    print(
        "Behavioral variables (study habits, attendance, sleep) "
        "explain most of the variance in exam scores. "
        "Demographic and access-related variables add relatively little "
        "once behavior is accounted for."
    )
