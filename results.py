from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_metric_comparison(metrics_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Bar plot comparing R² / adj R² / RMSE / MAE across models."""
    dfm = metrics_df.copy()
    dfm = dfm.set_index("model")

    cols = [c for c in ["r2", "adj_r2", "rmse", "mae"] if c in dfm.columns]
    if not cols:
        return

    plt.figure(figsize=(9, 4.5))
    dfm[cols].plot(kind="bar")
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coef_compare(
    coef_df: pd.DataFrame,
    *,
    top_k: int,
    title: str,
    out_path: Path,
    model_order: list[str],
    top_from_model: str = "Full",
) -> None:
    """Compare coefficients (with 95% CI) for the top predictors from the specified model across specs."""
    c = coef_df.copy()

    # pick top predictors from specified model by smallest p-value (excluding intercept)
    full = c[(c["model"] == top_from_model) & (c["predictor"] != "const")].copy()
    if full.empty:
        return

    full["abs_t"] = full["t"].abs()
    top = (
        full.sort_values(["p", "abs_t"], ascending=[True, False])
        .head(top_k)["predictor"]
        .tolist()
    )

    c = c[c["predictor"].isin(top)].copy()
    if c.empty:
        return

    # y positions (same ordering as `top`)
    y_base = {pred: i for i, pred in enumerate(top)}

    # small offsets per model to avoid overlap
    offsets = {m: (j - (len(model_order) - 1) / 2) * 0.18 for j, m in enumerate(model_order)}

    plt.figure(figsize=(9, 0.45 * len(top) + 2))

    for m in model_order:
        sub = c[c["model"] == m].copy()
        if sub.empty:
            continue
        ys = np.array([y_base[p] for p in sub["predictor"].tolist()], dtype=float) + offsets[m]
        beta = sub["coef"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        plt.errorbar(beta, ys, xerr=[beta - lo, hi - beta], fmt="o", label=m)

    plt.axvline(0.0)
    plt.yticks(range(len(top)), top)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Coefficient (95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coef_forest(ols_out, title: str, out_path: Path, top_k: int = 15) -> None:
    """
    Forest plot of top coefficients by smallest p-value (excluding intercept).
    Uses coef + 95% CI. Good for "what affects exam score most".
    """
    coef = ols_out.coefficients.copy()
    coef = coef[coef["predictor"] != "const"].copy()

    # sort by p then by |t| (same logic as main)
    coef["abs_t"] = coef["t"].abs()
    coef = coef.sort_values(["p", "abs_t"], ascending=[True, False]).head(top_k)

    # y positions
    y_pos = np.arange(len(coef))[::-1]

    # values
    beta = coef["coef"].to_numpy()
    lo = coef["ci_low"].to_numpy()
    hi = coef["ci_high"].to_numpy()
    labels = coef["predictor"].tolist()

    plt.figure()
    plt.errorbar(beta, y_pos, xerr=[beta - lo, hi - beta], fmt="o")
    plt.axvline(0.0)
    plt.yticks(y_pos, labels)
    plt.title(title)
    plt.xlabel("Coefficient (95% CI)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    """
    Scatter: predicted vs actual. Supports "model captures signal".
    """
    y_true_arr = y_true.to_numpy(dtype=float)

    plt.figure()
    plt.scatter(y_pred, y_true_arr)

    # identity line
    mn = float(np.nanmin([y_pred.min(), y_true_arr.min()]))
    mx = float(np.nanmax([y_pred.max(), y_true_arr.max()]))
    plt.plot([mn, mx], [mn, mx])

    plt.title(title)
    plt.xlabel("Predicted exam score")
    plt.ylabel("Actual exam score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residuals_vs_fitted(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    """
    Residual plot: fitted vs residuals. Supports "nothing obviously broken".
    """
    y_true_arr = y_true.to_numpy(dtype=float)
    resid = y_true_arr - y_pred

    plt.figure()
    plt.scatter(y_pred, resid)
    plt.axhline(0.0)
    plt.title(title)
    plt.xlabel("Fitted (predicted) exam score")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()