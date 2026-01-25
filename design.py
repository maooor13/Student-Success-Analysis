#building separate baseline and behavior matrixes with predictors and target vector and matrix mixing both  
from __future__ import annotations

from typing import Any
import pandas as pd


def _get_target(roles: dict[str, Any], target: str | None) -> str:
    if target is not None:
        return target
    t = roles.get("target")
    if not isinstance(t, str) or not t:
        raise ValueError("roles must contain a non-empty 'target' string")
    return t


def build_baseline_xy(
    df: pd.DataFrame,
    roles: dict[str, Any],
    *,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Baseline model: structural factors only (binary + ordinal)."""
    y_col = _get_target(roles, target)
    y = df[y_col]
    cols = list(roles.get("binary", [])) + list(roles.get("ordinal", []))
    X = df[cols]
    return X, y


def build_behavior_xy(
    df: pd.DataFrame,
    roles: dict[str, Any],
    *,
    target: str | None = None,
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Behavior model: continuous predictors + optional controls.

    A reasonable default control is exam difficulty if present.
    """
    y_col = _get_target(roles, target)
    y = df[y_col]

    base_cols = list(roles.get("continuous", []))

    if controls is None:
        controls = []
        if "exam_difficulty_num" in df.columns and "exam_difficulty_num" not in base_cols:
            controls.append("exam_difficulty_num")

    cols = base_cols + list(controls)
    X = df[cols]
    return X, y


def build_full_xy(
    df: pd.DataFrame,
    roles: dict[str, Any],
    *,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Full adjusted model: binary + ordinal + continuous."""
    y_col = _get_target(roles, target)
    y = df[y_col]
    cols = (
        list(roles.get("binary", []))
        + list(roles.get("ordinal", []))
        + list(roles.get("continuous", []))
    )
    X = df[cols]
    return X, y