import numpy as np
import pandas as pd

# -----------------------------
# Column name normalization
# -----------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip, and replace spaces with underscores for all columns."""
    new_df = df.copy()
    new_df.columns = (
        new_df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return new_df


def _norm_str_series(s: pd.Series) -> pd.Series:
    """Normalize categorical strings for matching.

    Assumes `clean_data` has already removed missing values.
    """
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )


def enforce_numeric_like_columns(
    df: pd.DataFrame,
    *,
    min_numeric_ratio: float = 0.9,
    min_unique: int = 7,
    exclude: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Coerce pre-identified numeric-like columns to numeric dtype.

    This function performs coercion only. All decisions about whether a column
    should be treated as numeric are made in `clean_data`.

    Any values that fail coercion become NaN and are removed by `clean_data`.
    """
    new_df = df.copy()
    exclude_set = set(exclude or [])
    coerced_cols: list[str] = []

    for col in new_df.columns:
        if col in exclude_set:
            continue

        if pd.api.types.is_numeric_dtype(new_df[col]):
            continue

        if col.endswith("_num") or col.endswith("_log"):
            continue

        # Coerce to numeric; any failures become NaN and will be dropped in clean_data
        coerced = pd.to_numeric(new_df[col], errors="coerce")
        new_df[col] = coerced
        coerced_cols.append(col)

    return new_df, coerced_cols


# -----------------------------
# Universal categorical inference
# -----------------------------

def infer_categorical_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Infer binary/ordinal mappings for *any* object column by its value set.

    Rules:
    - Binary (yes/no-like) -> create `<col>_num` (0/1)
    - Ordinal sets -> create `<col>_num`
      * {poor, average, good} -> 1..3
      * {easy, moderate, hard} -> 1..3
      * {low, medium, high} -> 1..3

    Columns that do not match known sets are left as-is.
    """
    new_df = df.copy()

    yes_set = {"yes", "y", "true", "1"}
    no_set = {"no", "n", "false", "0"}

    ordinal_maps = {
        frozenset(["poor", "average", "good"]): {"poor": 1, "average": 2, "good": 3},
        frozenset(["easy", "moderate", "hard"]): {"easy": 1, "moderate": 2, "hard": 3},
        frozenset(["low", "medium", "high"]): {"low": 1, "medium": 2, "high": 3},
    }

    obj_cols = new_df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        values = _norm_str_series(new_df[col])
        uniq = set(values.unique().tolist())
        if not uniq:
            continue

        # Binary inference
        if uniq.issubset(yes_set | no_set):
            new_df[f"{col}_num"] = values.isin(yes_set).astype(int)
            new_df = new_df.drop(columns=[col])
            continue

        # Ordinal inference
        created_num = False
        for key, mapping in ordinal_maps.items():
            if uniq.issubset(key):
                new_df[f"{col}_num"] = values.map(mapping)
                created_num = True
                break
        if created_num:
            new_df = new_df.drop(columns=[col])

    return new_df


# -----------------------------
# Preprocessing report (metadata)
# -----------------------------

def _count_new_num_columns(before_cols: set[str], after_cols: set[str]) -> int:
    """Count how many *_num columns were introduced by mapping/inference."""
    added = after_cols - before_cols
    return sum(1 for c in added if c.endswith("_num"))


# -----------------------------
# Universal outlier handling
# -----------------------------

def apply_iqr_outlier_removal(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    k: float = 1.5,
    min_unique: int = 7
) -> pd.DataFrame:
    """Apply IQR outlier removal ONLY to continuous numeric columns.

    Rules:
    - Operates only on numeric columns
    - Skips columns with low cardinality (binary / ordinal scales)
    - A column is considered continuous if it has >= `min_unique` unique values
    - Removes a row if it is an outlier in ANY included continuous column
    """
    new_df = df.copy()
    exclude = set(exclude or [])

    # candidate numeric columns
    numeric_cols = [
        c for c in new_df.select_dtypes(include=["number"]).columns
        if c not in exclude
    ]

    # keep only continuous-like columns
    continuous_cols = []
    for col in numeric_cols:
        uniq = new_df[col].unique()
        if len(uniq) >= min_unique:
            continuous_cols.append(col)

    if not continuous_cols:
        return new_df

    mask = pd.Series(True, index=new_df.index)

    for col in continuous_cols:
        series = new_df[col].astype(float)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        mask &= series.between(lower, upper)

    return new_df.loc[mask].copy()


def log1p_series(s: pd.Series) -> pd.Series:
    """Safe log transform. Assumes s is numeric and >= 0."""
    return np.log1p(s)


def add_log_features(
    df: pd.DataFrame,
    *,
    min_unique: int = 7,
    skew_threshold: float = 1.0,
    exclude: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Create log1p-transformed features for eligible continuous numeric columns.

    Eligibility (per column):
    - numeric dtype (or coercible to numeric)
    - >= `min_unique` unique values (continuous-like)
    - non-negative values
    - right-skew above `skew_threshold`

    Returns:
    - (new_df, created_log_columns)
    """
    new_df = df.copy()
    exclude_set = set(exclude or [])

    created: list[str] = []
    numeric_cols = [c for c in new_df.select_dtypes(include=["number"]).columns if c not in exclude_set]

    for col in numeric_cols:
        s = new_df[col].astype(float)
        s_nonnull = s

        # continuous-like?
        if s_nonnull.nunique() < min_unique:
            continue

        # must be non-negative to use log1p safely
        if (s_nonnull < 0).any():
            continue

        # only apply if strongly right-skewed
        skew = s_nonnull.skew()
        if pd.isna(skew) or skew <= skew_threshold:
            continue

        new_col = f"{col}_log"
        new_df[new_col] = log1p_series(s)
        created.append(new_col)
        # Drop the source column once the reshaped/log feature is created
        if col in new_df.columns:
            new_df = new_df.drop(columns=[col])

    return new_df, created


def load_data(filepath='data/Exam_Score_Prediction.csv'):
    '''
    filepath must be a path to a csv file. It can be absolute or relative.
    '''
    df = pd.read_csv(filepath)
    return standardize_column_names(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = standardize_column_names(df)

    report = {"rows_in": int(clean_df.shape[0])}
    report["remaining_indices"] = set(clean_df.index)

    # 1) First protective layer: normalize common missing tokens and drop missing rows
    obj_cols = clean_df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        # strip whitespace
        clean_df[obj_cols] = clean_df[obj_cols].apply(lambda s: s.astype(str).str.strip())
        # convert common missing tokens to real NaN so dropna removes them
        missing_tokens = {"", "nan", "na", "n/a", "none", "null"}
        clean_df[obj_cols] = clean_df[obj_cols].apply(
            lambda s: s.mask(s.str.lower().isin(missing_tokens), np.nan)
        )

    clean_df = clean_df.dropna()
    prev_idx = report["remaining_indices"]
    curr_idx = set(clean_df.index)
    report["dropped_rows_missing_initial"] = sorted(prev_idx - curr_idx)
    report["remaining_indices"] = curr_idx
    report["rows_after_dropna_initial"] = int(clean_df.shape[0])

    # 2) Infer binary/ordinal mappings for object columns (adds *_num)
    before_cols = set(clean_df.columns)
    clean_df = infer_categorical_mappings(clean_df)
    after_cols = set(clean_df.columns)
    report["num_columns_created"] = _count_new_num_columns(before_cols, after_cols)
    report["rows_after_mapping"] = int(clean_df.shape[0])

    # 3) Detect numeric-like object columns and coerce them
    coerced_cols = []
    for col in clean_df.select_dtypes(include=["object"]).columns:
        s = pd.to_numeric(clean_df[col], errors="coerce")
        numeric_ratio = s.notna().mean()

        if numeric_ratio < 0.9:
            continue

        if s.nunique() < 7:
            continue

        # Skip identifier-like columns by name
        col_l = col.lower()
        if any(k in col_l for k in ("id", "uuid", "guid", "email", "phone", "passport", "ssn")):
            continue

        clean_df[col] = s
        coerced_cols.append(col)

    report["numeric_columns_coerced"] = coerced_cols

    # 3b) Second protective layer: coercion may introduce NaNs; drop them here to keep invariant
    clean_df = clean_df.dropna()
    prev_idx = report["remaining_indices"]
    curr_idx = set(clean_df.index)
    report["dropped_rows_missing_after_coercion"] = sorted(prev_idx - curr_idx)
    report["remaining_indices"] = curr_idx
    report["rows_after_dropna_after_coercion"] = int(clean_df.shape[0])

    # 4) Rule-based log features for eligible skewed continuous numeric columns
    clean_df, created_logs = add_log_features(
        clean_df,
        min_unique=7,
        skew_threshold=1.0,
        exclude=["exam_score"],
    )
    report["log_columns_created"] = created_logs

    # 5) Global IQR outlier removal across continuous numeric columns only
    clean_df = apply_iqr_outlier_removal(
        clean_df,
        exclude=[],
        k=1.5,
        min_unique=7,
    )
    prev_idx = report["remaining_indices"]
    curr_idx = set(clean_df.index)
    report["dropped_rows_outliers"] = sorted(prev_idx - curr_idx)
    report["remaining_indices"] = curr_idx
    report["rows_after_outliers"] = int(clean_df.shape[0])

    # Attach report to dataframe metadata (does not change return type)
    report.pop("remaining_indices", None)
    clean_df.attrs["preprocess_report"] = report

    return clean_df


def infer_column_roles(
    df: pd.DataFrame,
    *,
    target: str = "exam_score",
    max_ordinal_levels: int = 7,
) -> dict[str, list[str] | str]:
 
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    roles: dict[str, list[str] | str] = {
        "target": target,
        "binary": [],
        "ordinal": [],
        "continuous": [],
    }

    for col in df.columns:
        if col == target:
            continue

        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        nunique = s.nunique()

        if nunique == 2:
            roles["binary"].append(col)
        elif 3 <= nunique <= max_ordinal_levels:
            roles["ordinal"].append(col)
        elif nunique > max_ordinal_levels:
            roles["continuous"].append(col)

    return roles
