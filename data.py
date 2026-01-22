import numpy as np
import pandas as pd

# -----------------------------
# Outlier handling (IQR method)
# -----------------------------

def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the IQR rule.
    Returns a filtered copy of df.
    """
    if column not in df.columns:
        return df

    series = pd.to_numeric(df[column], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return df[(series >= lower) & (series <= upper)].copy()


def load_data(filepath='data/Exam_Score_Prediction.csv'):
    '''
    filepath must be a path to a csv file. It can be absolute or relative.
    '''
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal cleaning pipeline:
    - Maps ordinal columns
    - Drops missing values in core modeling columns
    - Creates log-transformed features
    - Applies IQR outlier removal
    """
    clean_df = df.copy()

    # Ordinal/Binary mapping (keep original text columns, add *_num columns)
    clean_df = sleep_quality_map(clean_df)
    clean_df = exam_difficulty_map(clean_df)
    clean_df = facility_rating_map(clean_df)
    clean_df = internet_access_map(clean_df)

    # Drop missing values for core variables
    core_cols = [
        'exam_score',
        'sleep_hours',
        'age',
        'sleep_quality_num'
    ]
    clean_df = clean_df.dropna(subset=core_cols)

    # Log transforms (stabilize variance)
    clean_df['sleep_time_log'] = log1_column(clean_df, 'sleep_hours')
    clean_df['age_log'] = log1_column(clean_df, 'age')
    clean_df['sleep_quality_log'] = np.log(clean_df['sleep_quality_num'])

    # Outlier removal (IQR)
    clean_df = remove_outliers_iqr(clean_df, 'exam_score')
    clean_df = remove_outliers_iqr(clean_df, 'sleep_hours')
    clean_df = remove_outliers_iqr(clean_df, 'age')

    return clean_df


def log1_column (df, string): #valid for numeral data only
    tmp_df = df.copy()
    tmp_df['new_log'] = np.log1p(df[string])
    return tmp_df['new_log']


def sleep_quality_map(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sleep_quality (ordinal text) into numeric sleep_quality_num (1..3)."""
    new_df = df.copy()
    if 'sleep_quality' not in new_df.columns:
        return new_df

    # normalize
    sq = (
        new_df['sleep_quality']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    quality_num_map = {'poor': 1, 'average': 2, 'good': 3}
    new_df['sleep_quality_num'] = sq.map(quality_num_map)
    return new_df


def exam_difficulty_map(df: pd.DataFrame) -> pd.DataFrame:
    """Convert exam_difficulty (easy/moderate/hard) into exam_difficulty_num (1..3)."""
    new_df = df.copy()
    if 'exam_difficulty' not in new_df.columns:
        return new_df

    ed = (
        new_df['exam_difficulty']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    difficulty_map = {'easy': 1, 'moderate': 2, 'hard': 3}
    new_df['exam_difficulty_num'] = ed.map(difficulty_map)
    return new_df


def facility_rating_map(df: pd.DataFrame) -> pd.DataFrame:
    """Convert facility_rating (low/medium/high) into facility_rating_num (1..3)."""
    new_df = df.copy()
    if 'facility_rating' not in new_df.columns:
        return new_df

    fr = (
        new_df['facility_rating']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    rating_map = {'low': 1, 'medium': 2, 'high': 3}
    new_df['facility_rating_num'] = fr.map(rating_map)
    return new_df


def internet_access_map(df: pd.DataFrame) -> pd.DataFrame:
    """Convert internet_access (yes/no) into internet_access_num (0/1)."""
    new_df = df.copy()
    if 'internet_access' not in new_df.columns:
        return new_df

    ia = (
        new_df['internet_access']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Treat common truthy strings as 1
    new_df['internet_access_num'] = ia.isin(['yes', 'y', 'true', '1']).astype(int)
    return new_df


# -----------------------------
# Universal binary encoding
# -----------------------------

def binary_map_column(df: pd.DataFrame, column: str, positive_values=None) -> pd.DataFrame:
    """
    Maps a string column to binary 0/1.
    positive_values: list of values that should map to 1 (default: ['yes', 'y', 'true', '1'])
    """
    new_df = df.copy()
    if positive_values is None:
        positive_values = ['yes', 'y', 'true', '1']

    new_df[column] = (
        new_df[column]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(positive_values)
        .astype(int)
    )

    return new_df
