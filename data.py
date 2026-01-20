import numpy as np
import pandas as pd
from scipy import stats
from numpy.ma.extras import average


df = pd.read_csv('data/Exam_Score_Prediction.csv') # reading data file


def import_data():
    pass

def clean_data():
    pass


# --- Steiger's Z-Test Implementation ---
def calculate_steiger_z(r12, r13, r23, n):
    """
    Tests if the difference between dependent correlations r12 and r13 is significant.
    r12: corr(Target, Predictor A)
    r13: corr(Target, Predictor B)
    r23: corr(Predictor A, Predictor B)
    n:   Sample size
    """
    # 1. Fisher's z-transform
    z12 = np.arctanh(r12)
    z13 = np.arctanh(r13)
    
    # 2. Calculate Mean Correlation (r_bar)
    r_bar = (r12 + r13) / 2

    # 3. Calculate Factor 'f' (relationship between predictors vs outcome)
    # This adjusts for the fact that r12 and r13 are not independent
    f = (1 - r23) / (2 * (1 - r_bar**2))

    # 4. Calculate Factor 'h' (weighting factor for variance)
    h = (1 - (f * r_bar**2)) / (1 - r_bar**2)

    # 5. Standard Error of the difference
    se = np.sqrt((2 * (1 - r23) * h) / (n - 3))

    # 6. Z-score and P-value
    z_score = (z12 - z13) / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    return z_score, p_value

def log1_column (df, string): #valid for numeral data only
    tmp_df = df.copy()
    tmp_df['new_log'] = np.log1p(df[string])
    return tmp_df['new_log']

def sleep_quality_map (df):
    quality_num_map = {'poor': 1, 'average': 2, 'good': 3}  # creating a map to turn strings into numbers
    df["sleep_quality_num"] = df['sleep_quality'].map(quality_num_map)  # using the map to translate strings into numbers
    return df
