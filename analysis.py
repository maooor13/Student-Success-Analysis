'''
Aging sleeping time connection
Aging  sleeping quality connection
'''
import numpy as np
import pandas as pd
from scipy import stats


df = pd.read_csv('data/Exam_Score_Prediction.csv') # reading data file

def sleeping_time_age_correlation(df):
    df['sleep_time_log'] = np.log1p(df['sleep_hours']) #creating log values - for accuracy and stability of correlation
    df['age_log'] = np.log1p(df['age']) #creating log values - for accuracy and stability of correlation
    corr = df['sleep_time_log'].corr(df['age_log']) #calculating correlation between values
    '''
    close to 1 - sleeping more as aging, close to 0 - no connection, close to (-1) - sleeping less as aging
    '''
    return corr

def age_sleep_quality_correlation(df):
    df['age_log'] = np.log1p(df['age']) #creating log values - for accuracy and stability of correlation
    quality_num_map = {'poor' : 1, 'average': 2, 'good' : 3 } #creating a map to turn strings into numbers
    df["sleep_quality_num"] = df['sleep_quality'].map(quality_num_map) #using the map to translate strings into numbers
    df['sleep_quality_log'] = np.log(df['sleep_quality_num'])  #creating log values (using regular log, no possible 0 value)
    corr = df['age_log'].corr(df['sleep_quality_log']) #calculating correlation between values
    '''
    close to 1 - sleeping better as aging, close to 0 - no connection, close to (-1) - sleeping worse as aging
    '''
    return corr

'''
Is there a better correlation between
   sleep hours   and exam score 
or sleep quality and exam score?
'''
def sleep_hours_exam_score_correlation(df):
    return df["sleep_hours"].corr(df["exam_score"])


def sleep_quality_exam_score_correlation(df):
    return df["sleep_quality_num"].corr(df["exam_score"])


def comapre_sleep_quality_and_hours_with_score_correlation(quality_score_corr, hours_score_corr):
    pass 

# --- Step 1: Data Preparation ---
# Map the ordinal "sleep_quality" to a numeric scale (0, 1, 2)
quality_map = {"poor": 0, "average": 1, "good": 2}
df["sleep_quality_num"] = df["sleep_quality"].map(quality_map)

# --- Step 2: Calculate Correlations ---
# n: Sample size
n = 20000  # Or len(df)

# r12: Correlation between Sleep Hours and Exam Score
r12 = df["sleep_hours"].corr(df["exam_score"], method="spearman")

# r13: Correlation between Sleep Quality (numeric) and Exam Score
r13 = df["sleep_quality_num"].corr(df["exam_score"], method="spearman")

# r23: Correlation between Sleep Hours and Sleep Quality
# (Crucial for the test to account for the overlap between predictors)
r23 = df["sleep_hours"].corr(df["sleep_quality_num"], method="spearman")

# --- Step 3: Steiger's Z-Test Implementation ---
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

# --- Step 4: Run and Interpret ---
z_stat, p_val = calculate_steiger_z(r12, r13, r23, n)

print(f"Correlation (Hours vs Score):   {r12:.4f}")
print(f"Correlation (Quality vs Score): {r13:.4f}")
print(f"Correlation (Hours vs Quality): {r23:.4f}")
print("-" * 30)
print(f"Steiger's Z-Score: {z_stat:.4f}")
print(f"P-value:           {p_val:.4g}") # .4g handles very small scientific notation

print("-" * 30)
if p_val < 0.05:
    winner = "Sleep Hours" if abs(r12) > abs(r13) else "Sleep Quality"
    print(f"CONCLUSION: Significant difference found.\nThe stronger predictor is {winner}.")
else:
    print("CONCLUSION: No significant difference.\nBoth variables predict the exam score equally well.")

print(sleep_hours_exam_score_correlation(df))
print(sleep_quality_exam_score_correlation(df))