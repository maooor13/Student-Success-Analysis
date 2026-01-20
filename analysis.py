'''
Aging sleeping time connection
Aging  sleeping quality connection
'''
import numpy as np
import pandas as pd
from scipy import stats

from data import calculate_steiger_z


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
def sleep_hours_exam_score_correlation(df: pd.DataFrame):
    return df["sleep_hours"].corr(df["exam_score"], method="spearman")


def sleep_quality_exam_score_correlation(df: pd.DataFrame):
    return df["sleep_quality_num"].corr(df["exam_score"], method="spearman")


def sleep_quality_sleep_hours_correlation(df: pd.DataFrame):
    return df["sleep_hours"].corr(df["sleep_quality_num"], method="spearman")
 

def compare_sleep_quality_and_hours_with_score(df: pd.DataFrame):
    """
    Docstring for compare_sleep_quality_and_hours_with_score
    Compares correlation between 3 variables:
    - Sleep Quality
    - Sleep Hours
    - Exam Score

    :param df: Description
    :type df: pd.DataFrame
    """
    sample_size = len(df["exam_score"])
    hours_score_corr = sleep_hours_exam_score_correlation(df)
    quality_score_corr = sleep_quality_exam_score_correlation(df)
    quality_hours_corr = sleep_quality_sleep_hours_correlation(df)
    z_score, p_value = calculate_steiger_z(hours_score_corr, quality_score_corr, quality_hours_corr, sample_size)
    print(f"Sleep Correlation (Hours vs Score):   {hours_score_corr}")
    print(f"Sleep Correlation (Quality vs Score): {quality_score_corr}")
    print(f"Sleep Correlation (Hours vs Quality): {quality_hours_corr}")
    if p_value < 0.05:
        winner = "Sleep Hours" if abs(hours_score_corr) > abs(quality_score_corr) else "Sleep Quality"
        print(f"CONCLUSION: Significant difference found.\nThe stronger predictor is {winner}.")
    else:
        print("CONCLUSION: No significant difference.\nBoth variables predict the exam score equally well.")

    print("-" * 30)
    print(f"Sleep Steiger's Z-Score: {z_score:.4f}")
    print(f"Sleep P-value:           {p_value:.4g}") # .4g handles very small scientific notation

    print("-" * 30)

    return z_score, p_value
