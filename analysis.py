'''
Aging sleeping time connection
Aging  sleeping quality connection
'''
import numpy as np
import pandas as pd
from data import log1_column
from data import sleep_quality_map
from numpy.ma.extras import average
df = pd.read_csv('data/Exam_Score_Prediction.csv') # reading data file

sleep_quality_map(df) #pre-processing stage, creating new colum for sleep quality calculating

def sleeping_time_age_correlation(df):
    '''the following function will calculate the correlation between age and sleeping time - by crating temp df with log values
    for desired cells values, creating more stable calculations via pandas commands'''
    tmp_df = df.copy()
    tmp_df['sleep_time_log'] = np.log1p(tmp_df['sleep_hours']) #creating log values - for accuracy and stability of correlation
    tmp_df['age_log'] = np.log1p(tmp_df['age']) #creating log values - for accuracy and stability of correlation
    corr = tmp_df['sleep_time_log'].corr(tmp_df['age_log']) #calculating correlation between values
    '''
    close to 1 - sleeping more as aging, close to 0 - no connection, close to (-1) - sleeping less as aging
    '''
    return corr

def age_sleep_quality_correlation(df):
    '''the following function will calculate the correlation between age and sleeping quality - by crating temp df with log values
        for desired cells values, creating more stable calculations via pandas commands'''
    tmp_df = df.copy()
    tmp_df['age_log'] = log1_column(df, 'age')#creating log values via external function - for accuracy and stability of correlation
    tmp_df['sleep_quality_log'] = np.log(tmp_df['sleep_quality_num'])  #creating log values (using regular log, no possible 0 value)
    corr = tmp_df['age_log'].corr(tmp_df['sleep_quality_log']) #calculating correlation between values
    '''
    close to 1 - sleeping better as aging, close to 0 - no connection, close to (-1) - sleeping worse as aging
    '''
    return corr

def best_study_method(df):
    """
    The following function will sort the most successful learning method by returning a list where 0 index is the best
    scoring method, any method that wont be included in the list have a small data pool (under 5) that cant give accurate info
    """
    tmp_df = df.copy()
    clean_df = tmp_df.dropna(subset=['exam_score','study_method']) #clearing empty cells for honest mean and ranking
    method_size = clean_df.groupby('study_method')['exam_score'].agg(['mean','count'])
    #grouping, counting and mean calculating
    reliable_method_size = method_size[(method_size['count'] > 5)] #not be affected by few unordinary cases
    sorted_method = reliable_method_size.sort_values(by=['mean'], ascending=False) #sorting method where best performing at [0]
    return sorted_method['mean'].to_dict()

'''
ranking_method_dict = best_study_method(df)
print("ranking learning methods (above 5 subjects) ")
for i, (method, average) in enumerate(ranking_method_dict.items(), 1):
    print(f"{i}. {method}: {average:.2f}")
'''



