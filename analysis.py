'''
Aging sleeping time connection
Aging  sleeping quality connection
'''
import numpy as np
import pandas as pd
from numpy.ma.extras import average
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

"""
The following function will sort the most successful learning method by returning a list where 0 index is the best
scoring method, any method that wont be included in the list have a small data pool (under 5) that cant give accurate info
"""
def best_study_method(df):
    clean_df = df.dropna(subset=['exam_score','study_method']) #clearing empty cells for honest mean and ranking
    method_size = clean_df.groupby('study_method')['exam_score'].agg(['mean','count'])
    #grouping, counting and mean calculating
    reliable_method_size = method_size[(method_size['count'] > 5)] #not be affected by few unordinary cases
    sorted_method = reliable_method_size.sort_values(by=['mean'], ascending=False) #sorting method where best performing at [0]
    return sorted_method['mean'].to_dict()

'''main code for previous function: 

ranking_method_dict = best_study_method(df)
print("ranking learning methods (above 5 subjects) ")
for i, (method, average) in enumerate(ranking_method_dict.items(), 1):
    print(f"{i}. {method}: {average:.2f}")
'''



