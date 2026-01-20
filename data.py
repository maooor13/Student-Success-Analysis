import numpy as np
import pandas as pd
from numpy.ma.extras import average
df = pd.read_csv('data/Exam_Score_Prediction.csv') # reading data file

def log1_column (df, string): #valid for numeral data only
    tmp_df = df.copy()
    tmp_df['new_log'] = np.log1p(df[string])
    return tmp_df['new_log']

def sleep_quality_map (df):
    quality_num_map = {'poor': 1, 'average': 2, 'good': 3}  # creating a map to turn strings into numbers
    df["sleep_quality_num"] = df['sleep_quality'].map(quality_num_map)  # using the map to translate strings into numbers
    return df
