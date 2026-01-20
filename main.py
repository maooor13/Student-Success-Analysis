#from analysis import compare_sleep_quality_and_hours_with_score
from data import df
from data import log1_column
from data import sleep_quality_map
from analysis import sleeping_time_age_correlation
from analysis import age_sleep_quality_correlation
from analysis import best_study_method

def main():
    """
    Stages:
    1. import data
    2. pre-process data
    3. answer questions
    4. visualize 
    """
    sleep_and_score_z_score, sleep_and_score_p_value = compare_sleep_quality_and_hours_with_score(df)

corr_sleep_time_age = sleeping_time_age_correlation(df) #calculating sleep time - age question into a value
corr_sleep_quality_age = age_sleep_quality_correlation(df) #calculating sleep quality - age question into a value
print ("Does people tend to sleep less as they age?")
print ("close to 1 - sleeping more as aging, close to 0 - no connection, close to (-1) - sleeping less as aging")
print (f"Age and sleep quality correlation: {corr_sleep_time_age}")

print ("Does people tend to worse as they age?")
print("close to 1 - sleeping better as aging, close to 0 - no connection, close to (-1) - sleeping worse as aging")
print (f"Age and sleep quality correlation: {corr_sleep_quality_age}")

ranking_method_dict = best_study_method(df) #calculating best methods via external function
print("ranking learning methods (above 5 subjects) ") #specifing importance of reliable data
for i, (method, average) in enumerate(ranking_method_dict.items(), 1): #using loop for printing full dict
    print(f"{i}. {method}: {average:.2f}")
    
if __name__ == "__main__":
    main()

