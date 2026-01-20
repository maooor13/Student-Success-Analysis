from analysis import compare_sleep_quality_and_hours_with_score
from data import df


def main():
    """
    Stages:
    1. import data
    2. pre-process data
    3. answer questions
    4. visualize 
    """
    sleep_and_score_z_score, sleep_and_score_p_value = compare_sleep_quality_and_hours_with_score(df)

    
    
    
if __name__ == "__main__":
    main()