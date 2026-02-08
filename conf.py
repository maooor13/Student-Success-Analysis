"""
At the moment the conf.py is pretty hard-coded, for now you need to
change the code to change configuration.
In the future we can implement reading a conf.json file for more dynamic use.
"""

class Conf:
    def __init__(self):
        self.LOG_LEVEL = "INFO"
        self.CSV = "data/Exam_Score_Prediction.csv"
        self.TARGET = "exam_score" # The dependant variable.
        self.OSL_ROBUST = "HC1" # HC1 = Degrees of freedom. https://sociology-fa-cu.github.io/appliedregressioninr/08_heterscedasticity.html