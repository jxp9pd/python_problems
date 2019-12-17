"""Regression and Regularization for Boston Dataset"""
import regression_methods
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)

WORK_PATH = "C:/Users/jopentak/Documents/"
#%%
#Get the boston data
boston = pd.read_csv(WORK_PATH + "Boston.csv")
