"""Regression and Regularization for Boston Dataset"""
import regression_methods
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

WORK_PATH = "C:/Users/jopentak/Documents/"
#%%
#Get the boston data
boston = pd.read_csv(WORK_PATH + "Boston.csv")
#%%
#Visualizing the data
#sns.pairplot(boston)
boston_y = boston.crim
boston_x = boston.drop('crim', axis=1)
#%%
best_forward = regression_methods.forward_selection(boston_x, boston_y)
result_df = regression_methods.get_metric_df(best_forward[0], best_forward[1],
                                             boston_x, boston_y)
#%%
#Lasso Model
lasso_reg = regression_methods.lasso_fit(boston_x, boston_y, 5)
lasso_reg.score(boston_x, boston_y)
lasso_reg.mse_path_[-1].mean()

ridge_reg = regression_methods.ridge_fit(boston_x, boston_y, 5)
ridge_reg.cv_values_
#%%
#Throw in kfold cross-vals for the selection models
result_df['kfold_mse'] = result_df.apply(lambda x: regression_methods.kfold_test(
    boston_x, boston_y, x.features), axis=1)
