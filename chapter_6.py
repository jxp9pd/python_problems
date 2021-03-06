"""Linear Model Selection and Regularization"""
import regression_methods
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 20)
DATA_PATH = '/Users/johnpentakalos/Documents/Research Data/'

#%%
beta = np.array([15, 2, 1])
beta_2 = np.array([0, 0, 0, 0, 0, 0, 0.05])
X, y = regression_methods.generate_response(1000, 7, 3.8, beta_2, 10)

result_df = regression_methods.best_subset(X, y)
result_df.sort_values(by=['R_squared'], ascending=False).head()
result_df.sort_values(by=['BIC']).head()
result_df.sort_values(by=['adj_r2'], ascending=False).head(1)

best_forward = regression_methods.forward_selection(X, y) #tuple 0 features, 1 models
best_backward = regression_methods.backward_selection(X, y)
#%%
forward_results = regression_methods.get_metric_df(best_forward[0], best_forward[1],
                                                    X, y)
back_results = regression_methods.get_metric_df(best_backward[0], best_backward[1],
                                                   X, y)
#%%
lasso_regr = regression_methods.lasso_fit(X, y, 5)
lasso_regr.coef_
lasso_regr.intercept_
y.mean()
#%%
sparse_X, sparse_y, beta = regression_methods.generate_y(1000, 20)
s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(sparse_X, sparse_y,
                                                            test_size=0.9)
forward_2 = regression_methods.forward_selection(s_X_train, s_y_train)
result_df = regression_methods.get_metric_df(forward_2[0], forward_2[1], 
                                             s_X_train, s_y_train)
#%%
result_df['testMSE'] = result_df.apply(lambda x: regression_methods.test_mse(
    s_X_test, s_y_test, x['model'], x['features']), axis=1)
result_df['trainMSE'] = result_df.apply(lambda x: regression_methods.train_mse(
    s_X_train, s_y_train, x['model'], x['features']), axis=1)
best_features = result_df.sort_values(by=['testMSE']).head(1)['features'][0]
best_model = result_df.sort_values(by=['testMSE']).head(1)['model'][0]
#%%
#Compare plots of test error and training error.
regression_methods.mse_plot(result_df['numb_features'].values, result_df['testMSE'].values, 
         "Test MSE Plot")

regression_methods.mse_plot(result_df['numb_features'].values, result_df['trainMSE'].values,
         "Training MSE Plot")
#%%
#Compare true coefficient vector with produced one.
ceoff_vectors = regression_methods.get_coef_vector(best_features, best_model,
                                                     20)
coeff_vectors = result_df.apply(lambda x: regression_methods.get_coef_vector(
    x['features'], x['model'], 20), axis=1)
result_df['diff'] = coeff_vectors.apply(lambda x: regression_methods.coef_diffs(
    x, beta))
