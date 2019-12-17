"""Linear Model Selection and Regularization"""
import regression_methods
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
pd.set_option('display.max_columns', 20)
DATA_PATH = '/Users/johnpentakalos/Documents/Research Data/'

#%%
def lasso_alpha(regr):
    """Produces a scatterplot for lambda selection"""
    mse_path = regr.mse_path_
    mse_kfold = np.mean(mse_path, axis=1)
    alphas = regr.alphas_
    plt.title('Scatter plot lambda vs. MSE K-fold CV')
    plt.xlabel('Lambda value')
    plt.ylabel('5-Fold CV MSE')
    plt.scatter(alphas, mse_kfold)
    plt.show()

def lasso_fit(X, y, k):
    """Runs a lasso regression with k-fold cross-validation"""    
    regr = LassoCV(cv=k, random_state=3, max_iter=200).fit(X, y)
    lasso_alpha(regr)
    return regr

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
lasso_regr = lasso_fit(X, y, 5)
lasso_regr.coef_
lasso_regr.intercept_
y.mean()
#%%
sparse_X, sparse_y = regression_methods.generate_y(1000, 20)
s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(sparse_X, sparse_y,
                                                            test_size=0.9)
forward_2 = regression_methods.forward_selection(X, y)
result_df = regression_methods.get_metric_df(forward_2[0], forward_2[1], X, y)
#%%
result_df['testMSE'] = result_df.apply(lambda x: regression_methods.test_mse(
    s_X_test, s_y_test, x['model'], x['features']), axis=1)
result_df['trainMSE'] = result_df.apply(lambda x: regression_methods.train_mse(
    s_X_train, s_y_train, x['model'], x['features']), axis=1)
best_features = result_df.sort_values(by=['testMSE']).head(1)['features'][0]
#%%
regression_methods.mse_plot(result_df['numb_features'].values, result_df['testMSE'].values, 
         "Test MSE Plot")

regression_methods.mse_plot(result_df['numb_features'].values, result_df['trainMSE'].values,
         "Training MSE Plot")
