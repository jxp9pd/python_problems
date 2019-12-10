"""Linear Model Selection and Regularization"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
pd.set_option('display.max_columns', 20)

WORK_PATH = "C:/Users/jopentak/Documents/"
#%%
def fit_linear_reg(X, Y):
    """Fit linear regression model and return RSS and R squared values"""
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
    print('Model performance with R^2: {1:.4f}'.format(RSS, R_squared))
    return model_k

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

def forward_selection(X, y):
    """Linear model selection via forward selection"""
    best_models = []
    curr_model = np.array([])
    available_features = list(X.columns)
    pdb.set_trace()
    #Loop over each feature count.
    for i in range(X.shape[1]):
        best_score = sys.maxsize
        best_feature = ""
        #Loop over possible one feature additions
        for feature in available_features:
            test_model = np.concatenate([curr_model, [feature]])
            rss, r_2 = fit_linear_reg(X[test_model].values, y.values)
            if rss < best_score:
                best_score = rss
                best_feature = feature
        available_features.remove(best_feature)
        curr_model = np.append(curr_model, best_feature)
        best_models.append(curr_model)
    return best_models
forward_selection(X_train, y_train)
#%%
college = pd.read_csv(WORK_PATH + 'college.csv')
college.set_index('Unnamed: 0', inplace=True)
#%%
college = pd.get_dummies(college)
X = college.drop('Apps', axis=1)
y = college.Apps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lin_model = fit_linear_reg(X_train, y_train)
forward_selection(X_train, y_train)
#%%
