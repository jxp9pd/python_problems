"""Linear Model Selection and Regularization"""
import itertools
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', 20)
DATA_PATH = '/Users/johnpentakalos/Documents/Research Data/'
#%%
def generate_x(n, mu, sigma, cols):
    """Generates a predictor matrix with the given parameters.
    A single vector is generated from a norm. Cols refers to the number of
    vectors produced. (e.g. cols = 2, returns x and x^2)
    """
    x = np.random.normal(mu, sigma, n)
    predictors = []
    for i in range(1, cols):
        predictors.append(np.power(x, i))
    df = pd.DataFrame(predictors).T
    return df

def generate_response(n, mu, sigma, beta, d):
    """Produces both a feature matrix and response vector
    Response vector is generated from some set of true parameters (beta)
    combined with the feature matrix with a noise feature added in.
    Feature matrix is created by above helper method.
    """
    noise = np.random.normal(0, 1, n)
    X = generate_x(n, mu, sigma, d)
    features = X.iloc[:, :len(beta)]
    y = features.dot(beta) + noise
    return X, y

#%%
def mallow_cp(RSS, var, n, d):
    """Calculates Mallows Cp"""
    return (RSS + 2 * d * var)/n

def BIC(RSS, var, n, d):
    """Calculates BIC Criterion"""
    return (RSS + np.log(n) * d * var)/(n*var)

def adjusted_r2(RSS, y, d):
    """Calculates adjusted r^2"""
    n = len(y)
    TSS = np.sum((y - np.mean(y))**2)
    return 1 - (RSS/(n - d - 1))/(TSS/(n-1))

#%%
#Feature selection methods
def fit_linear_reg(X, Y):
    """Fit linear regression model and return RSS and R squared values"""
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
    return RSS, R_squared

def best_subset(X, y):
    """Runs a linear model fit for every possible combination of features"""
    RSS_list, R_squared_list, feature_list = [], [], []
    numb_features = []
    #Looping over k = 1 to k = 11 features in X
    for k in tnrange(1, len(X.columns) + 1, desc='Loop...'):
        #Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns, k):
            tmp_result = fit_linear_reg(X[list(combo)], y)   #Store temp result
            RSS_list.append(tmp_result[0])                   #Append lists
            R_squared_list.append(tmp_result[1])
            feature_list.append(combo)
            numb_features.append(len(combo))
    variance = np.var(y)
    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features, 'RSS': RSS_list, 
                       'R_squared': R_squared_list, 'features': feature_list})
    df['BIC'] = df.apply(lambda x: BIC(x.RSS, variance, len(y), len(x.features)), axis=1)
    df['Cp'] = df.apply(lambda x: mallow_cp(x.RSS, variance, len(y), len(x.features)), axis=1)
    df['adj_r2'] = df.apply(lambda x: adjusted_r2(x.RSS, y, len(x.features)), axis=1)
    return df

def forward_selection(X, y):
    """Linear model selection via forward selection"""
    best_models = []
    curr_model = np.array([])
    available_features = list(X.columns)
    #Loop over each feature count.
    for i in range(X.shape[1]):
        best_score = sys.maxsize
        best_feature = -1
        #Loop over possible one feature additions
        for feature in available_features:
            test_model = np.concatenate([curr_model, [feature]])
            rss, r_2 = fit_linear_reg(X[test_model], y)
            if rss < best_score:
                best_score = rss
                best_feature = feature
        available_features.remove(best_feature)
        curr_model = np.append(curr_model, best_feature)
        best_models.append(curr_model)
    return best_models

def backward_selection(X, y):
    """Linear model selection via forward selection"""
    curr_model = list(X.columns)
    best_models = [np.array(curr_model)]
    for i in range(1, X.shape[1]):
        best_score = sys.maxsize
        best_feature = -1
        for feature in curr_model:
            test_model = X.drop(feature, axis=1)
            rss, r_2 = fit_linear_reg(test_model, y)
            if rss < best_score:
                best_score = rss
                best_feature = feature
        X = X.drop(best_feature, axis=1)
        curr_model.remove(best_feature)
        best_models.append(np.array(curr_model))
    return best_models

def get_metrics(best_models, X, y):
    """Returns a dataframe of BIC criterion, Adjusted R^2, Mallows Cp
        best_models -- List of generated models. Each model is represented as a
        list of feature column names.
    """
    variance = np.var(y)
    models = []
    for features in best_models:
        rss, r_2 = fit_linear_reg(X[features], y)
        mal_cp = mallow_cp(rss, variance, len(y), len(features))
        bic = BIC(rss, variance, len(y), len(features))
        adj_r2 = adjusted_r2(rss, y, len(features))
        metadata = pd.DataFrame({'RSS': rss, 'R_squared': r_2, 'Cp': mal_cp,
                                 'BIC': bic, 'adj_r2': adj_r2, 'numb_features':
                                len(features), 'features': [features]})
        models.append(metadata)
    return pd.concat(models)
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

def lasso_fit(X, y, k):
    """Runs a lasso regression with k-fold cross-validation"""    
    regr = LassoCV(cv=k, random_state=3, max_iter=200).fit(X, y)
    lasso_alpha(regr)
    return regr

#%%
beta = np.array([15, 2, 1])
beta_2 = np.array([0, 0, 0, 0, 0, 0, 0.05])
X, y = generate_response(1000, 7, 3.8, beta_2, 10)
result_df = best_subset(X, y)
result_df.sort_values(by=['R_squared'], ascending=False).head()
result_df.sort_values(by=['BIC']).head()
result_df.sort_values(by=['adj_r2'], ascending=False).head(1)
best_forward = forward_selection(X, y)
best_backward = backward_selection(X, y)
#%%
forward_metrics = get_metrics(best_forward, X, y)
backward_metrics = get_metrics(best_backward, X, y)
#%%
lasso_regr = lasso_fit(X, y, 5)
lasso_regr.coef_
lasso_regr.intercept_
y.mean()
