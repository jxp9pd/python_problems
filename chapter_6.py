"""Linear Model Selection and Regularization"""
import itertools
import pdb
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', 20)
DATA_PATH = '/Users/johnpentakalos/Documents/Research Data/'
#%%
def generate_predictor(n, mu, sigma, cols):
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
    X = generate_predictor(n, mu, sigma, d)
    features = X.iloc[:, :len(beta)]
    y = features.dot(beta) + noise
    return X, y

def generate_x(n, num_cols):
    """Generates a predictor matrix. Each column is independently generated"""
    mu = np.random.random(num_cols) * 10
    sigma = np.random.random(num_cols) * 5
    print(mu)
    columns = []
    for mean, std in zip(mu, sigma):
        columns.append(np.random.normal(mean, std, n))
    return pd.DataFrame(columns).T

def generate_y(n, p):
    """Produces a 1-d response vector. Beta is randomly generated with half
    of the p features set to 0. That's combined with the predictor generated
    via generate_x
    """
    beta = np.random.random(p)
    beta[beta<0.5] = 0
    beta *= 5
    noise = np.random.normal(0, 1, n)
    X = generate_x(n, p)
    y = X.dot(beta) + noise
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
#            pdb.set_trace()
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

def fit_lm(X, y):
    """Produces a linear model for the given training data and response"""
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, y)
    return model_k

def test_lm(X_test, y_test, model):
    """Returns test MSE for the given feature set.
    Generates a regression model by stripping down X to the features given.
    Returns test MSE for the given data.
    """
    #pdb.set_trace()
    model = fit_lm(X_test[model], y_test)
    y_hat = model.predict(X_test[model])
    mse = np.sum((y_test - y_hat)**2)/len(y_test)
    return mse

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
#%%
sparse_X, sparse_y = generate_y(1000, 20)

s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(sparse_X, sparse_y,
                                                            test_size=0.9)
result_df = forward_selection(s_X_train, s_y_train)
result_df = get_metrics(result_df, s_X_train, s_y_train)
#%%
test_mse = result_df.apply(lambda x: test_lm(s_X_test, s_y_test, x.features), axis=1)


