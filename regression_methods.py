"""Regression and Regularization Code"""
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
#%%
#Data Generators
def generate_predictor(n, mu, sigma, cols):
    """Generates a polynomial predictor matrix with the given parameters.
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
    """Produces a polynomial feature matrix and response vector
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
    beta[beta<0.5] = 0 # Adjust what proportion of features are set to zero
    beta *= 5 # Scale the parameters
    noise = np.random.normal(0, 1, n)
    print(beta)
    X = generate_x(n, p)
    y = X.dot(beta) + noise #y as a linear combination of X and beta + noise
    return X, y
def get_coef_vector(features, model, p):
    """Returns a full coefficient vector.
    Model coef_ isn't aware of zero coefficients. Combining info between featu-
    res and coef_ produces a full coeff. vector.
    """
    
#%%
#Regression regularization metrics
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

def get_MSE(model, X, y):
    """Returns MSE for a given model"""
    return mean_squared_error(y, model.predict(X))

def get_metric_df(best_features, best_models, X, y):
    """Returns a dataframe of BIC criterion, Adjusted R^2, Mallows Cp
        best_models -- List of generated models. Each model is represented as a
        list of feature column names.
    """
    variance = np.var(y)
    rows = []
    for features, model in zip(best_features, best_models):
        # pdb.set_trace()
        p = len(features)
        rss = get_MSE(model, X[features], y) * len(y)
        r_2 = model.score(X[features], y)
        mal_cp = mallow_cp(rss, variance, len(y), p)
        bic = BIC(rss, variance, len(y), p)
        adj_r2 = adjusted_r2(rss, y, p)
        metadata = pd.DataFrame({'RSS': rss, 'R_squared': r_2, 'Cp': mal_cp,
                                  'BIC': bic, 'adj_r2': adj_r2, 'numb_features':
                                p, 'features': [features], 'model': model})
        rows.append(metadata)
    return pd.concat(rows)

#%%
#Model generation
def fit_linear_reg(X, Y):
    """Fit linear regression model and return RSS and R squared values"""
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, Y)
    RSS = mean_squared_error(Y, model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X, Y)
    return RSS, R_squared

def fit_lm(X, y):
    """Produces a linear model for the given training data and response"""
    print(X.head())
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, y)
    return model_k
#%%
#Model selection
def best_subset(X, y):
    """Runs a linear model fit for every possible combination of features"""
    RSS_list, R_squared_list, feature_list = [], [], []
    models, numb_features = [], []
    #Looping over k = 1 to k = 11 features in X
    for k in tnrange(1, len(X.columns) + 1, desc='Loop...'):
        #Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns, k):
            temp_x = X[list(combo)]
            model = fit_lm(temp_x, y)
            rss = get_MSE(model, temp_x, y) * len(y)
            RSS_list.append(rss)  #Append lists
            R_squared_list.append(model.score(temp_x, y))
            models.append(model)
            feature_list.append(combo)
            numb_features.append(len(combo))
    variance = np.var(y)
    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features, 'RSS': RSS_list, 
                       'R_squared': R_squared_list, 'features': feature_list, 
                       'Model': models})
    df['BIC'] = df.apply(lambda x: BIC(x.RSS, variance, len(y),
                                       len(x.features)), axis=1)
    df['Cp'] = df.apply(lambda x: mallow_cp(x.RSS, variance, len(y),
                                            len(x.features)), axis=1)
    df['adj_r2'] = df.apply(lambda x: adjusted_r2(x.RSS, y, len(x.features)),
                            axis=1)
    return df

def forward_selection(X, y):
    """Linear model selection via forward selection"""
    best_features, best_models = [], []
    curr_model = np.array([])
    available_features = list(X.columns)
    #Loop over each feature count.
    for i in range(X.shape[1]):
        best_score = sys.maxsize
        best_feature = -1
        #Loop over possible one feature additions
        for feature in available_features:
            print(curr_model)
            test_model = np.concatenate([curr_model, [feature]])
            lm = fit_lm(X[test_model], y)
            mse = get_MSE(lm, X[test_model], y)
            if mse < best_score:
                best_score = mse
                best_feature = feature
        available_features.remove(best_feature)
        curr_model = np.append(curr_model, best_feature)
        best_features.append(curr_model)
        best_models.append(fit_lm(X[curr_model], y))
    return best_features, best_models

def backward_selection(X, y):
    """Linear model selection via forward selection"""
    curr_model = list(X.columns)
    best_models = [fit_lm(X[curr_model], y)]
    best_features = [np.array(curr_model)]
    for i in range(1, X.shape[1]):
        best_score = sys.maxsize
        best_feature = -1
        for feature in curr_model:
            test_X = X.drop(feature, axis=1)
            lm = fit_lm(test_X, y)
            mse = get_MSE(lm, test_X, y)
            if mse < best_score:
                best_score = mse
                best_feature = feature
        X = X.drop(best_feature, axis=1)
        curr_model.remove(best_feature)
        best_features.append(np.array(curr_model))
        best_models.append(fit_lm(X, y))
    return best_features, best_models

def mse_plot(num_features, test_mse, label):
    """Produces a test MSE plot"""
    #pdb.set_trace()
    plt.title(label)
    plt.xlabel("Number of features")
    plt.ylabel("MSE")
    plt.scatter(num_features, test_mse)
    plt.show()    

def test_lm(X_test, y_test, X_train, y_train, model_features):
    """Returns test MSE for the given feature set.
    Generates a regression model by stripping down X to the features given.
    Returns test MSE for the given data.
    """
    model = fit_lm(X_train[model_features], y_train)
    y_hat = model.predict(X_test[model_features])
    mse = np.sum((y_test - y_hat)**2)/len(y_test)
    return mse
