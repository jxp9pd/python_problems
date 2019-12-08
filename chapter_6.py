'''Linear Model Selection and Regularization'''
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
DATA_PATH = '/Users/johnpentakalos/Documents/Research Data/'
#%%
def generate_X(n, mu, sigma, cols):
    '''Generates a predictor matrix with the given parameters. Cols refers to
    the number of polynomial degrees generated.'''
    x = np.random.normal(mu, sigma, n)
    predictors = []
    #pdb.set_trace()
    for i in range(cols):
        predictors.append(np.power(x, i))
    df = pd.DataFrame(predictors).T
    return df

def generate_response(n, mu, sigma, beta):
    '''Generates random data points n, from a normal distribution centered at
    mu with variance sigma.'''
    noise = np.random.normal(0, 1, 100)
#    beta = np.random.rand(4) * 3.5
    X = generate_X(n, mu, sigma, len(beta))
    y = X.dot(beta) + noise
    return X, y

#%%
def fit_linear_reg(X,Y):
    '''Fit linear regression model and return RSS and R squared values'''
    model_k = linear_model.LinearRegression(fit_intercept = False)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    print(model_k.coef_)
    return RSS, R_squared

#Initialization variables
def best_subset(X, y):
    RSS_list, R_squared_list, feature_list = [],[],[]
    numb_features = []
    #Looping over k = 1 to k = 11 features in X
    for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):
        #Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns,k):
            tmp_result = fit_linear_reg(X[list(combo)], y)   #Store temp result 
            RSS_list.append(tmp_result[0])                  #Append lists
            R_squared_list.append(tmp_result[1])
            feature_list.append(combo)
            numb_features.append(len(combo))   
    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, \
                       'R_squared':R_squared_list,'features':feature_list})
    return df
#%%
X = generate_X(100, 7, 3.8, 5)
beta = np.array([12, 15, 2, 0.1])
x, y = generate_response(100, 7, 3.8, beta)
RSS, r2 = fit_linear_reg(x, y)
#%%


