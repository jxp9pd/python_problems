"""Linear Model Selection and Regularization"""
import pdb
import sys
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

def get_metrics(best_models, X, y):
    """Returns a dataframe of BIC criterion, Adjusted R^2, Mallows Cp
        best_models -- List of generated models. Each model is represented as a
        list of feature column names.
    """
    variance = np.var(y)
    models = []
    for features in best_models:
        mse, r_2 = fit_linear_reg(X[features], y)
        rss = mse * len(y)
        mal_cp = mallow_cp(rss, variance, len(y), len(features))
        bic = BIC(rss, variance, len(y), len(features))
        adj_r2 = adjusted_r2(rss, y, len(features))
        metadata = pd.DataFrame({'RSS': rss, 'R_squared': r_2, 'Cp': mal_cp,
                                 'BIC': bic, 'adj_r2': adj_r2, 'numb_features':
                                len(features), 'features': [features]})
        models.append(metadata)
    return pd.concat(models)

def fit_linear_reg(X, y):
    """Fit linear regression model and return mse and R squared values"""
    #pdb.set_trace()
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(X, y)
    mse = mean_squared_error(y, model_k.predict(X))
    R_squared = model_k.score(X, y)
    print('Model performance with loss {0:.4f} and R^2: {0:.4f}'.format(mse, R_squared))
    return mse, model_k.score(X,y)

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
    #Loop over each feature count.
    for i in range(X.shape[1]):
        best_score = sys.maxsize
        best_feature = ""
        #Loop over possible one feature additions
        for feature in available_features:
            test_model = np.concatenate([curr_model, [feature]])
            mse, r_2 = fit_linear_reg(X[test_model].values, y.values)
            if mse < best_score:
                best_score = mse
                best_feature = feature
        available_features.remove(best_feature)
        curr_model = np.append(curr_model, best_feature)
        best_models.append(curr_model)
    return best_models

def test_lm(X_test, y_test, model):
    """Produces test metrics for a given linear model"""
    model.predict(X_test, y_test)
#%%
college = pd.read_csv(WORK_PATH + 'college.csv')
college.set_index('Unnamed: 0', inplace=True)
#%%
college = pd.get_dummies(college)
X = college.drop('Apps', axis=1)
y = college.Apps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lin_model = fit_linear_reg(X_train['Accept'], y_train)
#%%
forward_models = forward_selection(X_train, y_train)
result_df = get_metrics(forward_models, X_test, y_test)
best_features = result_df.sort_values(by=['BIC']).head(1).features[0]
best_model = fit_linear_reg(X_train[best_features], y_train)
#%%
test_score = 
print("Linear model performance: {0}".format())