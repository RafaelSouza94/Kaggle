# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualiztion
import matplotlib.pyplot as plt

# machine learning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# other
import os

# get the data
print("Parsing data with pandas... ")
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print("Done.\n")

# preprocessing
print("Preprocessing the data... ")
train_df = train_df.fillna(0)
y = train_df['SalePrice']
train_df = train_df.drop(['SalePrice'], axis=1)
X = pd.get_dummies(train_df)
# scaling - comment out when using an algorithm that does not benefit from scaling, such as RandomForests
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("Done\n")
#print("Shapes:\nX_test: {}\tX_train: {}\ty_test: {}\ty_train: {}".format(X_test.shape, X_train.shape, y_test.shape, y_train.shape))

# training

print("Training...")
'''param_grid = {'n_estimators': [5, 10, 15, 30, 40, 50],
                     'min_samples_split':[2, 3, 5],
                     'min_samples_leaf':[1, 5, 8]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Grid Search with cross-validation performance: {:.4f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best estimator: {}".format(grid_search.best_estimator_))'''
clf = RandomForestRegressor(min_samples_leaf=1, min_samples_split=5, n_estimators=50)
clf.fit(X_train, y_train)
print("Performance on test set: {:.4f}".format(clf.score(X_test, y_test)))

# preprocess - test data
print("Preprocessing test data... ")
test_df = test_df.fillna(0)
X_sub = pd.get_dummies(test_df)
for col in X_train:
    if len(X_sub.columns) == len(X_train.columns):
        break
    if col not in X_sub:
        X_sub[col] = 0
predictions = clf.predict(X_sub)
IDs = X_sub['Id']
print("Shapes:\nX_sub: {}\tX_train: {}".format(X_sub.shape, X_train.shape))
submission = pd.DataFrame({'Id' : IDs, 'SalePrice' : predictions})
submission.to_csv("Submission.csv", index=False)
print("submission file ready.")

