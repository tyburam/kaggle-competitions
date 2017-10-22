#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:37:15 2017

@author: mateusztybura
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

def fillnans(data):
    if data.dtype is pd.np.dtype(float) or data.dtype is pd.np.dtype(int):
        return data.fillna(0)
    elif data.dtype is pd.np.dtype(object):
        return data.fillna('.')
    else:
        return data

def process_data(file_path, is_train_data=False):
    raw_data = pd.read_csv(file_path)
    
    raw_data.dropna(1, 'any', 1200, None, True)
    raw_data.dropna(0, 'any', 40, None, True)
    raw_data = raw_data.apply(fillnans)
    raw_data.select_dtypes(['float64']).fillna(0.0, inplace=True)

    #dropcolumns = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    #               'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',
    #               'GarageQual', 'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st',
    #               'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
    #for col in dropcolumns:
    #    raw_data.drop(col, axis=1, inplace=True)
    
    raw_data[raw_data.select_dtypes(['object']).columns] = raw_data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    columns = raw_data.select_dtypes(['category']).columns.values.tolist()
    for col in columns:
        print(col)
        raw_data[col] = preprocessing.LabelEncoder().fit_transform(raw_data[col])
    
    if(is_train_data):
        y = raw_data['SalePrice']
        raw_data.drop('SalePrice', axis=1, inplace=True)
        raw_data.drop('Id', axis=1, inplace=True)
        return [raw_data,y]
    
    return (raw_data, [])

def id_column(x_data):
    id_vals = x_data['Id']
    x_data.drop('Id', axis=1, inplace=True)
    return (id_vals, x_data)
    
train_x, train_y = process_data('train.csv', True)
test_x, test_y = process_data('test.csv')

ids = test_x['Id'].tolist()
test_x.drop('Id', axis=1, inplace=True)

print(train_x.info())
print(test_x.info())

#print(train_x.head())
clf = DecisionTreeRegressor();
clf.fit(train_x, train_y)
test_y = clf.predict(test_x)

pred = pd.DataFrame({
        "Id": ids,
        "SalePrice": test_y
    })
pred.to_csv('predictions.csv', columns=['Id', 'SalePrice'], index=False)