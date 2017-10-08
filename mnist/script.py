#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:43:56 2017

@author: mateusztybura
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def process_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return extract_class(data)
    
def extract_class(data):
    if 'label' in data.columns:
        return(data.drop('label', 1), data['label'])
    return (data,[])

train_x, train_y = process_data('train.csv')
test_x, test_y = process_data('test.csv')

#print(train_x.head())
clf = KNeighborsClassifier();
clf.fit(train_x, train_y)
test_y = clf.predict(test_x)
#print(test_y)
print(clf.score(test_x, test_y))

#ImageId,Label
pred = pd.DataFrame({
        "ImageId": np.arange(1,28001),
        "Label": test_y
    })
pred.to_csv('predictions.csv', columns=['ImageId', 'Label'], index=False)