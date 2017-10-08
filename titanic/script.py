import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def embarked(value):
    if 'C':
        return 0
    elif 'Q':
        return 1
    return 2

def process_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return extract_class(clean_data(data))
    
def clean_data(data):
    data.drop('Name', axis=1, inplace=True)
    data.drop('Ticket', axis=1, inplace=True)
    data.drop('Cabin', axis=1, inplace=True)
    data['Sex'] = data['Sex'].map(lambda x: 1 if 'male' == x else 0)
    data['Embarked'] = data['Embarked'].map(embarked)
    data.fillna(0, inplace=True)
    return data
    
def extract_class(data):
    if 'Survived' in data.columns:
        return(data.drop('Survived', 1), data['Survived'])
    return (data,[])

train_x, train_y = process_data('train.csv')
test_x, test_y = process_data('test.csv')

#print(train_x.head())
clf = DecisionTreeClassifier();
clf.fit(train_x, train_y)
test_y = clf.predict(test_x)
#print(test_y)
#print(clf.score(test_x, test_y))

pred = pd.DataFrame({
        "PassengerId": test_x['PassengerId'],
        "Survived": test_y
    })
pred.to_csv('predictions.csv', columns=['PassengerId', 'Survived'], index=False)