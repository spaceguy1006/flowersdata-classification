# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:12:25 2022

@author: shrey
"""

"""  This is a project that will use random tree classification and 
random forest classification to classify types of flowers. """


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

flowers = pd.read_csv(r'C:\Users\shrey\OneDrive\Desktop\coding trials\flowers.csv')

list_of_col = list(flowers.columns)

print(list_of_col)

# There are only 5 columns. 4 of these columns are the features of the flower
# and the 5th is the species. All these features are numerical which makes
# handling the data easier.

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Dropping rows with any null values

flowers = flowers.dropna(axis=0)

X = flowers[features]

# Our target variable is the species. This is what I'm trying to predict.

y = flowers.species

# Splitting our data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, train_size = 0.5)

flower_model = DecisionTreeClassifier()

# Fitting the data

flower_model.fit(train_X, train_y)
y_pred = flower_model.predict(test_X)

print('The 15th flower is predicted to be {pre}. It actually is a {real}. '.format(pre = y_pred[15], real = list(test_y)[15]))

""" This prediction is correct however this may not be the case for all flowers.
    We can use the accuancy function to test how accurate this model is"""
    
from sklearn.metrics import accuracy_score

print('The model\'s accuracy is {}'.format(accuracy_score(y_pred, test_y)))

""" The prediction of this model is accurate as it has a 96% success rate."""
