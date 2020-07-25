# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:15:50 2020

@author: I505860
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

dataset = pd.read_csv('C:/Users/i505860/OneDrive - SAP SE/Documents/My Folder/Personnel/Extras/Extra_Dataset/WineQuality_Flask_Heroku/Dataset_winequality.csv')
dataset = dataset[['citric acid','density', 'pH', 'alcohol','Quality']]

dataset.fillna(0, inplace=True)


X = dataset.iloc[:, :4]
y = dataset.iloc[:, -1]


X = X.astype(float)
y  = y.astype(int)

regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model_wine.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_wine.pkl','rb'))


inputt=[float(x) for x in "1, 1, 3,15".split(',')]
final=[np.array(inputt)]

model.predict(final)
model.predict_proba(final)

regressor.predict(final)
regressor.predict_proba(final)
