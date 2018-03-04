# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:40:55 2018

@author: bhanupalagati
"""

# we are using pickle to predict data
from sklearn import model_selection
import pickle


test_verify = pd.read_csv('gender_submission.csv')
y_test = test_verify.iloc[:,[1]].values
filename = 'titanic_randomForest.sav'

loaded_model = pickle.load(open(filename,'rb'))
y_pred = loaded_model.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
