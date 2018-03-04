# Classifier template


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import pickle

# Importing the dataset for training
dataset = pd.read_csv('train.csv')
new_data = dataset.iloc[:,[0,1,2,4,5,6,7,9,11]]
##encoding the training data
data_dummy = pd.get_dummies(new_data)
##
X_train = data_dummy.iloc[:, [0,2,3,4,5,6,7,8,9,10,11]].values
y_train = data_dummy.iloc[:, 1].values

#missing valuse addressing for training data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X_train[:, [2]])
X_train[:, [2]] = imputer.transform(X_train[:, [2]])
np.set_printoptions(threshold = np.nan)






test_dataset = pd.read_csv('test.csv')
test_verify = pd.read_csv('gender_submission.csv')
new_test_data = test_dataset.iloc[:,[0,1,3,4,5,6,8,10]]
##encoding the training data
test_data_dummy = pd.get_dummies(new_test_data)
##
X_test = test_data_dummy.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
y_test = test_verify.iloc[:,[1]].values

#missing valuse addressing for training data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])
imputer = imputer.fit(X_test[:, [5]])
X_test[:, [5]] = imputer.transform(X_test[:, [5]])
np.set_printoptions(threshold = np.nan)






# fitting the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)


# pickling
filename = 'titanic_randomForest.sav'
pickle.dump(classifier,open(filename,'wb'))


# predicting the test set results
y_pred = classifier.predict(X_test)



# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Test set results
# high dimensionality reduction techniques needed
