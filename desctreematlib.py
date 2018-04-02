from __future__ import print_function
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder

social_data = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_trainingData1.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

X = social_data.values[1:, 1:]
#X=X.astype('int')
y = social_data.values[1:,0]
#Y=Y.astype('int')

#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf = DecisionTreeClassifier().fit(X, y)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
clf_gini.fit(X, y)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=5)
clf_entropy.fit(X, y)

print("done")

#print(clf_gini.predict([[49,36000,1]]))
social_data1 = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_testData1.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

X_test = social_data1.values[1:, :]
#X=X.astype('int')

#---------------------------------------

y_pred = clf_gini.predict(X_test)
print(y_pred)


