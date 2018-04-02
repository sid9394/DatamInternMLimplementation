import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
# Simulate some data
df = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_trainingData1.csv', delimiter=' *, *', engine='python',skipinitialspace=True)
social_df=df
#------------------------------------------------------------
#Preproccessing

#social_df['Gender'].replace('Male', 1, inplace=True)
#social_df['Gender'].replace('Female', 2, inplace=True)
#print(social_df)
#____________________________________________________________

X = social_df.values[1:, 1:]
X=X.astype('int')
y = social_df.values[1:,0]
#y=y.astype('int')
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)

#------------------------------------------------------------
# Fit the Naive Bayes classifier

clf = GaussianNB()

clf.fit(X_train, y_train)

#------------------------------------------------------------
social_data1 = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_testData1.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

X1_test = social_data1.values[1:, :]
#X=X.astype('int')

predicted1= clf.predict(X1_test)
print("Predicted:",predicted1)