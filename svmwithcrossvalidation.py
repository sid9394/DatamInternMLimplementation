import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np

social_data = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_trainingData1.csv',sep=',', dtype={'Name': str })

X = social_data.values[1:1000, 1:]

y = social_data.values[1:1000,0]


clf = svm.SVC(kernel='linear', C=1).fit(X, y)
print(clf.score(X, y))

#------------------------------------------------------------------------------------------------------------------

social_data1 = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_testData1.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

X1_test = social_data1.values[1:1000, :]
#X=X.astype('int')

predicted1= clf.predict(X1_test)
print("Predicted:",predicted1)

