from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\Social_Network_Ads.csv")
names = ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[2:,2:4]
y = array[2:,4]

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = GaussianNB()
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])

clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
eclf = eclf.fit(X,y)

for clf1, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree Classifier', 'Gaussian Naive Bayes', 'SVM', 'Voting Ensemble']):
   scores = cross_val_score(clf1, X, y, cv=5, scoring='accuracy')
   print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))