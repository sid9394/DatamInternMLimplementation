# Voting Decision Trees for Classification
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
Y = array[2:,4]

#kfold = model_selection.KFold(n_splits=10, random_state=seed)

model1 = GaussianNB().fit(X,Y)
cart_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=5)
cart_entropy.fit(X, Y)
#clf5 = svm.SVC(kernel='linear', C=1)
#clf5.fit(X, Y)
clf5=LogisticRegression(random_state=10).fit(X,Y)
eclf = VotingClassifier(estimators=[('GNB', model1), ('DTC', cart_entropy), ('svm', clf5)], voting='hard')

for clf1, label in zip([model1, cart_entropy, clf5, eclf], ['Gaussian Naive Bayes', 'Decision Tree Classifier', 'SVM', 'Voting Ensemble']):
   scores = cross_val_score(clf1, X, Y, cv=5, scoring='accuracy')
   print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))