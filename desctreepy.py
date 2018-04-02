from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

social_data = pd.read_csv('C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\super_learner_trainingData1.csv',sep= ',', header= None)

#social_data.replace({'Male': 7, 'Female': 8, 'Purchased': 4, 'EstimatedSalary': 3, 'Age': 2, 'Gender': 1}, inplace=True)

X = social_data.values[1:, 1:785]
#X=X.astype('int')
Y = social_data.values[1:,784]
Y=Y.astype('int')
print(X)
print(Y)
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#Cross validation
kf = KFold(n_splits=10) # Define the split - into 2 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = Y[train_index], Y[test_index]

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
#print(y_pred)

y_pred_en = clf_entropy.predict(X_test)
#print(y_pred_en)

# Making the Confusion Matrix
print("Confusion Matrix -")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
