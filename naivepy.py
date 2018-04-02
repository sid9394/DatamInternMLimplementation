import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#Data import
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Social_Network_Ads.csv")
names = ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
x = array[1:,2:4]
y = array[1:,4]
print(x)
print(y)
model = GaussianNB()

kf = KFold(n_splits=10) # Define the split - into 2 folds
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(x):
   print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = x[train_index], x[test_index]
y_train, y_test = y[train_index], y[test_index]

# Train the model using the training sets
model.fit(X_train, y_train)

#Predict Output
predicted= model.predict([1,2][3,4])
print(predicted)

