# Bagged Decision Trees for Classification
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\Social_Network_Ads.csv")
names = ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[2:,2:4]
Y = array[2:,4]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean()*100)