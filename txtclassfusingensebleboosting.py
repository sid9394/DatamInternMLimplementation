from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Tweets.csv")
documents = pd.read_csv(url)

array = documents.values
#choose tweet column
x = array[0:, 10]
#print(x)
y= array[0:, 1]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x)
#print(X_train_counts.shape)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed).fit(X_train_tfidf, y)
results = model_selection.cross_val_score(model, X_train_tfidf, y, cv=kfold)
print("Acc. to AdaBoost Classification")
print(results.mean())
#Stochastic Gradient Boosting
model1 = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed).fit(X_train_tfidf, y)
results = model_selection.cross_val_score(model1, X_train_tfidf, y, cv=kfold)
print("Acc. to Stochastic Gradient Boosting")
print(results.mean())

url1 = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\outputkrithika.csv")
documents1 = pd.read_csv(url1)
array1 = documents1.values
#choose tweet column
#x1 = array1[0:, 2]
x2= (documents1['tweet']).astype(str)

X_test = count_vect.transform(x2)
#print(X_test.shape)

test = tfidf_transformer.transform(X_test)
#print(test.shape)

predicted = model.predict(test)
print(predicted)
predicted1 = model1.predict(test)
print(predicted1)