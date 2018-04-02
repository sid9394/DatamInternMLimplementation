from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

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

kf = KFold(n_splits=10) # Define the split - into 2 folds
kf.get_n_splits(X_train_tfidf) # returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(X_train_tfidf):
   print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = X_train_tfidf[train_index], X_train_tfidf[test_index]
y_train, y_test = y[train_index], y[test_index]

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
#print(y_pred)

y_pred_en = clf_entropy.predict(X_test)
#print(y_pred_en)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)

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

predicted = clf_gini.predict(test)
print(predicted)

predicted = clf_entropy.predict(test)
print(predicted)