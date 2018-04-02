from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

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
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators).fit(X_train_tfidf, y)
results = model_selection.cross_val_score(ensemble, X_train_tfidf, y, cv=kfold)
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

predicted = ensemble.predict(test)
print(predicted)
