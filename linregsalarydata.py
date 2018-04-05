from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Salary_Data.csv")
documents = pd.read_csv(url)

array = documents.values
#choose tweet column
x = array[0:, 1].reshape(-1, 1)
#print(x)
y= array[0:, 0].reshape(-1, 1)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

max_x = max(x)
min_x = min(x)

reg = LinearRegression()
reg.fit (X_train,y_train)

results = model_selection.cross_val_score(reg, x, y)
print(results.mean())
print(X_test)
predicted = reg.predict(X_test)
print(predicted)

print(reg.score(X_test,y_test)*100)

# check that the coeffients are the expected ones.
m = reg.coef_[0]
b = reg.intercept_
#print(' y = {0} * x + {1}'.format(m, b))

# to plot the points and the model obtained
plt.scatter(x, y, color='blue')
plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
plt.title('Fitted linear regression', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
