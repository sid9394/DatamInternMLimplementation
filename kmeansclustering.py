from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Mall_Customers.csv")
df = pd.read_csv(url)

#choose
x = df['Annual Income (k$)'].values
print(x)
y = df['Spending Score (1-100)'].values
print(y)

X=list(zip(x,y))
kmeans = KMeans(n_clusters=5).fit(X)
y_kmeans = kmeans.predict(X)
# to plot the points and the model obtained
#plt.scatter(x, y, color='blue')
# plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
plt.scatter(x,y, c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0],centers[:, 1], c='black');
plt.title('Fitted KMeans', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()