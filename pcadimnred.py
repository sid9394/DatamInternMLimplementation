from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Social_Network_Ads.csv")
names = ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']
df = pd.read_csv(url, names=names)
features = ['Age', 'EstimatedSalary']
array = df.values
# Separating out the features
X = df.loc[1:, features].values
# Separating out the target
Y = df.loc[1:,['Purchased']].values
# Print out the data
print('X:',X)

# Print out the target values
print('Y:',Y)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

print("Principal df:",principalDf)

finalDf = pd.concat([principalDf, df[['Purchased']]], axis = 1)

print("Final DF",finalDf)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Purchased', 'Not purchased']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Purchased'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
