import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style
style.use("ggplot")

def Build_Data_Set(features = ["EstimatedSalary","Age"]):
    data_df = pd.DataFrame.from_csv("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Intern_work\\Social_Network_Ads.csv")

    data_df = data_df[:100]

    X = np.array(data_df[features].values)

    y = (data_df["Purchased"]
         .replace("1",0)
         .replace("0",1)
         .values.tolist())

    return X,y

def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X, y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]

    #h0 = plt.plot(xx, yy, "k-", label="age")

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.ylabel("Age")
    plt.xlabel("Estimated Salary")
    plt.legend()

    plt.show()

Analysis()


