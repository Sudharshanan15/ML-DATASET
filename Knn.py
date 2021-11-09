import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ads.csv")
data.head()
data.isnull().any()

x = data.iloc[:,1:4].values 
y = data.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x[:,0] = lb.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x, y, test_size = 0.1,random_state=0)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier ( n_neighbors = 5 , p = 2 )
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

import sklearn.metrics as metrics
fpr, tpr ,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr,tpr)
roc_auc
plt.plot(fpr,tpr)
plt.xlim([0,1]) 
plt.ylim([0,1])
plt.style.use("dark_background")

# Knn works based on minimum distance from the query instance to the training samples to determine the k- nearest neighbour.
# After we gather k- nearest neighbour we take simple majority of these k- nearest neighbour to the prediction of the query instance.
# The data for knn generation consist of several multivariate attribute name that will be used to classify.
# It uses all of the data for training while classifying a new data point or instance.