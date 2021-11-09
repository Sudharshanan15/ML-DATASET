import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv') 
dataset.isnull().any()
x = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X_1 = LabelEncoder() 
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1]) 
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from keras.models import Sequential
from keras.layers import Dense
model=Sequential() 
model.add(Dense(input_dim=11,init="random_uniform",activation="relu",output_dim=20)) 
model.add(Dense(init="random_uniform",activation="relu",output_dim=15)) 
model.add(Dense(init="random_uniform",activation="sigmoid",output_dim=1))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]) 
model.fit(x_train,y_train,batch_size=32,epochs=100)
y_pred=model.predict(x_test)
y_pred
y_pred=y_pred>0.5
y_pred
from sklearn.metrics import accuracy_score
print("Accuracy score",accuracy_score(y_test,y_pred)*100,"%")