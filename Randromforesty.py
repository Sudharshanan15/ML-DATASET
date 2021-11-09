import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("ads.csv")
dataset.head()
dataset.isnull().any()
x = dataset.iloc[:, 1:4].values 
y = dataset.iloc[:, 4].values
x[:5]
y[:5]
from sklearn.preprocessing import LabelEncoder 
lb=LabelEncoder() 
x[:,0]=lb.fit_transform(x[:,0]) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier(n_estimators=10000,criterion='entropy')
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test,y_pred)*100,"%")
from sklearn.metrics import confusion_matrix 
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred) 
roc_auc = metrics.auc(fpr, tpr) 
print("AUC:",roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()