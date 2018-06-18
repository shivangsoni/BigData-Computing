import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
filter = np.genfromtxt ('/home/sai/PycharmProjects/sta141c/filter_nexus_1_c_new.csv', delimiter=",")
filter=np.delete(filter, (0), axis=0)
np.random.shuffle(filter)
train=filter[0:filter.shape[0]*7//10,:];
test=filter[filter.shape[0]*7//10:filter.shape[0],:];
P=train[:,0:3]
q=train[:,3]
R=test[:,0:3]
s=test[:,3]
s=(np.asmatrix(s)).T

print(P.shape)
print(q.shape)
model = XGBClassifier()
model.fit(P, q)

ypred = model.predict(R)
predictions = [round(value) for value in ypred]

accuracy = accuracy_score(s, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
