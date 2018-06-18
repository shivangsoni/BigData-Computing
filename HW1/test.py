import sklearn;
import numpy as np;
import math;
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy import sparse;
mem = Memory("./mycache5")

@mem.cache
def get_data():
    data = load_svmlight_file("news20.binary.bz2")
    return data[0], data[1]


X, y = get_data();
row,col = X.shape;
X, y = get_data("news20.binary.bz2")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print("Xtrain",X_train.shape);
print("Xtest",X_test.shape);

y_train =  np.matrix(y_train).transpose() # 15996 x 1
y_test = np.matrix(y_test).transpose() # 4000 x 1
print("Y test",y_test.shape);
print("Y train",y_train.shape);
w_0 = np.random.rand(col);
w_0 = np.matrix(w_0);

print("W",w_0.shape);
def msefunction(X,Y,W):
    p = np.dot(X,W.transpose());
    print(p.shape);
    y = np.matrix(Y);
    print(y.shape);
    error = p-y.transpose();
    [row,col] = X.shape;
    sum = 0;
    print(error.shape);
    for i in range(row):
        sum=sum+error[i]*error[i];
    return sum/row;
print(X_test.shape);
print(y_test.shape);

def mse_new(X,y,w):
    return np.mean(np.square(X @ w - y));

print("The Mse is:",mse_new(X_test,y_test,w_0.transpose()));