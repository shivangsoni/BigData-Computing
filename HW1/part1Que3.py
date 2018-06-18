from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file


mem = Memory("./mycache4")

@mem.cache
def get_data(data):
    my_data = load_svmlight_file(data)
    return my_data[0], my_data[1]

X_train, y_train = get_data("E2006.train.bz2") # X_train: 16087 x 150360
X_test, y_test = get_data("E2006.test.bz2")

y_test = np.matrix(y_test)
y_test = np.transpose(y_test)


X_train = X_train[:, :-2]
y_train = np.matrix(y_train)
y_train = np.transpose(y_train)



def gradient_fix1(X, y, step_size, init_soln,Lambda=1):
    w = init_soln
    Lambda = 1.0
    y =  np.matrix(y)
    [row,col] =X.shape;
    grad_f_of_w =  (1/(row))*np.transpose(X).dot(X.dot(w) - y)+(Lambda * w)

    for i in range(200):
        g = (1/(row))*np.transpose(X).dot(X.dot(w) - y) + (Lambda * w)
        w = np.subtract(w , float(step_size) * g)
        break
    return w


def msefunction(X,Y,W):
    p = np.dot(W,X.transpose());
    #print(p.shape);
    error = p-Y;
    [row,col] = X.shape;
    sum = 0;
    for i in range(row):
        sum=sum+error[:,i]*error[:,i];
    return sum/row;


def mse_new(X,y,w):
    return np.mean(np.square(X @ w - y))


[row,col] = X_train.shape;
w_0 = np.random.randn(col) #
w_0 = np.matrix(w_0).transpose()
w = gradient_fix1(X_train,y_train, 0.0001,w_0,1);
print("MSE is",mse_new(X_test,y_test,w));
#MSE is 0.659665822743878