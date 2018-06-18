import sklearn;
import numpy as np;
import math;
from scipy import sparse;
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("cpsmall.txt")
    return data[0], data[1]


X, y = get_data();
row,col = X.shape;


X_array = sparse.csr_matrix.todense(X);
X_max = X_array.max();

for i in range(0,row):
    for j in range(0,col):
        X_array[i,j]=X_array[i,j]/(X_max);


maximum_y=max(y);
for i in range(0,len(y)):
    y[i] = y[i]/maximum_y;

w_0 = np.random.rand(col);
w_0 = np.matrix(w_0);
print("W is",w_0.shape)
def gradient_fix(X, y, w, Lambda):
    grad_f_of_w = Lambda * np.transpose(w)
    [rows,col] = X.shape;
    for i in range(X.shape[0]):
        y_i = float(y[i])
        x_i = X[i]
        temp = y_i * x_i
        denom = 1 + np.exp(-y_i * x_i * w)
        grad_f_of_w = np.add(grad_f_of_w,((1/denom - 1) * temp))
    return (1/rows)*grad_f_of_w

def gradient_logistic_regression(X,y,w,Lambda,error,step):
    grad_of_w = gradient_fix(X,y,w,Lambda);
    for iter in range(50):
        print(iter);
        g = gradient_fix(X,y,w,Lambda);
        q = np.sqrt(np.dot(g,np.transpose(g)));
        w = w - step*g;
    return w;

def log_regression(X_array,y,w_0,lamba):
    #print(X_array.shape);
    #print(y.shape);
    y=np.matrix(y);
    #print(w_0.shape);
    XX = X_array @ w_0.transpose();
    #print("XX,Shape",X_array);
    A = np.multiply(y.transpose(),XX);
    #print("A",A);
    M = 1/(1+np.exp(A));
    #print("M is",M.shape);
    #print("Y_Shape",y.shape);
    #print("X_shape",X_array.shape);
    P =  np.multiply(y.transpose() , X_array) ;
    #print("P is",P.shape);
    P = np.transpose(P);
    w = P @ M;
    #w=w_0;
    return w.T;

def log_grad(X_array,y,w,Lambda,step,e):
    grad_of_w = log_regression(X_array,y,w,Lambda);
    for iter in range(50):
        print(iter);
        g = log_regression(X_array,y,w,Lambda);
        #q = np.sqrt(np.dot(g,np.transpose(g)));
        [row,col]=X_array.shape;
        df=-(1/row)*g + w;
        w = w - step*df;
    return w;

print(X_array.shape,y.shape,w_0.shape);
p = log_grad(X_array,y,w_0,1,0.01,0.01);


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

print(X_array.shape);
print(len(y));
print(p.shape);
print("MSE is",msefunction(X_array,y,p));



#MSE is MSE is [[2.16784524]]