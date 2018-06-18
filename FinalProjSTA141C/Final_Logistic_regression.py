import sklearn;
import time;
import numpy as np;
import multiprocessing as mp;
import math;
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy import sparse;
filter = np.genfromtxt('/home/sai/PycharmProjects/sta141c/filter_nexus1_b_new.csv', delimiter=",")
#filter=np.delete(filter, (0), axis=0)

#print((filter.shape[0]*7)//10);

#print(filter.shape);

train=filter[0:filter.shape[0]*7//10,:];
test=filter[filter.shape[0]*7//10:filter.shape[0],:];

trainX=train[:,0:3]
trainy=train[:,3]
testX=test[:,0:3]
testy=test[:,3]

def log_regression(X_array,y,w_0,lamba):
   y=np.matrix(y);
   w_0 = np.matrix(w_0);
   #print(w_0.shape);
   XX = X_array.dot(w_0.transpose());
   #print("XX,Shape",X_array.shape);
   A = np.multiply(y.transpose(),XX);
   M = 1/(1+np.exp(A));
   P=sparse.csr_matrix(y.transpose()).multiply(X_array)
   #print("P is",P.shape);
   P = np.transpose(P);
   w = P.dot(M);
   return w.T;

def log_grad(X_array,y,w,Lambda,step,e):
   grad_of_w = log_regression(X_array,y,w,Lambda);
   for iter in range(200):
       g = log_regression(X_array,y,w,Lambda);
       [row,col]=X_array.shape;
       df=-(1/row)*g + w;
       w = w - step*df;
   return w;


#print(X_train.shape,y_train.shape,w_0.shape);
col = trainX.shape[1];
w_0 = np.random.rand(col)
p = log_grad(trainX,trainy,w_0,1,0.01,0.001)



def prediction(X,w):
   return X.dot(w);


def calculate_acc(X_test,y_test,p):
   Y_Test = prediction(X_test, p.transpose());
   F_S1 = 1 / (1 + np.exp(-Y_Test));
   r1 = len(F_S1);
   acc=0;
   y_001 = np.zeros(r1);
   k=0
   l=0
   for j in range(1,7):
     if(j==1):
       k=2
       l=1
     elif(j==2):
       k=3
       l=2
     elif(j==3):
       k=4
       l=3
     elif(j==4):
       k=5
       l=4
     elif(j==5):
       k=6
       l=5
     else:
       k=1
       l=6
     y1_test = np.zeros(len(y_test));
     for q in range(len(y_test)):
        if(y_test[q] == l):
          y1_test[q] = l
        else:
          y1_test[q] = k

     for i in range(0, r1):
         if F_S1[i] > 0.5:
             y_001[i]= k;
         else:
             y_001[i] = l;

     correct_prediction = 0;
     r = len(y1_test);
     for i in range(0, r):
         if y_001[i] == y1_test[i]:
             correct_prediction = correct_prediction + 1;
     acc =acc + (float(correct_prediction)/float(r))*100
   return acc/6;


print(calculate_acc(testX,testy,p));

