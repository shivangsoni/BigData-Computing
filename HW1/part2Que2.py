import sklearn;
import time;
import cPickle;
import numpy as np;
import multiprocessing as mp;
import math;
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy import sparse;
#mem = Memory("./mycache5")

#@mem.cache


def get_data():
    fin = open("data_files.pl","rb")
    data = cPickle.load(fin)
    return data

data = get_data();
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
X_train = data[0];
y_train = data[1];
X_test = data[2];
y_test = data[3];

y_train =  np.matrix(y_train).transpose() # 15996 x 1
y_test = np.matrix(y_test).transpose() # 4000 x 1
[row,col] = X_train.shape;
w_0 = np.random.rand(col);
w_0 = np.matrix(w_0);
#
#Xtrain (15996, 1355191)
#Xtest (4000, 1355191)
#Y test (4000, 1)
#Y train (15996, 1)
#W (1, 1355191)
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
    grad_of_w = gradient_fix(X,y,w.transpose(),Lambda);
    #print(grad_of_w);
    for iter in range(10):
        print(iter);
        g = gradient_fix(X,y,w.transpose(),Lambda);
        w = w - step*(g+w);
        print(w);
    return w;

#def logistic_reg(X,y,w,Lambda,error,step):
def log_regression(X_array,y,w_0,lamba):
    #print(X_array.shape);
    #print(y.shape);
    y=np.matrix(y);
    #print(w_0.shape);
    XX = X_array.dot(w_0.transpose());
    #print("XX,Shape",X_array.shape);
    A = np.multiply(y.transpose(),XX);
    M = 1/(1+np.exp(A));
    #print("M is",M.shape); #15996,1
    #print("Y_Shape",y.shape);
    #print("X_shape",X_array.shape);
    #P =  np.multiply(y.transpose(), X_array);
    P=sparse.csr_matrix(y.transpose()).multiply(X_array)
    #print("P is",P.shape);
    P = np.transpose(P);
    w = P.dot(M);
    return w.T;

def log_grad(X_array,y,w,Lambda,step,e,min = 0,max = None):
    if(max == None):
        max = X_array.shape[0];
    #print(X_array.shape);
    #print(y.shape);
    #print(w.shape);
    grad_of_w = log_regression(X_array[min:max,:],y[:,min:max],w,Lambda);
    for iter in range(200):
        #print(iter);
        g = log_regression(X_array[min:max,:],y[:,min:max],w,Lambda);
        #q = np.sqrt(np.dot(g,np.transpose(g)));
        [row,col]=X_array[min:max,:].shape;
        df=-(1/row)*g + w;
        w = w - step*df;
    return w;


#print(X_train.shape,y_train.shape,w_0.shape);

#p = gradient_logistic_regression(X_train,y_train,w_0,1,0.01,0.01);

start_time = time.time()
p = log_grad(X_train,y_train.transpose(),w_0,1,0.01,0.01);

def prediction(X,w):
    return X.dot(w);


def calculate_acc(X_text,y_test,p):
    Y_Test = prediction(X_test, p.transpose());

    F_S = 1 / (1 + np.exp(-Y_Test));
    r = len(F_S);
    y_00 = np.zeros(r);
    # print(F_S);

    for i in range(0, r):
        if F_S[i] > 0.5:
            y_00[i] = 1;
        else:
            y_00[i] = -1;

    F_S1 = 1 / (1 + np.exp(-y_test));
    r1 = len(F_S1);
    #print(r1, r)
    y_001 = np.zeros(r1);

    for i in range(0, r1):
        if F_S1[i] > 0.5:
            y_001[i] = 1;
        else:
            y_001[i] = -1;

    correct_prediction = 0;
    for i in range(0, r):
        if y_00[i] == 1:
            correct_prediction = correct_prediction + 1;

    return (correct_prediction/r)*100;

Y_Test =prediction(X_test,p.transpose());

F_S = 1/(1+np.exp(-Y_Test));
r=len(F_S);
y_00 = np.zeros(r);
#print(F_S);

for i in range(0,r):
    if F_S[i] > 0.5:
         y_00[i] = 1;
    else:
        y_00[i] = -1;


F_S1 = 1/(1+np.exp(-y_test));
r1=len(F_S1);
#print(r1,r)
y_001 = np.zeros(r1);

for i in range(0,r1):
    if F_S1[i] > 0.5:
         y_001[i] = 1;
    else:
        y_001[i] = -1;

def mse_new(X,y,w):
    return np.mean(np.square(X.dot(w) - y));

correct_prediction = 0;
for i in range(0,r):
    if y_00[i] == 1:
        correct_prediction = correct_prediction + 1;
print("For Single processor");
print("Prediction Accuracy is",(correct_prediction/r)*100);
print("time passed",time.time()-start_time,"Seconds");

def subProcess(Xtrain, ytrain,w_0, Xtest, ytest, shared_queue, min, max):
    """

    :type shared_queue: mp.Queue
    """
    p = log_grad(Xtrain,ytrain.transpose(),w_0,1,0.01,0.01,min,max);
    acc = calculate_acc(Xtest,ytest,p)
    shared_queue.put(acc)

sharedQueue = mp.Queue()
coreCount = 4
delta = X_train.shape[0] // 4;
start_time = time.time();
processes = list((mp.Process(target=subProcess,
                             args=(X_train, y_train,w_0 ,X_test, y_test, sharedQueue, i * delta, (i + 1) * delta))
                  for i in range(coreCount - 1)))
i = coreCount - 1
processes += [mp.Process(target=subProcess,
                         args=(X_train, y_train, w_0,X_test, y_test, sharedQueue, i * delta, X_train.shape[0]))]

for p in processes:
    p.start()

for p in processes:
    p.join()

accuracy = 0;

while sharedQueue.empty() == False:
    single = sharedQueue.get()
    accuracy += single

print("Multicore Accuracy %lf Time %lf secs.\n" % (accuracy/4, time.time() - start_time))

