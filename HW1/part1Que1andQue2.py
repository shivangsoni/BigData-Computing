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


############   Ask about minimum  ##########
def computeCostGradient(X_array , w_0 , lambd , y):
    X = np.array(X_array);
    XT = np.transpose(X);
    [row,col] = X.shape;
    return np.dot(XT,(np.dot(X,w_0) - y))  + lambd*w_0;


def gradient_descend(X,y,step_size,e,w_0,Lambda):
    w = w_0.transpose();
    #print("W shape",w.shape);
    #print("X shape",X.shape);
    #rint("Y shape",y.shape);
    y=np.matrix(y);
    #print("Y shape",y.shape)
    p = np.transpose(X).dot(X.dot(w) - y)+(Lambda*w);
    #print(p.shape);
    r_0 = np.sqrt(np.transpose(p).dot(p));
    print(r_0);
    for i in range(0,10):
        print(i);
        g=np.transpose(X).dot(X.dot(w) - y)+Lambda*w;
        if(np.sqrt(np.transpose(g).dot(g))<= e*r_0):
            break;
        r_0=np.sqrt(np.transpose(g).dot(g));
        print("Function,");
        print(X.dot(w) - y);
        w = w - step_size*g;

    return w.transpose();

def gradient_fix(X, y, step_size, stop_cond, init_soln,Lambda):
    w = init_soln.transpose()
    Lambda = 1.0
    y =  np.matrix(y).transpose()
    [row,col] =X.shape;
    grad_f_of_w =  (1/(row))*np.transpose(X).dot(X.dot(w) - y)+(Lambda * w);

    r_0 = np.sqrt(np.transpose(grad_f_of_w).dot(grad_f_of_w));

    for i in range(200):
        g = (1/(row))*np.transpose(X).dot(X.dot(w) - y) + (Lambda * w);
        if (np.sqrt(np.transpose(g).dot(g)) <= (stop_cond * r_0)):
            break
        r_0 = np.sqrt(np.transpose(g).dot(g));
        w = np.subtract(w , float(step_size) * g)
        break
    return w

def gradient_fix1(X, y, step_size, init_soln,Lambda=1):
    w = init_soln.transpose()
    Lambda = 1.0
    y =  np.matrix(y).transpose()
    [row,col] =X.shape;
    grad_f_of_w =  (1/(row))*np.transpose(X).dot(X.dot(w) - y)+(Lambda * w)

    for i in range(200):
        g = (1/(row))*np.transpose(X).dot(X.dot(w) - y) + (Lambda * w)
        w = np.subtract(w , float(step_size) * g)
        break
    return w


def gradient_descent_fixed(X_array,w_0,y,lambd,er0,neta):
    w = w_0;
    r0 = computeCostGradient(X_array,w,lambd,y);
    r0 = abs(np.sqrt(np.transpose(r0).dot(r0)));
    #print(r0);
    for i in range(200):
        g = computeCostGradient(X_array,w,lambd,y);
        q = np.sqrt(np.dot(np.transpose(g),g));
        if abs(q) < er0*r0:
            break
        r0=q;
        w = np.subtract(w, neta*g);
    return w;
####################

###################
def ComputeOptimizedW(X_array,w_0,y,lambd,er0,neta):
    p = np.transpose(X_array).dot(X_array);
    #print(p.shape);
    [row,col] = X_array.shape;
    r = lambd*np.identity(col);
    q = np.linalg.inv(p+r);
    #print("q is",q.shape);
    #print(X_array.shape);
    #print("y is",y);
    z=np.transpose(X_array).dot(y);

    w = np.dot(z,q);
    return w;


X, y = get_data();
row,col = X.shape;
#print(row,col);

maximum_y=max(y);


for i in range(0,len(y)):
    y[i] = y[i]/(maximum_y);
X_array = sparse.csr_matrix.todense(X);
print("y",y);
X_max = X_array.max();

for i in range(0,row):
    for j in range(0,col):
        X_array[i,j]=X_array[i,j]/(X_max);
#print("X_array",X_array);
w_0 = np.random.rand(col);
w_0 = np.matrix(w_0);





#p=gradient_fix(X_array,y,0.001,0.001,w_0,1);
################ Working ########
#p= ComputeOptimizedW(X_array,w_0,y,1,0.001,0.001)

#for i in range(0,row):
#    print(i);
#    for j in range(0,col):
#        X_array[i,j] = float(X_array[i,j]/max(X_array[:,j]));

#for i in range(0,len(y)):
#    y[i]=y[i]/max(y);

#####################################################
################CROSS VALIDATION ####################
###############For X#################################
[row,col] = X_array.shape;
FirstFold = np.matrix(X_array[0:math.ceil(row/5),:]);
FirstFoldy = np.array(y[0:math.ceil(row/5)]);
SecondFold = np.matrix(X_array[math.ceil(row/5):2*math.ceil(row/5),:]);
SecondFoldy = np.array(y[math.ceil(row/5):2*math.ceil(row/5)]);
ThirdFold = np.matrix(X_array[2*math.ceil(row/5):3*math.ceil(row/5),:]);
ThirdFoldy = np.array(y[2*math.ceil(row/5):3*math.ceil(row/5)])
FourthFold = np.matrix(X_array[3*math.ceil(row/5):4*math.ceil(row/5),:]);
FourthFoldy = np.array(y[3*math.ceil(row/5):4*math.ceil(row/5)]);
FifthFold =np.matrix(X_array[4*math.ceil(row/5):row,:]);
FifthFoldy = np.array(y[4*math.ceil(row/5):row])


def msefunction(X,Y,W):
    p = np.dot(X,W);
    Y=np.matrix(Y);
    error = p-Y.transpose();
    #print(error.shape);
    [row,col] = X.shape;
    sum = 0;
    for i in range(row):
        sum=sum+error[i]*error[i];

    return sum/row;

def mse_new(X,y,w):
    return np.mean(np.square(X @ w - y))
############################################
##################Testing###################
####################
#print(mse_new(X_test,y_test,test));
#############################################
##############################################
##############################################

MSE=[];
for i in range(0,5):
    if(i == 0):
        test = FirstFold;
        testy = FirstFoldy;
        train = np.concatenate((SecondFold,ThirdFold,FourthFold,FifthFold));
        #train = np.matrix(train);
        trainy =np.concatenate((SecondFoldy,ThirdFoldy,FourthFoldy,FifthFoldy));
        #trainy=np.array(trainy);
    elif (i ==1):
        test = SecondFold;
        testy = SecondFoldy;
        train = np.concatenate((FirstFold,ThirdFold,FourthFold,FifthFold));
        #train = np.matrix(train);
        trainy = np.concatenate((FirstFoldy, ThirdFoldy, FourthFoldy, FifthFoldy));
        #trainy = np.array(trainy);
    elif (i ==2):
        test = ThirdFold;
        train = np.concatenate((FirstFold,SecondFold,FourthFold,FifthFold));
        testy = ThirdFoldy;
        #train = np.matrix(train);
        trainy = np.concatenate((FirstFoldy,SecondFoldy,FourthFoldy,FifthFoldy));
        #trainy = np.array(trainy);
    elif (i ==3):
        test = FourthFold;
        train = np.concatenate((FirstFold,SecondFold,ThirdFold,FifthFold));
        testy = FourthFoldy;
        #train = np.matrix(train);
        trainy = np.concatenate((FirstFoldy,SecondFoldy,ThirdFoldy,FifthFoldy));
        #trainy = np.array(trainy);
    else :
        test = FifthFold;
        train = np.concatenate((FirstFold,SecondFold,ThirdFold,FourthFold));
        testy = FifthFoldy;
        #train = np.matrix(train);
        trainy = np.concatenate((FirstFoldy, SecondFoldy, ThirdFoldy, FourthFoldy));


    w_train = gradient_fix1(train, trainy, 0.01, w_0, 1);
    #print("MMMMMMMMMMMMM",test.shape);
    #print(testy.shape);
    print("w_train",w_train.transpose());
    result = mse_new(test,testy,w_train)
    #print(result);
    MSE.append(result)

finalMSE=0;
for i in range(0,5):
    finalMSE = finalMSE +MSE[i]
finalMSE = finalMSE/5
print("MSE is",finalMSE)




######################################################################
######################################################################
######################################################################
###########Now load the data from bz2 and report the findings#########

#MSE for step size 10^-7->1.4368499105835195
#MSE for step size 10^-6->2.1972458199928995
#MSE for step size 10^-5->1.9230778724356319
#MSE for step size 10^-4->0.8627778009614399
#MSE for step size 10^-3->0.7128257199290408
#MSE for step size 10^-2->1.2247898962886785


#Final MSE after 5 fold is for 10^-4 is: 0.29486