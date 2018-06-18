import multiprocessing as mp
import numpy as np
import time
import cPickle;
fin = open("data_files.pl", "rb");
data = cPickle.load(fin);
Xtrain = data[0]
ytrain = data[1]
Xtest = data[2]
ytest = data[3]

def subProcess(Xtrain,ytrain,Xtest,ytest,shared_queue,min,max):
    acc = go_nn(Xtrain,ytrain,Xtest,ytest,min,max)
    shared_queue.put(acc)

def go_nn(Xtrain, ytrain, Xtest, ytest,min = 0, max = None):
    if max == None:
       max = Xtest.shape[0]
    correct =0
    #print("XShape",Xtest.shape[0]);
    for i in range(min,max): ## For all testing instances
        #print(i);
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest) 
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:  
            correct += 1
    acc = correct/float(Xtest.shape[0])
    return acc

start_time = time.time()
sharedQueue = mp.Queue()
coreCount = 4
delta = Xtest.shape[0] // 4
processes = list((mp.Process(target=subProcess,
                                 args=(Xtrain, ytrain, Xtest, ytest, sharedQueue, i * delta, (i + 1) * delta))for i in range(coreCount - 1)))
i = coreCount - 1
processes += [mp.Process(target=subProcess,
                             args=(Xtrain, ytrain, Xtest, ytest, sharedQueue, i * delta, Xtest.shape[0]))]

for p in processes:
    p.start()

for p in processes:
    p.join()

accuracy = 0

while sharedQueue.empty() == False:
     single = sharedQueue.get()
     accuracy += single

print("Multicore Accuracy %lf Time %lf secs.\n" % (accuracy, time.time() - start_time))

start_time = time.time()
acc = go_nn(Xtrain, ytrain, Xtest, ytest)

print("Single Core Accuracy %lf Time %lf secs.\n" % (acc, time.time() - start_time))
    # print("Difference in accuracy: %lf " % (acc - accuracy))
#print(start_time);
#acc = go_nn(Xtrain, ytrain, Xtest, ytest)
#print "Accuracy %lf Time %lf secs.\n"%(acc, time.time()-start_time)

