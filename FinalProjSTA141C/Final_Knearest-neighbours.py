# coding: utf-8

# In[61]:


import time
import numpy as np

filter = np.genfromtxt('/home/sai/PycharmProjects/sta141c/filter_s3_1_b_new.csv', delimiter=",")
filter = np.delete(filter, (0), axis=0)
np.random.shuffle(filter)
train=filter[0:filter.shape[0]*7//10,:];
test=filter[filter.shape[0]*7//10:filter.shape[0],:];
P = train[:, 0:3]
q = train[:, 3]
R = test[:, 0:3]
s = test[:, 3]
q = (np.asmatrix(q)).T
s = (np.asmatrix(s)).T
pro = 10

# In[63]:


print(filter.shape)
print(train.shape)
print(test.shape)
print(P.shape)
print(q.shape)
print(R.shape)
print(s.shape)
samptestx = R[1:1001, ]
samptesty = s[1:1001, ]
print(samptestx.shape)
print(samptesty.shape)


# In[64]:


def go_nn(Xtrain, ytrain, Xtest, ytest, que):
    correct = 0
    for i in range(Xtest.shape[0]):  ## For all testing instances
        nowXtest = Xtest[i, :]
        print(i)
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0, :] - nowXtest)
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest - Xtrain[j, :])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        # print ("expected : ", ytrain[idx], " actual : ", ytest[i])
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:
            correct += 1
    que.put(correct)


# In[67]:


fl = []
l = ["{},{}".format(round((samptestx.shape[0] / pro) * i), round((samptestx.shape[0] / pro) * (i + 1))) for i in
     range(pro)]
for i in range(pro):
    lsub = (l[i].split(','))
    lsub = [int(i) for i in lsub]
    fl.append(lsub)

# In[ ]:


import multiprocessing as mp

que = mp.Queue()
procs = [
    mp.Process(target=go_nn, args=(P, q, samptestx[fl[i][0]:fl[i][1] + 1, ], samptesty[fl[i][0]:fl[i][1] + 1, ], que))
    for i in range(pro)]

start_time = time.time()
for p in procs:
    p.start()

# Exit the completed processes
for p in procs:
    p.join()

results = [que.get() for i in range(pro)]
acc = sum(results) / float(samptesty.shape[0])
print("Accuracy %lf Time %lf secs.\n" % (acc, time.time() - start_time))

