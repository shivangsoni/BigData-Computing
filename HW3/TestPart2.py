import pickle;
import scipy;
import sklearn;
import time;
import numpy as np;
import math;
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy import sparse;


def cluster_mean(cluster):
    return np.mean(cluster,axis = 0)


def kmeans_plotting(points):
    [row,col] = points.shape
    init_centers= np.zeros((10,col))
    init_centers[0:10,:] = points[0:10,:].todense()
    centers = init_centers
    start_time = time.time()
    for m in range(40):
        if (m!=0 and m%10 == 0):
            print("The time taken for iterating from 0","Iteration to ",m,"iteration is ",time.time()-start_time)
        objective = 0
        clusters = [[] for i in range(10)]
        B = [0]*10
        for k in range(10):
            b = centers[k]
            B[k] = np.linalg.norm(b) ** 2

        for i in range(points.shape[0]):
            distance = [0]*10
            a = points[i,:]
            A = scipy.sparse.linalg.norm(a) ** 2
            for k in range(10):
                b = centers[k]

                TWOAB = 2 * (a.dot(b.T))
                B_sq = B[k]
                euc_dist = A + B_sq - TWOAB

                distance[k] = euc_dist

            objective += np.min(distance)
            c = np.argmin(distance)
            clusters[c].append(i)
        print(objective)
        centers = []
        for i in range(10):
             centers.append(cluster_mean(points[clusters[i],:].todense()))
    print("The time taken for iterating from 0 Iteration to 40 iteration is", time.time() - start_time)

def main():
    fin = open("data_sparse_E2006.pl", "rb");
    data = pickle.load(fin, encoding="latin1")
    points = data
    kmeans_plotting(scipy.sparse.csr_matrix(points))


if __name__ == "__main__":
    main()