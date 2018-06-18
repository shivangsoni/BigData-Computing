import numpy as np
import time
import pickle;

def ed(a,b):
    return np.linalg.norm(a-b)

def assign_cluster(a,centers):
    dists = np.array([ed(a,x) for x in centers])
    return [np.argmin(dists),np.min(dists)]

def cluster_mean(cluster):
    return np.mean(cluster,axis=0)


def kmeans_plotting(points):
    start_time = time.time()
    [row,col] = points.shape
    init_centers= np.zeros((10,col))
    init_centers[0:10,:] = points[0:10,:]
    centers = init_centers
    for m in range(40):
        if (m!=0 and m%10 == 0):
            print("The time taken for iterating from 0","Iteration to ",m,"iteration is ",time.time()-start_time)
        objective = 0
        clusters = [[] for i in range(10)]
        for i in range(points.shape[0]):
            a = points[i,:]
            c,mindist = assign_cluster(a,centers)
            objective += mindist**2
            clusters[c].append(a)

        print(objective)
        centers = []
        for i in range(10):
            centers.append(cluster_mean(clusters[i]))
    print("The time taken for iterating from 0 Iteration to 40 iteration is" , time.time() - start_time)

def main():
    fin = open("data_dense.pl", "rb");
    data = pickle.load(fin, encoding="latin1")
    points = data
    kmeans_plotting(points)


if __name__ == "__main__":
    main()
