import numpy as np
import scipy.stats as stats
import operator
import functools
import sys
import random
import matplotlib.pyplot as plt

infile = sys.argv[1]
k = int(sys.argv[2]) #number of clusters specified by user

data = np.loadtxt(infile) #load in the file given by the user using numpy

n = data.shape[0] #data points number
d = data.shape[1] #data points degree

centroids = data[np.random.randint(0, n, k)] #choose k random points
m = centroids.shape[0] #centroid points number

def calc_eucl_dist(a, b):
    return np.linalg.norm(a-b)


classes = np.zeros((n,1), dtype = data.dtype) #create array of purely zeros. Same length as data

i = 0
while(i < 20):
    classesExt = np.append(classes, data, axis = 1) #new array with classes as the first column leaving data alone

    clusters = [ [] for x in range(k)]  #list of lists to hold the clusters
	
	#find nearest centroid for each data point
    for x in range(n):
        distances = []
        for y in range(m):
            distances.append(calc_eucl_dist(data[x], centroids[y]))
        classesExt[x][0] = np.argmin(distances)

    newCentroids = np.zeros((k, d), dtype = centroids.dtype) # ndarray same dimensions as centroids

	#create new clusters and compute their centroids
    for x in range(k):
        clusters[x] = classesExt[classesExt[:,0] == x]
        clusters[x] = clusters[x][:, 1:]
        newCentroids[x] = np.mean(clusters[x][:,:], axis =0)

    
    sigma = sum(calc_eucl_dist(centroids[x], newCentroids[x]) for x in range(k)) #find distance all centroids moved
    np.copyto(centroids, newCentroids) #update centroids
    print(str(sigma))
    if(sigma < 0.001): break
	
    i += 1

print("Number of Iterations: " + str(i))
print("Final means: ")
print(data[d-1])
print("Number of clusters: " + str(k))

#only plot 2d data
if(d == 2):
    for cluster in clusters:
        plt.plot(cluster[:, 0], cluster[:, 1], '.')
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], 'o', markersize = 15)
    plt.axis('equal')
    plt.show()