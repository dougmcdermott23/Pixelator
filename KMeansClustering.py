from sklearn.cluster import KMeans
import numpy as np
import pdb

def SortData(im, numClusters):
    print ("Starting K-Means Clustering")
    sortedIm = KMeans(n_clusters=numClusters).fit(im)

    print ("Cluster Centers:")
    print (sortedIm.cluster_centers_)
    print ("Labels:")
    print (sortedIm.labels_)

    newIm = np.zeros(im.shape)
    for i in range(0, im.shape[0]):
        newIm[i] = sortedIm.cluster_centers_[sortedIm.labels_[i]]

    return newIm