from sklearn.cluster import KMeans
import numpy as np

#############################################
# Helper Function for First Filter
# Use Mean Filtering technique to create super-samples
#############################################

def SetMean(im_filter_one, x, y, pixel_factor):
    mean = np.zeros(im_filter_one.shape[2])
    numCells = pixel_factor ** 2

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            mean += im_filter_one[x+i, y+j]

    mean = np.divide(mean, numCells)

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            im_filter_one[x+i, y+j] = mean

#############################################
# First Filter
#############################################

def MeanFilter(im, rows, cols, pixel_factor):
    im_filter_one = np.copy(im)

    for x in range (0, rows):
        for y in range (0, cols):
            
            # Handles edge case for pixles on the right hand side of the image
            if x + pixel_factor > rows:
                for i in range (0, rows-x):
                    im_filter_one[x+i, y] = im_filter_one[x-1, y]
                continue

            # Handles edge case for pixles on the bottom side of the image
            if y + pixel_factor > cols:
                for j in range (0, cols-y):
                    im_filter_one[x, y+j] = im_filter_one[x, y-1]
                continue

            if x % pixel_factor == 0 and y % pixel_factor == 0:
                SetMean(im_filter_one, x, y, pixel_factor)

    return im_filter_one

#############################################
# Second Filter
#############################################

def LimitPaletteSize(im, rows, cols, num_clusters):
    im_filter_one = im.reshape(rows*cols, im.shape[2])

    print ("Starting K-Means Clustering")
    sorted_im = KMeans(n_clusters=num_clusters, n_init=1).fit(im_filter_one)

    print ("Cluster Centers:")
    print (sorted_im.cluster_centers_)

    im_filter_two = np.zeros(im_filter_one.shape, dtype=np.uint8)
    for i in range(0, im_filter_two.shape[0]):
        im_filter_two[i] = sorted_im.cluster_centers_[sorted_im.labels_[i]]

    im_filter_two = im_filter_two.reshape(rows, cols, im.shape[2])
    return im_filter_two