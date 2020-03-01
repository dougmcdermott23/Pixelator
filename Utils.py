from sklearn.cluster import KMeans
import numpy as np
import pdb

#############################################
# Helper Function for First Filter
# Use Mean Filtering technique to create super-samples
#############################################
def SetMean(im_filter_one, x, y, pixel_factor):
    mean = np.zeros(im_filter_one.shape[2])
    num_cells = pixel_factor ** 2

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            mean += im_filter_one[x+i, y+j]

    mean = np.divide(mean, num_cells)

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            im_filter_one[x+i, y+j] = mean

#############################################
# First Filter
#############################################
def MeanFilter(im, rows, cols, pixel_factor):
    im_filter_one = np.copy(im)

    num_pixels_last_row = rows % pixel_factor
    num_pixels_last_col = cols % pixel_factor

    done = 0

    for x in range (0, rows - num_pixels_last_row + 1):
        if done == 1:
            break
        for y in range (0, cols-num_pixels_last_col + 1):
            
            # Handles edge case for pixles in the bottom right corner and right hand side 
            if x + pixel_factor >= rows:
                if y + pixel_factor >= cols:
                    for i in range (0, rows-x):
                        for j in range (0, cols-y):
                            im_filter_one[x+i, y+j] = im_filter_one[x-1, y-1]
                    done = 1
                    break
                else:
                    for i in range (0, rows-x):
                        im_filter_one[x+i, y] = im_filter_one[x-1, y]
                    continue

            # Handles edge case for pixles on the bottom side of the image
            if y + pixel_factor >= cols:
                for j in range (0, cols-y):
                    im_filter_one[x, y+j] = im_filter_one[x, y-1]
                continue

            if x % pixel_factor == 0 and y % pixel_factor == 0:
                SetMean(im_filter_one, x, y, pixel_factor)

    return im_filter_one

#############################################
# Helper Function for Second Filter
# Use Mean Filtering on super-samples to smooth image
#############################################
def SmoothCell(im, im_filter_two, rows, cols, x, y, pixel_factor):
    cell_colour = np.zeros(im.shape[2], dtype=np.uint32)
    num_samples = 0

    for i in range (x-pixel_factor, x+pixel_factor+1, pixel_factor):
        if i < 0 or i >= rows:
            continue
        for j in range (y-pixel_factor, y+pixel_factor+1, pixel_factor):
            if j < 0 or j >= cols:
                continue
            cell_colour += im[i, j]
            num_samples += 1

    cell_colour = np.divide(cell_colour, num_samples)

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            im_filter_two[x+i, y+j] = cell_colour

#############################################
# Second Filter
#############################################
def SmoothingFilter(im, rows, cols, pixel_factor):
    im_filter_two = np.copy(im)

    num_pixels_last_row = rows % pixel_factor
    num_pixels_last_col = cols % pixel_factor

    for x in range (0, rows):
        for y in range (0, cols):
            if x < rows - num_pixels_last_row and y < cols - num_pixels_last_col:
                if x % pixel_factor == 0 and y % pixel_factor == 0:
                    SmoothCell(im, im_filter_two, rows, cols, x, y, pixel_factor)

    return im_filter_two  

#############################################
# Third Filter
#############################################
def LimitPaletteSize(im, rows, cols, num_clusters):
    im_filter_two = im.reshape(rows*cols, im.shape[2])

    print ("Starting K-Means Clustering")
    sorted_im = KMeans(n_clusters=num_clusters, n_init=1).fit(im_filter_two)

    print ("Cluster Centers:")
    print (sorted_im.cluster_centers_)

    im_filter_three = np.zeros(im_filter_two.shape, dtype=np.uint8)
    for i in range (0, im_filter_three.shape[0]):
        im_filter_three[i] = sorted_im.cluster_centers_[sorted_im.labels_[i]]

    im_filter_three = im_filter_three.reshape(rows, cols, im.shape[2])
    return im_filter_three

