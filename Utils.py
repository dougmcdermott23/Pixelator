import colorsys
from sklearn.cluster import KMeans
import numpy as np
import pdb

#############################################
# Helper Function for First Filter
# Use Mean Filtering technique to create super-samples
#############################################
def SetMean(im, x, y, pixel_factor):
    mean = np.zeros(im.shape[2])
    num_cells = pixel_factor ** 2

    for i in range (0, pixel_factor):
        for j in range (0, pixel_factor):
            mean += im[x+i, y+j]

    mean = np.divide(mean, num_cells)

    return mean

#############################################
# First Filter
#############################################
def MeanFilter(im, rows, cols, pixel_factor):
    filter_one_rows = int(rows/pixel_factor)
    filter_one_cols = int(cols/pixel_factor)
    im_filter_one = np.zeros((filter_one_rows, filter_one_cols, im.shape[2]), dtype=np.uint8)

    num_pixels_last_row = rows % pixel_factor
    num_pixels_last_col = cols % pixel_factor

    for x in range (0, rows - num_pixels_last_row):
        for y in range (0, cols-num_pixels_last_col):
            if x % pixel_factor == 0 and y % pixel_factor == 0:
                i = int(x / pixel_factor)
                j = int(y / pixel_factor)
                im_filter_one[i, j] = SetMean(im, x, y, pixel_factor)

    return im_filter_one

#############################################
# Helper Function for Second Filter
# Use Mean Filtering on super-samples to smooth image
#############################################
def SmoothCell(im, rows, cols, x, y):
    cell_colour = np.zeros(im.shape[2], dtype=np.uint32)
    num_samples = 0

    for i in range (x-1, x+2):
        if i < 0 or i >= rows:
            continue
        for j in range (y-1, y+2):
            if j < 0 or j >= cols:
                continue
            cell_colour += im[i, j]
            num_samples += 1

    cell_colour = np.divide(cell_colour, num_samples)

    return cell_colour

#############################################
# Second Filter
#############################################
def SmoothingFilter(im):
    im_filter_two = np.copy(im)

    rows = im.shape[0]
    cols = im.shape[1]

    for x in range (0, rows):
        for y in range (0, cols):
            cell_colour = SmoothCell(im, rows, cols, x, y)
            im_filter_two[x, y] = cell_colour

    return im_filter_two  

#############################################
# Third Filter
#############################################
def LimitPaletteSize(im, num_clusters):
    rows = im.shape[0]
    cols = im.shape[1]

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

#############################################
# Fourth Filter
#############################################
def Saturation(im, saturation):
    im_filter_four = np.copy(im)

    rows = im.shape[0]
    cols = im.shape[1]

    for x in range (0, rows):
        for y in range (0, cols):
            cell_colour = np.array([im[x, y, 0], im[x, y, 1], im[x, y, 2]])
            cell_colour = np.divide(cell_colour, 255)

            hsv = colorsys.rgb_to_hsv(cell_colour[0], cell_colour[1], cell_colour[2])

            new_saturation = hsv[1] + saturation
            new_saturation = min(new_saturation, 1)
            new_saturation = max(new_saturation, 0)

            rgb = np.array(colorsys.hsv_to_rgb(hsv[0], new_saturation, hsv[2]))
            rgb = rgb*255

            im_filter_four[x, y, 0] = rgb[0]
            im_filter_four[x, y, 1] = rgb[1]
            im_filter_four[x, y, 2] = rgb[2]

    return im_filter_four