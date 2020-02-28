from PIL import Image
from matplotlib import cm
import numpy as np
import KMeansClustering as kmeans
import sys
import pdb

def SetMean(im, x, y, pixelFactor):
    mean = np.zeros(3)
    numCells = pixelFactor ** 2

    for i in range (0, pixelFactor):
        for j in range (0, pixelFactor):
            mean += im[x+i, y+j]

    mean = np.divide(mean, numCells)

    for i in range (0, pixelFactor):
        for j in range (0, pixelFactor):
            im[x+i, y+j] = mean


def MeanFilter(im, rows, cols, pixelFactor):
    for x in range (0, rows):
        for y in range (0, cols):
            
            if x + pixelFactor > rows:
                for i in range (0, rows-x):
                    im[x+i, y] = im[x-1, y]
                continue

            if y + pixelFactor > cols:
                for j in range (0, cols-y):
                    im[x, y+j] = im[x, y-1]
                continue

            if x % pixelFactor == 0 and y % pixelFactor == 0:
                SetMean(im, x, y, pixelFactor)

def Main(argv):
    imageName = argv[1]
    pixelFactor = int(argv[2])
    paletteSize = int(argv[3])

    fileName = "images\{}".format(imageName)
    im = np.array(Image.open(fileName))

    print("Image data type: {}".format(im.dtype))
    print("Image number of dimensions: {}".format(im.ndim))
    print("Image shape: {}".format(im.shape))

    rows = im.shape[0]
    cols = im.shape[1]

    MeanFilter(im, rows, cols, pixelFactor)

    # pixIm = im.reshape(rows*cols, 3)
    # pixIm = kmeans.SortData(pixIm, paletteSize)
    # print (pixIm) # NEED TO CONVERT THIS FROM FLOAT TO UINT8
    # pixIm = pixIm.reshape(rows, cols, 3)

    pil_fileName = "images\pixelate_{}".format(imageName)
    pil_img = Image.fromarray(im)
    pil_img.save(pil_fileName)
    
if __name__ == "__main__":
    Main(sys.argv)