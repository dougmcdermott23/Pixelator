from PIL import Image
from matplotlib import cm
import numpy as np
import Utils as utils
import sys

def Main(argv):
    # Parse command line for image information and convert image to numpy array
    image_name = argv[1]
    pixel_factor = int(argv[2])
    palette_size = int(argv[3])

    file_name = "images\{}".format(image_name)
    im = np.array(Image.open(file_name))

    print("Image data type: {}".format(im.dtype))
    print("Image number of dimensions: {}".format(im.ndim))
    print("Image shape: {}".format(im.shape))

    rows = im.shape[0]
    cols = im.shape[1]

    # First filter - Mean Filter
    im_filter_one = utils.MeanFilter(im, rows, cols, pixel_factor)

    # Second Filter - Use K-Means Clustering to limit palette size
    im_filter_two = utils.LimitPaletteSize(im_filter_one, rows, cols, palette_size)
    
    # Convert numpy array to image
    pil_file_name = "images\pixelate_{}".format(image_name)
    pil_img = Image.fromarray(im_filter_two)
    pil_img.save(pil_file_name)
    
if __name__ == "__main__":
    Main(sys.argv)