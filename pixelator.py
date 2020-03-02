from PIL import Image
from matplotlib import cm
import numpy as np
import Utils as utils
import sys
import pdb

def Main(argv):
    if len(argv) < 5:
        print ("Parameters required: image_name, pixel_factor, smooth_iterations, palette_size")
        return

    # Parse command line for image information and convert image to numpy array
    image_name = argv[1]
    pixel_factor = int(argv[2])
    smooth_iterations = int(argv[3])
    palette_size = int(argv[4])

    file_name = "images\{}".format(image_name)
    im = np.array(Image.open(file_name))

    print("Image data type: {}".format(im.dtype))
    print("Image number of dimensions: {}".format(im.ndim))
    print("Image shape: {}".format(im.shape))

    rows = im.shape[0]
    cols = im.shape[1]

    if pixel_factor > rows or pixel_factor > cols:
        pixel_factor = min(rows, cols)

    # First filter - Mean Filter
    print ("Applying Mean Filter")
    im_filter_one = utils.MeanFilter(im, rows, cols, pixel_factor)

    # Second Filter - Smooth super-samples
    im_filter_two = np.copy(im_filter_one)
    for iter in range (0, smooth_iterations):
        print ("Applying Smooth Filter Iteration [{}]".format(iter+1))
        im_filter_two = utils.SmoothingFilter(im_filter_two)

    # Third Filter - Use K-Means Clustering to limit palette size
    im_filter_three = np.copy(im_filter_two)
    if palette_size > 0:
        print ("Applying Palette Size Filter")
        im_filter_three = utils.LimitPaletteSize(im_filter_two, palette_size)

    num_pixels_last_row = rows % pixel_factor
    num_pixels_last_col = cols % pixel_factor
    im_final = np.zeros((rows - num_pixels_last_row, cols - num_pixels_last_col, im.shape[2]), dtype=np.uint8)

    # Convert the new image size to the same size as the original image (approximatley)
    # Loop through each pixel of the new image
    for x in range (0, im_filter_three.shape[0]):
        for y in range (0, im_filter_three.shape[1]):
            # Copy the pixel, pixel_factor^2 times, to create a super-sample
            for i in range (0, pixel_factor):
                for j in range (0, pixel_factor):
                    im_final[x*pixel_factor + i, y*pixel_factor + j] = im_filter_three[x, y]
    
    # Convert numpy array to image
    pil_file_name = "images\pixelate_{}".format(image_name)
    pil_img = Image.fromarray(im_final)
    pil_img.save(pil_file_name)
    
if __name__ == "__main__":
    Main(sys.argv)