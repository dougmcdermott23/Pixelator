from PIL import Image
import numpy as np

imageAddress = 'images\matterhorn.jpg'
im = np.array(Image.open(imageAddress))

print(im.dtype)

print(im.ndim)

print(im.shape)