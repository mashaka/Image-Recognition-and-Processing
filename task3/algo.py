import numpy as np
from image_pyramid import ImagePyramid


def calc_luminance(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def calc_img_luminance(img):
    luminance_arr = np.zeros(img.shape[:2])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            luminance_arr[y][x] = calc_luminance(img[y][x])
    return luminance_arr      
    

def binirize(img):
    # Get luminocity representation of image
    img_lum = calc_img_luminance(img[1])
    # Build mean, min and max pyramid
    img_pyramid = ImagePyramid(img_lum)
    # Go down the pyramid and calculate thresholds
    img_pyramid.calc_thresholds()
    return img_pyramid.get_processed_img()
