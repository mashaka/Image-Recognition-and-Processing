import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from image_pyramid import ImagePyramid, ImagePyramidLayer

WORKING_DIR = os.path.dirname(__file__)
TEST_FOLDER = os.path.join(WORKING_DIR, 'binarizationTestSet')
TIFF_SUFFIX = '.tiff'


def open_img(name):
    img = Image.open(name)
    return img, np.array(img.convert('RGB'), dtype='int64')


def load_images():
    images = []
    names = []
    for f in os.listdir(TEST_FOLDER):
        if os.path.isfile(os.path.join(TEST_FOLDER, f)):
            names.append(f)
            images.append(open_img(os.path.join(TEST_FOLDER, f)))
    return images, names


def save_img(img_arr, name, folder):
    img = Image.fromarray(img_arr).convert('1')
    img.save(os.path.join(folder, name[:-4] + TIFF_SUFFIX))