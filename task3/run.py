import os
import utils
import algo
import tqdm

WORKING_DIR = os.path.dirname(__file__)
OUTPUT_FOLDER = os.path.join(WORKING_DIR, 'testOutput')


def process_images():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    images, names = utils.load_images()
    for i, img in tqdm.tqdm(enumerate(images)):
        utils.save_img(algo.binirize(img), names[i], OUTPUT_FOLDER)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
process_images()