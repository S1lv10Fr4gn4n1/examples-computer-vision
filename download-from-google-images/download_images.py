# USAGE
# python download_images.py --urls urls.txt --output images/santa

# https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

import argparse
import requests
import cv2
import os
from multiprocessing import Pool
import random
import string

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", default="urls.txt", help="path to file containing image URLs")
ap.add_argument("-o", "--output", default="images", help="path to output directory of images")
args = vars(ap.parse_args())

urls_path = args["urls"]
output_directory = args["output"]

# create directory if it doesn't exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
urls = open(urls_path).read().strip().split("\n")

def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def delete_non_image(image_path):
    # initialize if the image should be deleted or not
    delete = False

    # try to load the image
    try:
        image = cv2.imread(image_path)

        # if the image is `None` then we could not properly load it
        # from disk, so delete it
        if image is None:
            print("None")
            delete = True

    # if OpenCV cannot load the image then the image is likely
    # corrupt so we should delete it
    except:
        print("Except")
        delete = True

    # check to see if the image should be deleted
    if delete:
        print("[INFO] deleting {}".format(image_path))
        os.remove(image_path)

def download_image(url):
    try:
        # define file name and path
        file_name = random_string()
        p = os.path.sep.join([output_directory, "{}.jpg".format(file_name)])

        # try to download the image
        r = requests.get(url, timeout=60)

        # save the image to disk
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] downloaded: {}".format(p))

        # delete image if not valid
        delete_non_image(p)
    # handle if any exceptions are thrown during the download process
    except:
        print("[INFO] error downloading {}...skipping".format(p))

number_of_threads = os.cpu_count()
print("[INFO] downloading images using {} threads".format(number_of_threads))
pool = Pool(number_of_threads)
results = pool.map(download_image, urls)
pool.close()
pool.join()