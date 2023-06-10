# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# import necessary packages
import argparse
import matplotlib.pyplot as plt
from face_detector import FaceDetector

# construct the argument parse and parse the arguments
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", 
        default="image_1.jpg",
        help="path to input image")
    ap.add_argument("-p", "--prototxt", 
        default="deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", 
        default="res10_300x300_ssd_iter_140000.caffemodel",
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    return args["image"], args["prototxt"], args["model"], args["confidence"]

image, prototxt, model, min_confidence = parse_arguments()

fd = FaceDetector(prototxt, model, min_confidence)
image_fd = fd.detect(plt.imread(image))

plt.imshow(image_fd)
plt.show()