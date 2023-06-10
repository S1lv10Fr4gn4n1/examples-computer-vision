# https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import dlib
import cv2
import matplotlib.pyplot as plt
 
# Download the shape_predictor_68_face_landmarks.dat file from 
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", default="images/example_01.jpg",
	help="path to input image")
args = vars(ap.parse_args())

shape_predictor = args["shape_predictor"]
image_path = args["image"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(output)
plt.show()