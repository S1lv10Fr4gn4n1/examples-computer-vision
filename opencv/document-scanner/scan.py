# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/page.jpg",
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load image and get attributes
image = cv2.imread(args["image"])
orig = image.copy()
(h, w, _) = image.shape
ratio = h / 500.0
new_shape = (int(w/ratio), int(h/ratio))
print("new shape", new_shape)
image = cv2.resize(image, new_shape)

# convert image to grayscale, blur it and find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.set_title("Original")
ax1.imshow(image)
ax2.set_title("Edged")
ax2.imshow(edged, cmap="gray")
plt.show()

# find the outlines in the edged image, keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] # opencv3
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, 
    # then we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
    
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
plt.imshow(image)
plt.show()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

# convert the warped image to grayscale then threshold it 
# to give it that black and white paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# show original and scanned images
print("STEP 3: Apply perspective transform")
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.set_title("Original")
ax1.imshow(cv2.resize(orig, new_shape))
ax2.set_title("Scanned")
ax2.imshow(cv2.resize(warped, new_shape), cmap="gray")
plt.show()