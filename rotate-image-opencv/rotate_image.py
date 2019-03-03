# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image file")
args = vars(ap.parse_args())

# load image from disk
image_orig = cv2.imread( args["image"])

# setup image display
fig, ax = plt.subplots()
fig.show()

def display_image(image, title, timeout=0.1):
    ax.imshow(image)
    fig.canvas.draw()
    plt.title(title)
    plt.pause(timeout)
    ax.cla()

def rotate(image, angle, w, h):
    # define the image center
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def rotate_bound(image, angle, w, h):
    # Extra sources
    # https://stackoverflow.com/questions/11764575/python-2-7-3-opencv-2-4-after-rotation-window-doesnt-fit-image

    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

    # define the image center
    (cX, cY) = (w // 2, h // 2)

    # grab the matrix rotation and the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs([M[0, 1]])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def show_rotation_problem(image):
    (h, w) = image.shape[:2]

    angle_steps = 15
    # rotate image and display the problem
    for angle in np.arange(0, 360, angle_steps):
        rotated = rotate(image, angle, w, h)
        display_image(rotated, "Rotation with problem")

    # rotate image and display in bound
    for angle in np.arange(0, 360, angle_steps):
        rotated = rotate_bound(image, angle, w, h)
        display_image(rotated, "Correct rotation")

# convert to gray scale, blur with Gaussian and get contours with Canny
gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 20, 100)

# find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# check if has at least one contour
if len(cnts) > 0:
    # grab the largest contour and draw a mask fir the pill
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute its bounding box of pill, then extract the RIO and apply mask
    (x, y, w, h) = cv2.boundingRect(c)
    imageROI = image_orig[y:y + h, x:x + w]
    maskROI = mask[y:y + h, x:x + w]
    imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
    
    show_rotation_problem(imageROI)
else:
    show_rotation_problem(image_orig)