# https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

import os
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def plot_images(image_1, image_2, cmap_2=None, bgr2rgb_1=True, bgr2rgb_2=False):
    if bgr2rgb_1:
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

    if bgr2rgb_2:
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title("Original")
    ax1.imshow(image_1)
    ax2.set_title("Processed")
    ax2.imshow(image_2, cmap=cmap_2)
    plt.show()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/test_01.png", help="path to the input image")
args = vars(ap.parse_args())

image_path = os.path.abspath(os.path.dirname(__file__)) + "/" + args["image"]

# define the answers key which maps the quetions number to the correct answer
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

# load the image, convert to graycale, blur it, find edges
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

# output_edged = cv2.drawContours(image.copy(), [edged], -1, (0, 255, 0), 2)

# plot curren results
plot_images(image, edged, cmap_2="gray")

# find the contours (outlines) in the edge map, 
# then initialize the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) == 0:
    print("Contour (outline) not found")
    exit()

# sort the contours according to their size
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over sorted contour
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    # if your approximated contour has four points,
    # the we can assume we have found the paper
    if len(approx) == 4:
        docCnt = approx
        break

output_image = cv2.drawContours(image.copy(), [docCnt], -1, (0, 255, 0), 3)
plot_images(image, output_image, bgr2rgb_2=True)

# apply a four point perspective transform to both the original image
# and grayscale image to obtain a top-down birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))    
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# plot curren results
plot_images(image, paper, bgr2rgb_2=True)

# apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# plot curren results
plot_images(image, thresh, cmap_2="gray")

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionsCnts = []

for c in cnts:
    # compute the bounding box of the contour, the use the 
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region should be sufficientluy wide,
    # sufficiently tall and have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionsCnts.append(c)


# plot curren results
output_image = cv2.drawContours(paper.copy(), questionsCnts, -1, (0, 255, 0), 3)
plot_images(image, output_image, bgr2rgb_2=True)

# sort the question contours top-top-bottom, the initialize
# the total number of correct answers
questionsCnts = contours.sort_contours(questionsCnts, method="top-to-bottom")[0]
correct = 0

# each questions has 5 opssible answers, to loop over the questions in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionsCnts), 5)):
    # sort the contours for the current question from left to right
    # then initialize the index of the bubbled answer
    cnts = contours.sort_contours(questionsCnts[i:i+5])[0]
    bubbled = None

    # plot curren results
    output_image = cv2.drawContours(paper.copy(), cnts, -1, (0, 255, 0), 3)
    plot_images(image, output_image, bgr2rgb_2=True)

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the curernt 
        # bubble for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -2)

        # apply the mask to the thresholded image, then count the number
        # of non-zero pixels in the bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        # if the current total has a larger number of total non-zero pixels,
        # the we are examining the currently bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    
    # initialize the contour color and the index of the "correct" answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    # check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # draw the outline of the corect answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

output_contour = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
plot_images(image, output_contour)