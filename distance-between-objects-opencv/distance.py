# https://www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, default=24.257,
    help="width of the left-most object in the image (in millimeters)")
args = vars(ap.parse_args())

image_path = args["image"]
width = args["width"]

# load image file from disk, converto to gray scale and blur a little
image = cv2.imread(image_path)
image = imutils.resize(image, width=1280)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)


# perform edge detection, dilation + erosion to close the gaps between edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort contours from left-to-right
(cnts, _) = contours.sort_contours(cnts)

colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
refObj = None

# setup image display
fig, ax = plt.subplots()
fig.canvas.manager.window.showMaximized()
fig.show()

# loop over the contours
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue
    
    # compute the rotated bounding vox of the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # sort contours
    box = perspective.order_points(box)

    # compute the center of the bounding box
    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])

    # save object reference, which should be the first contour (left-most-object)
    if refObj is None:
        # unpack the ordered boundingbox and compute the midpoint
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints, 
        # then construct the reference object
        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        refObj = (box, (cX, cY), D / width)
        continue
    
    # draw the contours on the image
    orig = image.copy()
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

    # stack the reference coordinates and the object coordinates to include the object center
    refCoords = np.vstack([refObj[0], refObj[1]])
    objCoords = np.vstack([box, (cX, cY)])

    # loop over the original points
    for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
        # draw circles corresponding to the current points and
        # connect them with a line
        cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
        cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
        cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
 
        # compute the Euclidean distance between the coordinates,
        # and then convert the distance in pixels to distance in
        # units
        D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
        (mX, mY) = midpoint((xA, yA), (xB, yB))
        cv2.putText(orig, "{:.1f}mm".format(D), (int(mX), int(mY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
 
        # show the output image
        output = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        ax.imshow(output)
        fig.canvas.draw()
        plt.pause(1.0)
        ax.cla()