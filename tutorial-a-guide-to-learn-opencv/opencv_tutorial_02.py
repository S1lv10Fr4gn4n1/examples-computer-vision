import argparse
import matplotlib.pyplot as plt
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="tetris_blocks.png", help="path to input image")
args = vars(ap.parse_args())
var_image = args["image"]

# load the input image
image = cv2.imread(var_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# convert image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap="gray")
plt.show()

# applying edge detection we can find the outlines of objects in images
edged = cv2.Canny(gray, 30, 150)
plt.imshow(edged, cmap="gray")
plt.show()

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(thresh, cmap="gray")
plt.show()

# find contours (outlines) of the foreground objects in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] # for opencv 3
output = image.copy()
# loop over the contours
for c in cnts:
    # draw each contours on the output image
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    plt.imshow(output)
    plt.show()

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
plt.imshow(output)
plt.show()

# we apply erosions to reduce the size of the foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
plt.imshow(mask, cmap="gray")
plt.show()

# dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
plt.imshow(mask, cmap="gray")
plt.show()

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(output)
plt.show()