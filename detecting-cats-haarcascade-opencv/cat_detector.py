# https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/

import argparse
import cv2
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())

# load file from disk and convert to gray scale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the cat detector Haar cascade and then detect cat faces
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))

# loop over the cat faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(image, "Cat #{}".format(i+1), (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.title("Cat faces")
plt.imshow(output)
plt.show()