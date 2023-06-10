# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

from imutils.video import VideoStream
import imutils
from collections import deque
import numpy as np
import argparse
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

video = args["video"]
buffer = args["buffer"]

# define the lower and upper boundaries of the "color" ball in
# HSV color space, the initialize the list of tracked points
# green
colorLower = np.array([29, 86, 6])
colorUpper = np.array([64, 255, 255])

# blue
# colorLower = np.array([110,50,50])
# colorUpper = np.array([130,255,255])

# # yellow
# colorLower = np.array([20, 100, 100])
# colorUpper = np.array([30, 255, 255])

pts = deque(maxlen=buffer)

# if a video path was not supplied, grab the reference to the webcam
if video == None:
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(video)

# allow the camera or video file to warm up
time.sleep(1.0)

def get_frame(frame):
    # handle the frame from VideoCapture or VideoStream
    return frame[1] if video != None else frame

# start matplotlib to display the video feed 
frame = get_frame(vs.read())
imagePlot = plt.imshow(frame)

# loop replaces the while loop
def update(i):
    # grab the current frame
    frame = get_frame(vs.read())

    # if we are viewing a video and we did not grav a frame, 
    # the we have reached the end of the video
    if frame is None:
        return

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the ball color, then perform a series of dilataions and erosions
    # to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x,y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at last one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute
        # the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radious meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame, then update
            # the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(buffer / float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)
    
    # show the frame to our screen
    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagePlot.set_data(output)
    # hsv_to_rgb()

# closes the matplotlib window when pressed 'q'
def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

# initialize matplotlib to display video feed
anim = animation.FuncAnimation(plt.gcf(), update, interval=50)
anim._start()

# attach close window event
plt.gcf().canvas.mpl_connect("key_press_event", close)
plt.show()