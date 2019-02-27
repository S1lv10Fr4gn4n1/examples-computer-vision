# https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import json
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())

video_path = args["video"]
min_area = args["min_area"]

# if a video path was not supplied, grab the reference to the webcam
if video_path == None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(video_path)

def get_frame(frame):
    # handle the frame from VideoCapture or VideoStream
    return frame[1] if video_path != None else frame

def finish_video_stream():
    if video_path is None:
        vs.stop()
    else:
        vs.release()

def close_window():
    finish_video_stream()
    plt.close()

# initialize the first frame in the video stream
firstFrame = None

# start matplotlib to display the video feed 
frame = get_frame(vs.read())
imagePlot = plt.imshow(frame)

def update(i):
    global firstFrame

    # grab the current frame
    frame = get_frame(vs.read())
    text = "Unoccupied"

    # if we are viewing a video and we did not grav a frame, 
    # the we have reached the end of the video
    if frame is None:
        close_window()
        return
    
    # resize the frame, convert to grayscale and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        return
    
    # compute the absolute difference between the current frame and the first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes,
    # then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        
        # computer the bounding box for the contour, draw it on the frame and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
    
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagePlot.set_data(output)
    # imagePlot.set_data(thresh)
    # imagePlot.set_data(frameDelta)

# closes the matplotlib window when pressed 'q'
def close(event):
    if event.key == 'q':
        finish_video_stream()
        plt.close(event.canvas.figure)

# initialize matplotlib to display video feed
anim = animation.FuncAnimation(plt.gcf(), update, interval=50)
anim._start()

# attach close window event
plt.gcf().canvas.mpl_connect("key_press_event", close)
plt.show()