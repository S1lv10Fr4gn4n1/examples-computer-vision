# https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/

from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Download the shape_predictor_68_face_landmarks.dat file from 
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
args = vars(ap.parse_args())

shape_predictor = args["shape_predictor"]
video_path = args["video"]

# initialize the frame counters and the total number of blicks
counter = 0
total = 0

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")

# if a video path was not supplied, grab the reference to the webcam
if video_path == None:
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(video_path)

time.sleep(2.0)

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

# start matplotlib to display the video feed 
frame = get_frame(vs.read())
imagePlot = plt.imshow(frame)

# loop replaces the while loop
def update(i):
    global counter, total

    # grab the current frame
    frame = get_frame(vs.read())

    # if we are viewing a video and we did not grav a frame, 
    # the we have reached the end of the video
    if frame is None:
        close_window()
        return

    # resize and convert to gray scale
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # detect faces in grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            counter += 1

        # otherwise, the eye aspect ratio is not below the blink threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if counter >= EYE_AR_CONSEC_FRAMES:
                total += 1
            
            # reset the eye frame counter
            counter = 0
        
        # draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagePlot.set_data(output)
    

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