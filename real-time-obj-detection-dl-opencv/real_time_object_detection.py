# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
 help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel", 
 help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

prototxt_path = args["prototxt"]
model_path = args["model"]
min_confidence = args["confidence"]
video_path = args["video"]

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("[INFO] starting video stream...")
# if a video path was not supplied, grab the reference to the webcam
if video_path == None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(video_path)

# allow the cammera sensor to warmup, and initialize the FPS counter
time.sleep(2.0)
fps = FPS().start()

def get_frame(frame):
    # handle the frame from VideoCapture or VideoStream
    return frame[1] if video_path != None else frame

# start matplotlib to display the video feed 
frame = get_frame(vs.read())
imagePlot = plt.imshow(frame)

def finish_video_stream():
    if video_path is None:
        vs.stop()
    else:
        vs.release()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def close_window():
    finish_video_stream()
    plt.close()

def update(i):
    # grab the current frame
    frame = get_frame(vs.read())

    # if we are viewing a video and we did not grav a frame, 
    # the we have reached the end of the video
    if frame is None:
        close_window()
        return

    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > min_confidence:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagePlot.set_data(output)
    
    # update the FPS counter
    fps.update()

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