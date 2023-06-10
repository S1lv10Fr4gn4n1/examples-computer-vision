import cv2
import sys

s = 0

if len(sys.argv) > 1:
    s = int(sys.argv[1])

# Where to download the model and checking the parameters
# https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml


# starting model
net = cv2.dnn.readNetFromCaffe(
    'models/deploy.prototxt',
    'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# Model parameters
in_width = 300
in_height = 300
mean = (104, 177, 123)
conf_threshold = 0.7

source = cv2.VideoCapture(s)
winName = "Camera Preview"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # run the model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow(winName, frame)

source.release()
cv2.destroyWindow(winName)