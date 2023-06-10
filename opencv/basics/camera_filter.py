import cv2
import sys
import numpy as np

s = 0

if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv2.VideoCapture(s)
# output = cv2.VideoWriter('videos/camera_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))

winName = "Camera Preview"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

feature_params = dict(maxCorners=500,
                    qualityLevel=0.2,
                    minDistance=15,
                    blockSize=9)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)
    # frame = cv2.blur(frame, (15, 15))
    # frame = cv2.Canny(frame, 80, 150)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
    if corners is not None:
        for x, y in np.int32(corners).reshape(-1, 2):
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 1)

    cv2.imshow(winName, frame)
    # output.write(frame)

source.release()
# output.release()
cv2.destroyWindow(winName)
