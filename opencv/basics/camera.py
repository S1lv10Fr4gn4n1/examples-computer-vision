import cv2
import sys

s = 0

if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv2.VideoCapture(s)
output = cv2.VideoWriter('videos/camera_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))

winName = "Camera Preview"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break

    cv2.imshow(winName, frame)
    output.write(frame)

source.release()
output.release()
cv2.destroyWindow(winName)
