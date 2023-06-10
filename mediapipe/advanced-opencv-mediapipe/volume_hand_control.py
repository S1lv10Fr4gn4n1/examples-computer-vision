import cv2
import mediapipe as mp
import time
import numpy as np
import math
from subprocess import Popen

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


def set_volume(volume):
    volume_cmd = "set volume output volume {}".format(volume)
    Popen(['osascript', '-e', volume_cmd])


while cv2.waitKey(1) != 27:  # Escape
    has_frame, img = cap.read()
    if not has_frame:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        h, w, c = img.shape

        x1, y1 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
        x2, y2 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [15, 210], [0, 100])
        print(length, vol)
        set_volume(vol)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
